import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

import cv2
import torch
import numpy as np
from collections import deque
import threading
import queue
import mediapipe as mp
from ultralytics import YOLO
from src.model.bigru_model import AffectiveBiGRU

# Limit PyTorch CPU thread pool to avoid resource contention on mobile P/E cores
torch.set_num_threads(2)
torch.set_num_interop_threads(1)

# Multi-strategy MediaPipe Face Mesh importer
try:
    import mediapipe.python.solutions.face_mesh as mp_face_mesh
except (ModuleNotFoundError, AttributeError):
    try:
        import mediapipe.solutions.face_mesh as mp_face_mesh
    except (ModuleNotFoundError, AttributeError):
        try:
            mp_face_mesh = mp.solutions.face_mesh
        except AttributeError:
            raise ImportError("[ERROR] MediaPipe solutions module failed to load. Reinstall mediapipe in your environment.")


class AffectivePerceptionEngine:
    """
    Optimized Multimodal Perception Engine for Affect-Drive Cockpit.
    - Decoupled asynchronous Road Hazard extraction (frame-skipping queue)
    - Lightweight MediaPipe FaceMesh (refine_landmarks=False, tracking enabled)
    - Direct class-filtering YOLOv8n inference on downsampled (640x480) streams
    - Buffer-capped OpenCV VideoCapture (CAP_PROP_BUFFERSIZE=1) to prevent lag
    - Fast PyTorch inference under torch.inference_mode()
    """
    def __init__(
        self,
        driver_src=0,
        road_src="data/sample_road.mp4",
        model_path="data/bigru_affective.pth",
        seq_len=30,
        skip_road_frames=2,
        target_resolution=(640, 480)
    ):
        self.seq_len = seq_len
        self.skip_road_frames = max(1, skip_road_frames)
        self.target_w, self.target_h = target_resolution
        self.frame_area = self.target_w * self.target_h
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.frame_counter = 0

        # 1. Capture initialization with latency caps & resolution constraints
        self.driver_cap = cv2.VideoCapture(driver_src)
        self.driver_cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_w)
        self.driver_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_h)
        self.driver_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.road_cap = cv2.VideoCapture(road_src)
        self.road_cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_w)
        self.road_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_h)
        self.road_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # 2. Optimized Vision Models
        # Pre-filter target classes directly in YOLO: 0=Person, 2=Car, 3=Motorcycle, 5=Bus, 7=Truck
        self.target_classes = [0, 2, 3, 5, 7]
        self.yolo = YOLO("yolov8n_int8_openvino_model/yolov8n.xml")

        # Disable iris refinement for ~10ms/frame speedup; enable tracking confidence
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # 12 Distinct Key Facial Landmarks (3D -> 36 features):
        # Eyes: [33, 133, 362, 263], Mouth: [61, 291, 0, 17], Face Contour/Orientation: [10, 152, 1, 199]
        self.LANDMARK_IDS = [33, 133, 362, 263, 61, 291, 0, 17, 10, 152, 1, 199]

        # 3. Asynchronous Road Inference Worker Queues
        self.road_input_queue = queue.Queue(maxsize=1)
        self.road_result_queue = queue.Queue(maxsize=1)
        self.worker_running = True
        self.road_worker_thread = threading.Thread(target=self._road_inference_worker, daemon=True)
        self.road_worker_thread.start()

        self.cached_road_feats = [0, 0.0, 0.0, 0.0, 0.0]

        # 4. Temporal Brain (PyTorch BiGRU)
        self.model = AffectiveBiGRU(input_size=41, hidden_size=64, num_classes=3).to(self.device)
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"[FUSION ENGINE] Successfully loaded PyTorch BiGRU weights from {model_path}")
        except Exception as e:
            print(f"[FUSION ENGINE WARNING] Could not load weights ({e}). Running with initialized model.")
        self.model.eval()

        # 5. Temporal Feature Rolling Buffer & Spatial Gaze State
        self.feature_buffer = deque(maxlen=self.seq_len)
        self.label_map = {0: "NORMAL", 1: "ANXIOUS", 2: "DISTRACTED"}
        self.face_lost_counter = 0
        self.distraction_hold_frames = 0

    def _extract_driver_features(self, frame):
        """
        Extracts 36-dimensional driver face vector and real-time head pose / gaze metrics.
        Returns: (feats_list, face_detected, is_looking_away)
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            feats = []
            for idx in self.LANDMARK_IDS:
                lm = landmarks[idx]
                feats.extend([round(lm.x, 4), round(lm.y, 4), round(lm.z, 4)])

            # Real-time spatial Head Pose & Gaze Asymmetry calculation:
            # Landmark 33 (Left Eye), Landmark 263 (Right Eye), Landmark 1 (Nose Tip)
            # Landmark 10 (Forehead Top), Landmark 152 (Chin Bottom)
            p_left = landmarks[33]
            p_right = landmarks[263]
            p_nose = landmarks[1]
            p_top = landmarks[10]
            p_chin = landmarks[152]

            d_left = abs(p_nose.x - p_left.x)
            d_right = abs(p_right.x - p_nose.x)
            yaw_ratio = d_left / (d_right + 1e-6)

            face_height = abs(p_chin.y - p_top.y)
            pitch_ratio = (p_nose.y - p_top.y) / (face_height + 1e-6)

            # Looking far left, far right, or looking down (e.g. phone)
            is_head_turned = (yaw_ratio < 0.40) or (yaw_ratio > 2.50)
            is_looking_down = pitch_ratio > 0.68

            is_looking_away = is_head_turned or is_looking_down
            return feats, True, is_looking_away

        return [0.0] * (len(self.LANDMARK_IDS) * 3), False, True

    def _extract_road_features(self, frame):
        """Extracts 5-dimensional road hazard vector via YOLOv8n."""
        results = self.yolo.predict(
            frame,
            verbose=False,
            conf=0.40,
            iou=0.5,
            classes=self.target_classes,
            device="cpu",
            imgsz=self.target_w
        )[0]

        frame_h, frame_w = frame.shape[:2]
        frame_area = frame_h * frame_w if (frame_h * frame_w) > 0 else self.frame_area
        boxes = []

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)
            cx = ((x1 + x2) / 2.0) / frame_w
            cy = ((y1 + y2) / 2.0) / frame_h
            boxes.append((area / frame_area, cx, cy))

        if boxes:
            boxes.sort(key=lambda x: x[0], reverse=True)
            closest = boxes[0]
            hazard_count = len(boxes)
            has_close = 1.0 if closest[0] > 0.18 else 0.0
            return [hazard_count, round(closest[0], 4), round(closest[1], 4), round(closest[2], 4), has_close]

        return [0, 0.0, 0.0, 0.0, 0.0]

    def _road_inference_worker(self):
        """Background thread executing road detection without stalling camera acquisition."""
        while self.worker_running:
            try:
                frame = self.road_input_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if frame is None:
                break

            road_feats = self._extract_road_features(frame)

            # Keep only the freshest result in the queue
            if not self.road_result_queue.empty():
                try:
                    self.road_result_queue.get_nowait()
                except queue.Empty:
                    pass
            self.road_result_queue.put(road_feats)
            self.road_input_queue.task_done()

    def step(self):
        """Executes a single optimized perception cycle."""
        ret_d, d_frame = self.driver_cap.read()
        ret_r, r_frame = self.road_cap.read()

        # Seamless looping for test videos
        if not ret_r:
            self.road_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret_r, r_frame = self.road_cap.read()

        if not ret_d or not ret_r:
            return None, None, None

        # Ensure standard resolution
        if d_frame.shape[1] != self.target_w or d_frame.shape[0] != self.target_h:
            d_frame = cv2.resize(d_frame, (self.target_w, self.target_h), interpolation=cv2.INTER_LINEAR)
        if r_frame.shape[1] != self.target_w or r_frame.shape[0] != self.target_h:
            r_frame = cv2.resize(r_frame, (self.target_w, self.target_h), interpolation=cv2.INTER_LINEAR)

        self.frame_counter += 1

        # 1. Driver Feature Extraction (Spatial Landmarks & Head Pose)
        d_feats, face_detected, is_looking_away = self._extract_driver_features(d_frame)

        if not face_detected:
            self.face_lost_counter += 1
        else:
            self.face_lost_counter = 0

        # 2. Async Road Hazard Dispatch (Decoupled on every Nth frame)
        if self.frame_counter % self.skip_road_frames == 0:
            if self.road_input_queue.empty():
                self.road_input_queue.put(r_frame.copy())

        # Retrieve latest completed road inference (or reuse cached)
        try:
            self.cached_road_feats = self.road_result_queue.get_nowait()
        except queue.Empty:
            pass

        r_feats = self.cached_road_feats

        # 3. Build 41-dimensional multimodal feature vector
        frame_vector = d_feats + r_feats
        self.feature_buffer.append(frame_vector)

        # 4. Temporal BiGRU Inference
        bigru_status = "NORMAL"
        if len(self.feature_buffer) == self.seq_len:
            tensor_input = torch.as_tensor(
                [list(self.feature_buffer)],
                dtype=torch.float32,
                device=self.device
            )

            with torch.inference_mode():
                logits = self.model(tensor_input)
                pred_idx = torch.argmax(logits, dim=1).item()
                bigru_status = self.label_map.get(pred_idx, "NORMAL")

        # 5. Hybrid Spatial-Temporal Driver State Fusion:
        # Distraction is triggered when:
        # - Head turned left/right or looking down (is_looking_away == True)
        # - Face disengaged / looking away for multiple frames (face_lost_counter >= 5)
        # - BiGRU temporal network classifies sequence as DISTRACTED
        if is_looking_away or (self.face_lost_counter >= 5) or (bigru_status == "DISTRACTED"):
            driver_status = "DISTRACTED"
            self.distraction_hold_frames = 10  # Hold for smooth UI visibility
        elif self.distraction_hold_frames > 0:
            self.distraction_hold_frames -= 1
            driver_status = "DISTRACTED"
        elif bigru_status == "ANXIOUS":
            driver_status = "ANXIOUS"
        else:
            driver_status = "NORMAL"

        # 6. Multimodal Risk Decision Fusion
        has_hazard = r_feats[4] == 1.0
        system_risk = "HIGH" if (driver_status in ["DISTRACTED", "ANXIOUS"] or has_hazard) else "LOW"

        state_payload = {
            "driver_state": {"driver_status": driver_status},
            "road_state": {"high_risk_hazard": "Close Vehicle/Pedestrian" if has_hazard else "Clear"},
            "system_risk_level": system_risk
        }

        # 7. High-Visibility HUD Overlay on Driver Feed
        if driver_status == "DISTRACTED":
            cv2.rectangle(d_frame, (5, 5), (420, 42), (0, 0, 180), -1)
            cv2.putText(d_frame, "BiGRU: DISTRACTED [Eyes Off Road]", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        elif driver_status == "ANXIOUS":
            cv2.rectangle(d_frame, (5, 5), (420, 42), (0, 140, 220), -1)
            cv2.putText(d_frame, "BiGRU: ANXIOUS [Stress Detected]", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        else:
            cv2.rectangle(d_frame, (5, 5), (320, 42), (0, 80, 0), -1)
            cv2.putText(d_frame, "BiGRU: NORMAL [Attentive]", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        return d_frame, r_frame, state_payload

    def stop(self):
        """Clean shutdown of hardware captures and background threads."""
        self.worker_running = False
        if self.road_input_queue.empty():
            self.road_input_queue.put(None)
        self.driver_cap.release()
        self.road_cap.release()