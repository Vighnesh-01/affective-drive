import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

import time
import threading
from collections import deque
import cv2
import torch
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

from src.model.bigru_model import AffectiveBiGRU
from src.perception.emotion_detector import FacialEmotionDetector
from src.perception.cabin_behavior import CabinBehaviorDetector

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
            raise ImportError(
                "[ERROR] MediaPipe solutions module failed to load. Reinstall mediapipe in your environment."
            )


class AffectivePerceptionEngine:
    """
    Multimodal Driver Affect, In-Cabin Behavior & Road Context Engine.

    Key Capabilities:
    - Real-time FACS Facial Emotion Recognition (Neutral, Happy, Sad, Angry, Surprise, Fear, Drowsy)
    - Explicit NO FACE DETECTED handling when webcam is covered/closed or driver leaves view
    - In-Cabin Driver Behavior Detection: Seatbelt (ON/OFF), Talking, Eating/Drinking, Phone Use, Yawning
    - Real-time YOLOv8n Road Vehicle & Pedestrian Detection with live bounding-box overlays
    - Seamless looping car-cabin video mode + live webcam switching
    - Temporal BiGRU fusion under torch.inference_mode()
    """

    def __init__(
        self,
        driver_src="data/sample_driver.mp4",
        road_src="data/sample_road.mp4",
        model_path="data/bigru_affective.pth",
        seq_len=30,
        skip_road_frames=2,
        target_resolution=(640, 480),
        show_landmarks=True,
    ):
        self.driver_src = driver_src
        self.road_src = road_src
        self.seq_len = seq_len
        self.skip_road_frames = max(1, skip_road_frames)
        self.target_w, self.target_h = target_resolution
        self.frame_area = self.target_w * self.target_h
        self.show_landmarks = show_landmarks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.frame_counter = 0

        # 1. Capture initialization
        self.driver_is_file = self._is_video_file(driver_src)
        self.road_is_file = self._is_video_file(road_src)
        self.cap_lock = threading.Lock()

        self.driver_cap = self._open_capture(driver_src)
        self.road_cap = self._open_capture(road_src)

        # 2. Vision Models
        # Road classes: 0=Person, 2=Car, 3=Motorcycle, 5=Bus, 7=Truck
        self.road_target_classes = [0, 2, 3, 5, 7]
        self.cabin_target_classes = CabinBehaviorDetector.CABIN_TARGET_IDS
        self.yolo = YOLO("yolov8n.pt")

        # MediaPipe FaceMesh (468 landmarks for head pose + FACS emotions)
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # 12 Key Facial Landmarks (3D -> 36 features for BiGRU)
        self.LANDMARK_IDS = [33, 133, 362, 263, 61, 291, 0, 17, 10, 152, 1, 199]

        # Facial Emotion Detector & In-Cabin Behavior Detector
        self.emotion_detector = FacialEmotionDetector(smoothing_window=12)
        self.cabin_behavior = CabinBehaviorDetector(mar_window=16, seatbelt_window=25)

        # 3. Single Background YOLO Worker (Thread-safe for both Road & Cabin detections)
        self._yolo_lock = threading.Lock()
        self._pending_road_frame = None
        self._pending_cabin_frame = None
        self._worker_wakeup = threading.Event()
        self.worker_running = True

        self.cached_road_feats = [0, 0.0, 0.0, 0.0, 0.0]
        self.cached_road_boxes = []
        self.cached_cabin_boxes = []
        self._cache_lock = threading.Lock()

        self.yolo_worker_thread = threading.Thread(target=self._yolo_inference_worker, daemon=True)
        self.yolo_worker_thread.start()

        # 4. Temporal Brain (PyTorch BiGRU)
        self.model = AffectiveBiGRU(input_size=41, hidden_size=64, num_classes=3).to(self.device)
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"[FUSION ENGINE] Loaded PyTorch BiGRU weights from {model_path}")
        except Exception as e:
            print(f"[FUSION ENGINE WARNING] Could not load BiGRU weights ({e}). Using initialized weights.")
        self.model.eval()

        # 5. Temporal Buffers & State Tracking
        self.feature_buffer = deque(maxlen=self.seq_len)
        self.label_map = {0: "NORMAL", 1: "ANXIOUS", 2: "DISTRACTED"}
        self.face_lost_counter = 0
        self.distraction_hold_frames = 0
        self.looking_away_streak = 0
        self.fps_counter = 0
        self.fps_start = time.time()
        self.current_fps = 0.0

    @staticmethod
    def _is_video_file(src) -> bool:
        return isinstance(src, str) and not src.isdigit() and src != "0"

    def _open_capture(self, src):
        parsed = str(src) if self._is_video_file(src) else (int(src) if str(src).isdigit() else src)
        cap = cv2.VideoCapture(parsed)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_h)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def switch_driver_source(self, new_src):
        """Hot-swaps the driver camera/video source without tearing down models."""
        with self.cap_lock:
            if self.driver_cap is not None:
                self.driver_cap.release()
            self.driver_src = new_src
            self.driver_is_file = self._is_video_file(new_src)
            self.driver_cap = self._open_capture(new_src)

        self.face_lost_counter = 0
        self.looking_away_streak = 0
        self.distraction_hold_frames = 0
        self.feature_buffer.clear()
        self.emotion_detector.reset()
        self.cabin_behavior.reset()
        with self._cache_lock:
            self.cached_cabin_boxes = []
        mode_str = f"Video ({new_src})" if self.driver_is_file else f"Live Camera ({new_src})"
        print(f"[FUSION ENGINE] Switched driver feed to: {mode_str}")

    def _extract_driver_features(self, bgr_frame):
        """
        Extracts 36-dim landmark vector, head pose, facial emotion, and raw 468 landmarks.
        Returns: (feats, face_detected, is_looking_away, emotion, emotion_scores, metrics, landmarks, rgb_frame)
        """
        # Detect completely black/covered camera lens early
        if float(np.mean(bgr_frame)) < 6.0:
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            return (
                [0.0] * (len(self.LANDMARK_IDS) * 3),
                False,
                False,
                "none",
                {},
                {"ear": 0.0, "mar": 0.0},
                None,
                rgb_frame,
            )

        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            feats = []
            for idx in self.LANDMARK_IDS:
                lm = landmarks[idx]
                feats.extend([round(lm.x, 4), round(lm.y, 4), round(lm.z, 4)])

            # Head Pose & Gaze Orientation
            p_left = landmarks[33]
            p_right = landmarks[263]
            p_nose = landmarks[1]
            p_top = landmarks[10]
            p_chin = landmarks[152]

            eye_center_x = (p_left.x + p_right.x) / 2.0
            eye_center_y = (p_left.y + p_right.y) / 2.0
            eye_width = abs(p_right.x - p_left.x) + 1e-6
            face_height = abs(p_chin.y - p_top.y) + 1e-6

            norm_yaw = (p_nose.x - eye_center_x) / eye_width
            norm_pitch = (p_nose.y - eye_center_y) / face_height

            # Calibrated distraction bounds so normal slight turns stay NORMAL
            is_head_turned = abs(norm_yaw) > 0.58
            is_looking_down = norm_pitch > 0.45
            is_looking_away = is_head_turned or is_looking_down

            # Real Facial Emotion Classification via FACS landmarks
            emotion, emotion_scores, metrics = self.emotion_detector.predict(landmarks)
            metrics["yaw"] = round(norm_yaw, 3)
            metrics["pitch"] = round(norm_pitch, 3)

            return feats, True, is_looking_away, emotion, emotion_scores, metrics, landmarks, rgb_frame

        return (
            [0.0] * (len(self.LANDMARK_IDS) * 3),
            False,
            False,
            "none",
            {},
            {"ear": 0.0, "mar": 0.0},
            None,
            rgb_frame,
        )

    def _extract_road_features(self, frame):
        """Extracts 5-dim road hazard vector AND bounding boxes via YOLOv8n."""
        results = self.yolo.predict(
            frame,
            verbose=False,
            conf=0.30,
            iou=0.50,
            classes=self.road_target_classes,
            device="cpu",
            imgsz=self.target_w,
        )[0]

        frame_h, frame_w = frame.shape[:2]
        frame_area = frame_h * frame_w if (frame_h * frame_w) > 0 else self.frame_area
        boxes_meta = []
        draw_boxes = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = self.yolo.names.get(cls_id, str(cls_id))
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = max(0, (x2 - x1) * (y2 - y1))
            area_ratio = area / frame_area
            cx = ((x1 + x2) / 2.0) / max(frame_w, 1)
            cy = ((y1 + y2) / 2.0) / max(frame_h, 1)
            is_close = area_ratio > 0.18
            boxes_meta.append((area_ratio, cx, cy))
            draw_boxes.append((x1, y1, x2, y2, label, conf, is_close))

        if boxes_meta:
            boxes_meta.sort(key=lambda x: x[0], reverse=True)
            closest = boxes_meta[0]
            hazard_count = len(boxes_meta)
            has_close = 1.0 if closest[0] > 0.18 else 0.0
            feats = [hazard_count, round(closest[0], 4), round(closest[1], 4), round(closest[2], 4), has_close]
            return feats, draw_boxes

        return [0, 0.0, 0.0, 0.0, 0.0], []

    def _extract_cabin_objects(self, frame):
        """Extracts in-cabin objects (cell phone, cup, bottle, food) via YOLOv8n."""
        results = self.yolo.predict(
            frame,
            verbose=False,
            conf=0.32,
            iou=0.50,
            classes=self.cabin_target_classes,
            device="cpu",
            imgsz=self.target_w,
        )[0]

        cabin_boxes = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = self.yolo.names.get(cls_id, str(cls_id))
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cabin_boxes.append((x1, y1, x2, y2, label, conf))
        return cabin_boxes

    def _yolo_inference_worker(self):
        """Background thread running Road & Cabin YOLO inference safely without stalling UI FPS."""
        while self.worker_running:
            self._worker_wakeup.wait(timeout=0.05)
            self._worker_wakeup.clear()

            with self._yolo_lock:
                road_frame = self._pending_road_frame
                self._pending_road_frame = None
                cabin_frame = self._pending_cabin_frame
                self._pending_cabin_frame = None

            if road_frame is not None:
                road_feats, road_boxes = self._extract_road_features(road_frame)
                with self._cache_lock:
                    self.cached_road_feats = road_feats
                    self.cached_road_boxes = road_boxes

            if cabin_frame is not None and self.worker_running:
                cabin_boxes = self._extract_cabin_objects(cabin_frame)
                with self._cache_lock:
                    self.cached_cabin_boxes = cabin_boxes

    def _draw_road_overlay(self, r_frame, road_boxes, road_feats):
        """Draws YOLO vehicle/pedestrian bounding boxes and road monitor HUD onto r_frame."""
        for (x1, y1, x2, y2, label, conf, is_close) in road_boxes:
            color = (0, 0, 255) if is_close else (0, 230, 0)
            cv2.rectangle(r_frame, (x1, y1), (x2, y2), color, 2)

            tag = f"{label.upper()} {conf:.2f}" + (" [CLOSE]" if is_close else "")
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            ty = max(th + 8, y1)
            cv2.rectangle(r_frame, (x1, ty - th - 8), (x1 + tw + 8, ty), color, -1)
            cv2.putText(r_frame, tag, (x1 + 4, ty - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        count = int(road_feats[0])
        has_close = road_feats[4] == 1.0
        cv2.rectangle(r_frame, (0, 0), (self.target_w, 38), (18, 18, 18), -1)
        status_txt = "HAZARD CLOSE!" if has_close else "PATH CLEAR"
        status_col = (0, 0, 255) if has_close else (0, 255, 140)
        hud = f"ROAD CONTEXT | Detections: {count} | {status_txt}"
        cv2.putText(r_frame, hud, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.56, status_col, 2, cv2.LINE_AA)

    def _draw_driver_overlay(
        self, d_frame, face_detected, landmarks, driver_status, emotion,
        emotion_scores, metrics, cabin_state, cabin_boxes
    ):
        """Draws clean driver emotion HUD, cabin activity badges, seatbelt status, and landmark guides."""
        h, w = d_frame.shape[:2]

        # 1. Cabin distraction object boxes (phone, cup, bottle, food)
        for (x1, y1, x2, y2, label, conf) in cabin_boxes:
            cv2.rectangle(d_frame, (x1, y1), (x2, y2), (255, 180, 0), 2)
            tag = f"{label.upper()} {conf:.2f}"
            cv2.putText(d_frame, tag, (x1, max(18, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 0), 2, cv2.LINE_AA)

        # 2. Optional subtle facial landmark tracking points & seatbelt visual
        if face_detected and landmarks is not None and self.show_landmarks:
            key_indices = [
                33, 160, 158, 133, 153, 144,   # Left eye
                362, 385, 387, 263, 373, 380,  # Right eye
                61, 291, 13, 14,               # Mouth
                105, 334,                      # Brows
            ]
            for idx in key_indices:
                lx = int(landmarks[idx].x * w)
                ly = int(landmarks[idx].y * h)
                cv2.circle(d_frame, (lx, ly), 2, (0, 255, 255), -1)

            if cabin_state.get("seatbelt") == "ON" and cabin_state.get("seatbelt_line"):
                sx1, sy1, sx2, sy2 = cabin_state["seatbelt_line"]
                cv2.line(d_frame, (sx1, sy1), (sx2, sy2), (0, 255, 120), 2)

        # 3. Top Header Banner (Driver Status & Primary Emotion)
        status_colors = {
            "NORMAL":     (0, 130, 0),
            "ANXIOUS":    (0, 120, 230),
            "STRESSED":   (0, 100, 200),
            "DISTRACTED": (0, 0, 200),
            "DROWSY":     (0, 0, 220),
            "NO FACE":    (30, 30, 180),
        }
        bg_col = status_colors.get(driver_status, (60, 60, 60))
        cv2.rectangle(d_frame, (0, 0), (w, 68), (18, 18, 18), -1)
        cv2.rectangle(d_frame, (6, 6), (270, 38), bg_col, -1)

        if driver_status == "NO FACE":
            cv2.putText(d_frame, "NO FACE DETECTED", (14, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(d_frame, "EMOTION: NONE | CAMERA OBSTRUCTED / NO DRIVER", (10, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (120, 120, 255), 1, cv2.LINE_AA)
            # Center screen warning frame
            cv2.rectangle(d_frame, (w // 2 - 200, h // 2 - 35), (w // 2 + 200, h // 2 + 35), (0, 0, 140), -1)
            cv2.rectangle(d_frame, (w // 2 - 200, h // 2 - 35), (w // 2 + 200, h // 2 + 35), (0, 0, 255), 2)
            cv2.putText(d_frame, "NO FACE DETECTED", (w // 2 - 140, h // 2 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(d_frame, "Ensure driver is visible to camera", (w // 2 - 145, h // 2 + 23),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 255), 1, cv2.LINE_AA)
            return

        # State badge
        cv2.putText(d_frame, f"STATE: {driver_status}", (14, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA)

        # Emotion readout badge
        em_conf = int(emotion_scores.get(emotion, 0.0) * 100)
        em_color = (0, 255, 160) if emotion in ("neutral", "happy") else (0, 180, 255)
        cv2.putText(d_frame, f"EMOTION: {emotion.upper()} ({em_conf}%)", (282, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, em_color, 2, cv2.LINE_AA)

        # Second HUD row: Seatbelt + Activity + FPS
        sb = cabin_state.get("seatbelt", "N/A")
        act = cabin_state.get("primary_activity", "ATTENTIVE")
        sb_col = (0, 230, 100) if sb == "ON" else ((0, 70, 255) if sb == "OFF" else (180, 180, 180))
        act_col = (0, 230, 100) if act == "ATTENTIVE" else (0, 200, 255)

        cv2.putText(d_frame, f"SEATBELT: {sb}", (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, sb_col, 2, cv2.LINE_AA)
        cv2.putText(d_frame, f"ACTIVITY: {act}", (185, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, act_col, 2, cv2.LINE_AA)
        cv2.putText(d_frame, f"FPS: {self.current_fps:.1f}", (545, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)

        # 4. Mini Emotion Probability Bars (bottom-left above footer)
        if emotion_scores:
            bar_x = 10
            bar_y0 = h - 108
            cv2.rectangle(d_frame, (4, bar_y0 - 16), (175, h - 6), (20, 20, 20), -1)
            cv2.putText(d_frame, f"EAR:{metrics.get('ear',0):.2f} MAR:{metrics.get('mar',0):.2f}",
                        (8, bar_y0 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
            display_ems = [("neutral", "NEU"), ("happy", "HAP"), ("angry", "ANG"), ("sad", "SAD"), ("drowsy", "DRW")]
            for i, (em_key, short) in enumerate(display_ems):
                y = bar_y0 + 14 + i * 17
                val = float(emotion_scores.get(em_key, 0.0))
                bar_w = int(val * 95)
                col = (0, 255, 160) if em_key == emotion else (130, 130, 130)
                cv2.putText(d_frame, short, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1)
                cv2.rectangle(d_frame, (45, y - 9), (45 + 95, y), (45, 45, 45), -1)
                if bar_w > 0:
                    cv2.rectangle(d_frame, (45, y - 9), (45 + bar_w, y), col, -1)

    def _make_offline_driver_frame(self):
        """Creates a clear diagnostic frame when a live webcam is closed or unavailable."""
        frame = np.zeros((self.target_h, self.target_w, 3), dtype=np.uint8)
        cv2.rectangle(frame, (0, 0), (self.target_w, 68), (18, 18, 18), -1)
        cv2.rectangle(frame, (6, 6), (270, 38), (30, 30, 180), -1)
        cv2.putText(frame, "NO FACE DETECTED", (14, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "CAMERA CLOSED OR DISCONNECTED", (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (120, 120, 255), 1, cv2.LINE_AA)

        cv2.putText(frame, "WEBCAM CLOSED / NO SIGNAL", (self.target_w // 2 - 195, self.target_h // 2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.80, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Press [C] to switch to Car Cabin Video", (self.target_w // 2 - 180, self.target_h // 2 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)
        return frame

    def step(self):
        """Executes a single multimodal perception cycle."""
        with self.cap_lock:
            ret_d, d_frame = self.driver_cap.read()
            if not ret_d and self.driver_is_file:
                self.driver_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret_d, d_frame = self.driver_cap.read()

            ret_r, r_frame = self.road_cap.read()
            if not ret_r and self.road_is_file:
                self.road_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret_r, r_frame = self.road_cap.read()

        # If road stream failed temporarily, create dark placeholder instead of stalling UI
        if not ret_r or r_frame is None:
            r_frame = np.zeros((self.target_h, self.target_w, 3), dtype=np.uint8)
            cv2.putText(r_frame, "ROAD FEED UNAVAILABLE", (160, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        elif r_frame.shape[1] != self.target_w or r_frame.shape[0] != self.target_h:
            r_frame = cv2.resize(r_frame, (self.target_w, self.target_h), interpolation=cv2.INTER_LINEAR)

        # FPS calculation
        self.fps_counter += 1
        now = time.time()
        if now - self.fps_start >= 1.0:
            self.current_fps = self.fps_counter / (now - self.fps_start)
            self.fps_counter = 0
            self.fps_start = now

        # Handle disconnected / closed live driver camera cleanly without freezing
        if not ret_d or d_frame is None:
            self.face_lost_counter += 1
            self.emotion_detector.reset()
            self.cabin_behavior.reset()
            d_frame = self._make_offline_driver_frame()

            with self._cache_lock:
                r_feats = list(self.cached_road_feats)
                r_boxes = list(self.cached_road_boxes)
            self._draw_road_overlay(r_frame, r_boxes, r_feats)

            has_hazard = r_feats[4] == 1.0
            payload = {
                "driver_state": {
                    "driver_status": "NO FACE",
                    "emotion": "none",
                    "emotion_scores": {},
                    "activity": "NO DRIVER",
                    "seatbelt": "N/A",
                    "talking": False,
                    "eating_drinking": False,
                    "phone_use": False,
                    "yawning": False,
                },
                "road_state": {
                    "high_risk_hazard": "Close Vehicle/Pedestrian" if has_hazard else "Clear",
                    "vehicle_count": int(r_feats[0]),
                },
                "system_risk_level": "HIGH",
            }
            return d_frame, r_frame, payload

        if d_frame.shape[1] != self.target_w or d_frame.shape[0] != self.target_h:
            d_frame = cv2.resize(d_frame, (self.target_w, self.target_h), interpolation=cv2.INTER_LINEAR)

        self.frame_counter += 1

        # 1. Driver FaceMesh + Emotion Extraction
        (
            d_feats, face_detected, is_looking_away,
            emotion, emotion_scores, metrics, landmarks, rgb_frame
        ) = self._extract_driver_features(d_frame)

        # 2. Schedule async Road & Cabin YOLO tasks (thread-safe single worker)
        trigger_yolo = False
        with self._yolo_lock:
            if self.frame_counter % self.skip_road_frames == 0:
                self._pending_road_frame = r_frame.copy()
                trigger_yolo = True
            if self.frame_counter % (self.skip_road_frames * 2) == 0:
                self._pending_cabin_frame = d_frame.copy()
                trigger_yolo = True
        if trigger_yolo:
            self._worker_wakeup.set()

        with self._cache_lock:
            r_feats = list(self.cached_road_feats)
            r_boxes = list(self.cached_road_boxes)
            cabin_boxes = list(self.cached_cabin_boxes)

        # 3. In-Cabin Behavior Evaluation (Seatbelt, Talking, Eating, Phone, Yawning)
        cabin_state = self.cabin_behavior.evaluate(
            d_frame, rgb_frame, landmarks, metrics.get("mar", 0.0), cabin_boxes=cabin_boxes
        )

        # 4. Handle Face Lost -> Explicit NO FACE State
        if not face_detected:
            self.face_lost_counter += 1
            self.looking_away_streak = 0
            if self.face_lost_counter >= 5:
                self.emotion_detector.reset()
        else:
            self.face_lost_counter = 0

        # 5. Build 41-dim Multimodal Vector & Run BiGRU (for sequence embedding consistency)
        frame_vector = d_feats + r_feats
        self.feature_buffer.append(frame_vector)

        bigru_status = "NORMAL"
        if len(self.feature_buffer) == self.seq_len and face_detected:
            tensor_input = torch.as_tensor(
                [list(self.feature_buffer)], dtype=torch.float32, device=self.device
            )
            with torch.inference_mode():
                logits = self.model(tensor_input)
                pred_idx = torch.argmax(logits, dim=1).item()
                bigru_status = self.label_map.get(pred_idx, "NORMAL")

        # 6. Temporal Facial Agitation (Calibrated threshold so normal webcam stays NORMAL)
        is_agitated = False
        if len(self.feature_buffer) == self.seq_len and face_detected:
            buf_arr = np.array(self.feature_buffer)
            face_frames = buf_arr[:, :36]
            valid_mask = np.any(face_frames != 0, axis=1)
            if np.sum(valid_mask) >= 20:
                temporal_jitter = float(np.std(face_frames[valid_mask], axis=0).mean())
                is_agitated = temporal_jitter > 0.038

        # 7. Emotion-First Driver State Decision Fusion
        # Road hazards NO LONGER force the driver emotion to ANXIOUS.
        if is_looking_away:
            self.looking_away_streak += 1
        else:
            self.looking_away_streak = 0

        if not face_detected and self.face_lost_counter >= 5:
            driver_status = "NO FACE"
            self.distraction_hold_frames = 0
        elif emotion == "drowsy" or cabin_state["yawning"]:
            driver_status = "DROWSY"
            self.distraction_hold_frames = 0
        elif self.looking_away_streak >= 6 or cabin_state["phone_use"] or cabin_state["eating_drinking"]:
            driver_status = "DISTRACTED"
            self.distraction_hold_frames = 5
        elif self.distraction_hold_frames > 0:
            self.distraction_hold_frames -= 1
            driver_status = "DISTRACTED"
        elif emotion in ("angry", "fear") or (is_agitated and emotion not in ("neutral", "happy")):
            driver_status = "ANXIOUS"
        elif emotion == "sad" and emotion_scores.get("sad", 0.0) >= 0.40:
            driver_status = "STRESSED"
        else:
            driver_status = "NORMAL"

        # 8. Risk Decision Fusion
        has_hazard = r_feats[4] == 1.0
        unsafe_cabin = (
            cabin_state["seatbelt"] == "OFF"
            or cabin_state["phone_use"]
            or cabin_state["eating_drinking"]
            or cabin_state["yawning"]
        )
        unsafe_driver = driver_status in ("DISTRACTED", "ANXIOUS", "DROWSY", "NO FACE", "STRESSED")
        system_risk = "HIGH" if (unsafe_driver or unsafe_cabin or has_hazard) else "LOW"

        state_payload = {
            "driver_state": {
                "driver_status": driver_status,
                "emotion": emotion,
                "emotion_scores": emotion_scores,
                "activity": cabin_state["primary_activity"],
                "seatbelt": cabin_state["seatbelt"],
                "talking": cabin_state["talking"],
                "eating_drinking": cabin_state["eating_drinking"],
                "phone_use": cabin_state["phone_use"],
                "yawning": cabin_state["yawning"],
                "bigru_signal": bigru_status,
                "ear": metrics.get("ear", 0.0),
                "mar": metrics.get("mar", 0.0),
            },
            "road_state": {
                "high_risk_hazard": "Close Vehicle/Pedestrian" if has_hazard else "Clear",
                "vehicle_count": int(r_feats[0]),
            },
            "system_risk_level": system_risk,
        }

        # 9. Render Visual Overlays onto both feeds
        self._draw_driver_overlay(
            d_frame, face_detected, landmarks, driver_status,
            emotion, emotion_scores, metrics, cabin_state, cabin_boxes
        )
        self._draw_road_overlay(r_frame, r_boxes, r_feats)

        return d_frame, r_frame, state_payload

    def stop(self):
        """Clean shutdown of captures, workers, and MediaPipe resources."""
        self.worker_running = False
        self._worker_wakeup.set()
        with self.cap_lock:
            if self.driver_cap is not None:
                self.driver_cap.release()
            if self.road_cap is not None:
                self.road_cap.release()
        self.cabin_behavior.close()
