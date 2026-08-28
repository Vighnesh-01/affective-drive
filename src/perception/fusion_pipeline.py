import cv2
import torch
import numpy as np
from collections import deque
import mediapipe as mp
from ultralytics import YOLO
from src.model.bigru_model import AffectiveBiGRU

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
    def __init__(self, driver_src=0, road_src="data/sample_road.mp4", model_path="data/bigru_affective.pth", seq_len=30):
        self.seq_len = seq_len
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Video Captures
        self.driver_cap = cv2.VideoCapture(driver_src)
        self.road_cap = cv2.VideoCapture(road_src)

        # 2. Vision Models (MediaPipe + YOLO)
        self.yolo = YOLO("yolov8n.pt")
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        self.LANDMARK_IDS = [33, 133, 362, 263, 61, 291, 0, 17, 10, 152, 1, 263]

        # 3. PyTorch BiGRU Model Initialization & Weight Loading
        self.model = AffectiveBiGRU(input_size=41, hidden_size=64, num_classes=3).to(self.device)
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print(f"[FUSION ENGINE] Successfully loaded PyTorch BiGRU from {model_path}")
        except Exception as e:
            print(f"[FUSION ENGINE WARNING] Could not load model weights ({e}). Initializing default weights.")

        # 4. Temporal Feature Buffer (Sliding Window of 30 frames)
        self.feature_buffer = deque(maxlen=self.seq_len)
        self.label_map = {0: "NORMAL", 1: "ANXIOUS", 2: "DISTRACTED"}

    def _extract_driver_features(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            feats = []
            for idx in self.LANDMARK_IDS:
                lm = landmarks[idx]
                feats.extend([round(lm.x, 4), round(lm.y, 4), round(lm.z, 4)])
            return feats
        return [0.0] * (len(self.LANDMARK_IDS) * 3)

    def _extract_road_features(self, frame):
        results = self.yolo.predict(frame, verbose=False, conf=0.35)[0]
        frame_h, frame_w = frame.shape[:2]
        frame_area = frame_h * frame_w
        
        target_classes = [0, 2, 3, 5, 7]  # Person, Car, Motorcycle, Bus, Truck
        boxes = []
        
        for box in results.boxes:
            if int(box.cls[0]) in target_classes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                cx = ((x1 + x2) / 2) / frame_w
                cy = ((y1 + y2) / 2) / frame_h
                boxes.append((area / frame_area, cx, cy))

        if boxes:
            boxes.sort(key=lambda x: x[0], reverse=True)
            closest = boxes[0]
            hazard_count = len(boxes)
            has_close = 1.0 if closest[0] > 0.18 else 0.0
            return [hazard_count, round(closest[0], 4), round(closest[1], 4), round(closest[2], 4), has_close]
        return [0, 0.0, 0.0, 0.0, 0.0]

    def step(self):
        ret_d, d_frame = self.driver_cap.read()
        ret_r, r_frame = self.road_cap.read()

        if not ret_r:
            self.road_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret_r, r_frame = self.road_cap.read()

        if not ret_d or not ret_r:
            return None, None, None

        # 1. Extract current frame features (41 dimensions)
        d_feats = self._extract_driver_features(d_frame)
        r_feats = self._extract_road_features(r_frame)
        frame_vector = d_feats + r_feats

        # 2. Append to rolling temporal buffer
        self.feature_buffer.append(frame_vector)

        # 3. Model Real-Time Inference
        driver_status = "INITIALIZING"
        if len(self.feature_buffer) == self.seq_len:
            tensor_input = torch.tensor([list(self.feature_buffer)], dtype=torch.float32).to(self.device)
            
            with torch.no_grad():
                logits = self.model(tensor_input)
                pred_idx = torch.argmax(logits, dim=1).item()
                driver_status = self.label_map.get(pred_idx, "UNKNOWN")

        # 4. Decision Fusion
        has_hazard = r_feats[4] == 1.0
        system_risk = "HIGH" if (driver_status in ["DISTRACTED", "ANXIOUS"] or has_hazard) else "LOW"

        state_payload = {
            "driver_state": {"driver_status": driver_status},
            "road_state": {"high_risk_hazard": "Close Vehicle/Pedestrian" if has_hazard else "Clear"},
            "system_risk_level": system_risk
        }

        color = (0, 0, 255) if system_risk == "HIGH" else (0, 255, 0)
        cv2.putText(d_frame, f"BiGRU State: {driver_status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return d_frame, r_frame, state_payload

    def stop(self):
        self.driver_cap.release()
        self.road_cap.release()