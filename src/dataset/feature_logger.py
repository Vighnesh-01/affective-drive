import cv2
import csv
import time
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

# Dynamic resolution for MediaPipe Face Mesh solutions
if hasattr(mp, "solutions"):
    mp_face_mesh = mp.solutions.face_mesh
else:
    import mediapipe.solutions.face_mesh as mp_face_mesh  # type: ignore

class FeatureLogger:
    def __init__(self, output_csv="data/phase2_dataset.csv"):
        self.output_csv = output_csv
        self.yolo = YOLO("yolov8n.pt")
        
        # Initialize FaceMesh directly
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

        # Key landmark indices: Left Eye, Right Eye, Mouth, Eyebrows, Nose
        self.LANDMARK_IDS = [33, 133, 362, 263, 61, 291, 0, 17, 10, 152, 1, 263]
        
        self._init_csv()

    def _init_csv(self):
        """Creates the CSV header if the file does not exist."""
        header = ["sequence_id", "timestamp", "label"]
        
        # 12 Facial Landmarks -> (x, y, z) per landmark = 36 features
        for i in range(len(self.LANDMARK_IDS)):
            header.extend([f"lm_{i}_x", f"lm_{i}_y", f"lm_{i}_z"])
            
        # Road Context Features = 5 features
        header.extend(["hazard_count", "closest_box_area", "closest_box_cx", "closest_box_cy", "has_close_hazard"])
        
        with open(self.output_csv, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
        print(f"[LOGGER] Initialized dataset file at {self.output_csv}")

    def extract_driver_features(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        features = []
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            for idx in self.LANDMARK_IDS:
                lm = landmarks[idx]
                features.extend([round(lm.x, 4), round(lm.y, 4), round(lm.z, 4)])
        else:
            features = [0.0] * (len(self.LANDMARK_IDS) * 3)
            
        return features

    def extract_road_features(self, frame):
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
                norm_area = area / frame_area
                boxes.append((norm_area, cx, cy))

        if boxes:
            boxes.sort(key=lambda x: x[0], reverse=True)
            closest = boxes[0]
            hazard_count = len(boxes)
            has_close = 1.0 if closest[0] > 0.18 else 0.0
            return [hazard_count, round(closest[0], 4), round(closest[1], 4), round(closest[2], 4), has_close]
        else:
            return [0, 0.0, 0.0, 0.0, 0.0]

    def log_frame(self, sequence_id, label, driver_frame, road_frame):
        driver_feats = self.extract_driver_features(driver_frame)
        road_feats = self.extract_road_features(road_frame)
        
        row = [sequence_id, time.time(), label] + driver_feats + road_feats
        
        with open(self.output_csv, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)