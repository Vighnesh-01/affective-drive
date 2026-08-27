import cv2
from ultralytics import YOLO

class RoadHazardDetector:
    def __init__(self):
        # Downloads lightweight YOLOv8 nano model on first run
        self.model = YOLO("yolov8n.pt")
        # COCO class IDs: 2=car, 3=motorcycle, 5=bus, 7=truck, 0=person
        self.target_classes = [0, 2, 3, 5, 7]

    def process(self, frame):
        results = self.model.predict(frame, verbose=False, conf=0.4)[0]
        
        detected_objects = []
        high_risk_detected = False
        frame_height, frame_width = frame.shape[:2]

        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id in self.target_classes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = self.model.names[cls_id]
                box_area = (x2 - x1) * (y2 - y1)
                
                # Bounding box taking >20% frame area indicates close proximity
                is_close = box_area > (frame_height * frame_width * 0.20)
                if is_close:
                    high_risk_detected = True

                detected_objects.append({"label": label, "close_proximity": is_close})

                # Visual overlay
                color = (0, 0, 255) if is_close else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {'[CLOSE]' if is_close else ''}", 
                            (x1, max(y1-10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        road_state = {
            "object_count": len(detected_objects),
            "high_risk_hazard": high_risk_detected,
            "detected_objects": detected_objects
        }

        return frame, road_state