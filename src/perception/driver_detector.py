import cv2
import numpy as np

class DriverStateTracker:
    def __init__(self):
        # Load OpenCV default face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    def process(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=3, 
            minSize=(60, 60)
        )

        driver_state = {
            "face_detected": False,
            "driver_status": "NORMAL",
            "bbox": None
        }

        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            driver_state["face_detected"] = True
            driver_state["bbox"] = (x, y, w, h)
            
            # Simple heuristic: Draw tracking box and assess head alignment
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, "Driver Detected", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        else:
            driver_state["driver_status"] = "DISTRACTED_OR_LOOKING_AWAY"
            cv2.putText(frame, "WARNING: Driver Looking Away", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return frame, driver_state