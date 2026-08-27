import sys
from pathlib import Path

# Resolve project root folder (affective-drive)
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import cv2
from src.perception.dual_stream import DualStreamPipeline
from src.perception.driver_detector import DriverStateTracker
from src.perception.road_detector import RoadHazardDetector

class AffectivePerceptionEngine:
    def __init__(self, driver_src=0, road_src="data/sample_road.mp4"):
        self.dual_stream = DualStreamPipeline(driver_src, road_src)
        self.driver_tracker = DriverStateTracker()
        self.road_detector = RoadHazardDetector()

    def step(self):
        driver_frame, road_frame, synced = self.dual_stream.get_synced_frames()
        if driver_frame is None or road_frame is None:
            return None, None, None

        # Process individual perception channels
        proc_driver, driver_state = self.driver_tracker.process(driver_frame)
        proc_road, road_state = self.road_detector.process(road_frame)

        # Dual-Path Adaptive Fusion Payload
        state_payload = {
            "synced": synced,
            "driver_state": driver_state,
            "road_state": road_state,
            "system_risk_level": "HIGH" if (not driver_state["face_detected"] or road_state["high_risk_hazard"]) else "LOW"
        }

        return proc_driver, proc_road, state_payload

    def stop(self):
        self.dual_stream.stop()

if __name__ == "__main__":
    engine = AffectivePerceptionEngine(driver_src=0, road_src="data/sample_road.mp4")
    try:
        while True:
            d_frame, r_frame, payload = engine.step()
            if d_frame is not None and r_frame is not None:
                r_resized = cv2.resize(r_frame, (d_frame.shape[1], d_frame.shape[0]))
                cv2.imshow("Affective-Drive Perception Output", cv2.hconcat([d_frame, r_resized]))
                print(f"[TELEM] System Risk: {payload['system_risk_level']} | Driver Status: {payload['driver_state']['driver_status']}")
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        engine.stop()
        cv2.destroyAllWindows()