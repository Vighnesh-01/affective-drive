import cv2
import time
import threading

class FrameStreamer:
    """Multithreaded stream reader for live cameras or video files."""
    def __init__(self, source):
        self.source = source
        self.cap = cv2.VideoCapture(source)
        self.is_file = isinstance(source, str)
        self.ret, self.frame = self.cap.read()
        self.timestamp = time.time()
        self.stopped = False
        self.lock = threading.Lock()

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            if not self.cap.isOpened():
                break
            
            ret, frame = self.cap.read()
            
            # Loop video file seamlessly if it reaches the end
            if not ret and self.is_file:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()

            if ret:
                with self.lock:
                    self.frame = frame
                    self.timestamp = time.time()
            time.sleep(0.01)  # Prevents CPU overload

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None, self.timestamp

    def stop(self):
        self.stopped = True
        if self.cap.isOpened():
            self.cap.release()


class DualStreamPipeline:
    """Synchronizes Driver and Road streams within a configurable time window."""
    def __init__(self, driver_src=0, road_src="data/sample_road.mp4", sync_threshold_ms=50):
        self.driver_stream = FrameStreamer(driver_src).start()
        self.road_stream = FrameStreamer(road_src).start()
        self.sync_threshold = sync_threshold_ms / 1000.0

    def get_synced_frames(self):
        driver_frame, ts_driver = self.driver_stream.read()
        road_frame, ts_road = self.road_stream.read()

        if driver_frame is None or road_frame is None:
            return None, None, False

        # Calculate capture delta
        delta = abs(ts_driver - ts_road)
        is_synced = delta <= self.sync_threshold
        
        return driver_frame, road_frame, is_synced

    def stop(self):
        self.driver_stream.stop()
        self.road_stream.stop()


if __name__ == "__main__":
    # Test script standalone execution
    pipeline = DualStreamPipeline(driver_src=0, road_src="data/sample_road.mp4")
    print("--- Starting Dual Stream Feed (Press 'q' to quit) ---")

    try:
        while True:
            driver, road, synced = pipeline.get_synced_frames()
            
            if driver is not None and road is not None:
                # Resize road video for side-by-side display matching driver resolution
                road_resized = cv2.resize(road, (driver.shape[1], driver.shape[0]))
                
                # Visual sync indicator badge
                color = (0, 255, 0) if synced else (0, 0, 255)
                status_text = "SYNCED" if synced else "OUT OF SYNC"
                cv2.putText(driver, f"Driver Feed | {status_text}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(road_resized, "Road Feed Context", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                # Combine both views into a single cockpit window
                combined_view = cv2.hconcat([driver, road_resized])
                cv2.imshow("Affective-Drive Cockpit View", combined_view)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()