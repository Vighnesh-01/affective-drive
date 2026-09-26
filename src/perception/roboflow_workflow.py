import threading
import queue
import time
import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient, InferenceConfiguration


class RoboflowWorkflowStreamer:
    """
    Robust, Multi-source Roboflow Workflow Streamer for Affect-Drive.
    Uses standard HTTPS Workflow API (no WebRTC STUN/TURN errors) with an asynchronous
    background worker thread to maintain 25-30 FPS display on both Webcams and Video Files.
    """
    def __init__(
        self,
        source=0,  # 0 or integer for webcam; file path string for video file
        api_url="https://detect.roboflow.com",
        api_key="pbLftKrM5HXO73JtKBTS",
        workflow="humanrobotpose-2-vhumanrobotpose-2-6a3rj-1-yolo26n-pose-t1-logic",
        workspace="vighnesh-b-shriyan",
        resolution=(640, 480),
        skip_frames=2  # Process every Nth frame via API to keep real-time latency low
    ):
        self.source_input = source
        self.api_url = api_url
        self.api_key = api_key
        self.workflow = workflow
        self.workspace = workspace
        self.target_w, self.target_h = resolution
        self.skip_frames = max(1, skip_frames)

        # Initialize Roboflow HTTP Client
        config = InferenceConfiguration(api_key_transport="header")
        self.client = InferenceHTTPClient(
            api_url=self.api_url,
            api_key=self.api_key
        )
        self.client.use_configuration(config)

        # OpenCV Capture (handles webcam and video files)
        self.is_video_file = isinstance(source, str) and not source.isdigit() and source != "0"
        src = str(source) if self.is_video_file else (int(source) if str(source).isdigit() else 0)
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_h)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Threading state & Queues
        self.is_running = False
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_data = {}
        self.cached_predictions = []
        self.input_queue = queue.Queue(maxsize=1)
        self.worker_thread = None

    def start(self):
        """Starts video capture reader and background inference worker."""
        if self.is_running:
            return self

        self.is_running = True
        self.worker_thread = threading.Thread(target=self._inference_worker, daemon=True)
        self.worker_thread.start()
        print(f"[ROBOFLOW WORKFLOW] Streaming started from {'Video File: ' + str(self.source_input) if self.is_video_file else 'Webcam ' + str(self.source_input)}")
        return self

    def _inference_worker(self):
        """Background worker thread that dispatches frames to Roboflow Workflow API."""
        while self.is_running:
            try:
                frame_to_infer = self.input_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if frame_to_infer is None:
                break

            try:
                # Convert BGR -> RGB for Roboflow API
                rgb = cv2.cvtColor(frame_to_infer, cv2.COLOR_BGR2RGB)
                response = self.client.run_workflow(
                    workspace_name=self.workspace,
                    workflow_id=self.workflow,
                    images={"image": rgb}
                )

                if response and isinstance(response, list) and len(response) > 0:
                    data = response[0]
                    with self.lock:
                        self.latest_data = data
                        self.cached_predictions = self._extract_raw_predictions(data)

            except Exception as e:
                # Silently catch network drops without crashing main loop
                pass

            self.input_queue.task_done()

    def _extract_raw_predictions(self, data):
        """Extracts list of predictions/boxes/keypoints from workflow dictionary."""
        preds = []
        if not isinstance(data, dict):
            return preds

        # Scan top-level keys for predictions list
        for k, v in data.items():
            if isinstance(v, list):
                preds.extend(v)
            elif isinstance(v, dict):
                sub_preds = v.get("predictions", [])
                if isinstance(sub_preds, list):
                    preds.extend(sub_preds)
        return preds

    def read(self):
        """
        Fetches the latest frame from the camera/video, overlays cached workflow predictions,
        and returns (annotated_frame, raw_data).
        """
        if not self.cap.isOpened():
            return None, {}

        ret, frame = self.cap.read()

        # Loop video file seamlessly
        if not ret and self.is_video_file:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()

        if not ret or frame is None:
            return None, {}

        # Resize if needed
        if frame.shape[1] != self.target_w or frame.shape[0] != self.target_h:
            frame = cv2.resize(frame, (self.target_w, self.target_h), interpolation=cv2.INTER_LINEAR)

        # Dispatch frame to background worker queue (non-blocking)
        if self.input_queue.empty():
            try:
                self.input_queue.put_nowait(frame.copy())
            except queue.Full:
                pass

        # Draw predictions on frame
        with self.lock:
            preds = list(self.cached_predictions)
            raw_data = dict(self.latest_data)

        self._draw_overlay(frame, preds)
        return frame, raw_data

    def _draw_overlay(self, frame, predictions):
        """Draws bounding boxes, labels, and keypoints on the display frame."""
        for p in predictions:
            if isinstance(p, dict) and "x" in p and "y" in p:
                x, y = int(p["x"]), int(p["y"])
                w, h = int(p.get("width", 20)), int(p.get("height", 20))
                label = p.get("class", p.get("label", "hazard"))
                conf = float(p.get("confidence", 1.0))

                x1, y1 = max(0, x - w // 2), max(0, y - h // 2)
                x2, y2 = min(self.target_w, x + w // 2), min(self.target_h, y + h // 2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    frame,
                    f"{label} {conf:.2f}",
                    (x1, max(18, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2
                )

    def extract_hazard_metrics(self, data, frame_shape=(480, 640)):
        """
        Extracts 5-dimensional feature vector:
        [hazard_count, closest_box_area, closest_cx, closest_cy, has_close_hazard]
        """
        frame_h, frame_w = frame_shape[:2]
        frame_area = frame_h * frame_w
        with self.lock:
            preds = list(self.cached_predictions)

        boxes = []
        for p in preds:
            if isinstance(p, dict) and "width" in p and "height" in p:
                w, h = p["width"], p["height"]
                norm_area = (w * h) / frame_area if frame_area else 0
                cx = p.get("x", 0) / frame_w if frame_w else 0
                cy = p.get("y", 0) / frame_h if frame_h else 0
                boxes.append((norm_area, cx, cy))

        if boxes:
            boxes.sort(key=lambda x: x[0], reverse=True)
            closest = boxes[0]
            has_close = 1.0 if closest[0] > 0.10 else 0.0
            return [len(boxes), round(closest[0], 4), round(closest[1], 4), round(closest[2], 4), has_close]

        return [0, 0.0, 0.0, 0.0, 0.0]

    def stop(self):
        """Releases video resources and shuts down background worker."""
        self.is_running = False
        if self.input_queue.empty():
            try:
                self.input_queue.put_nowait(None)
            except queue.Full:
                pass
        if self.cap.isOpened():
            self.cap.release()
