import threading
import time
import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient
from inference_sdk.webrtc import WebcamSource, VideoFileSource, StreamConfig, VideoMetadata


class RoboflowServerlessStreamer:
    """
    Non-blocking Roboflow Serverless WebRTC Streamer for Affect-Drive.
    Uses Roboflow's serverless cloud GPU infrastructure (https://serverless.roboflow.com)
    to stream live webcam or video files with real-time WebRTC pose/detection output.
    """
    def __init__(
        self,
        source=0,  # 0 for webcam, or path string e.g. "data/sample_road.mp4"
        api_url="https://serverless.roboflow.com",
        api_key="pbLftKrM5HXO73JtKBTS",
        workflow="humanrobotpose-2-vhumanrobotpose-2-6a3rj-1-yolo26n-pose-t1-logic",
        workspace="vighnesh-b-shriyan",
        requested_plan="webrtc-gpu-medium",  # webrtc-gpu-small, webrtc-gpu-medium, webrtc-gpu-large
        requested_region="us",               # us, eu, ap
        resolution=(1280, 720)
    ):
        self.source_input = source
        self.api_url = api_url
        self.api_key = api_key
        self.workflow = workflow
        self.workspace = workspace
        self.requested_plan = requested_plan
        self.requested_region = requested_region
        self.resolution = resolution

        # Thread synchronization
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_data = {}
        self.is_running = False
        self.session = None
        self.worker_thread = None

        # Initialize Roboflow Serverless Client
        self.client = InferenceHTTPClient.init(
            api_url=self.api_url,
            api_key=self.api_key
        )

        # Configure Source (Webcam or Video File)
        self.is_webcam = isinstance(source, int) or str(source).isdigit() or source == "webcam"
        if self.is_webcam:
            self.source = WebcamSource(resolution=self.resolution)
            self.config = StreamConfig(
                data_output=["predictions"],
                processing_timeout=3600,
                requested_plan=self.requested_plan,
                requested_region=self.requested_region
            )
        else:
            self.source = VideoFileSource(str(source), realtime_processing=True)
            self.config = StreamConfig(
                data_output=["predictions"],
                processing_timeout=3600,
                requested_plan=self.requested_plan,
                requested_region=self.requested_region
            )

    def start(self):
        """Starts the WebRTC streaming session in a background daemon thread."""
        if self.is_running:
            return self

        self.is_running = True
        print(f"[ROBOFLOW SERVERLESS] Initializing WebRTC session on {self.requested_plan} ({self.requested_region})...")

        self.session = self.client.webrtc.stream(
            source=self.source,
            workflow=self.workflow,
            workspace=self.workspace,
            image_input="image",
            config=self.config
        )

        # Incoming frame handler
        @self.session.on_frame
        def handle_frame(frame, metadata):
            with self.lock:
                self.latest_frame = frame

        # Incoming prediction data handler
        @self.session.on_data()
        def handle_data(data: dict, metadata: VideoMetadata):
            with self.lock:
                self.latest_data = data

        def _run_worker():
            try:
                self.session.run()
            except Exception as e:
                print(f"[ROBOFLOW SERVERLESS ERROR] {e}")
            finally:
                self.is_running = False

        self.worker_thread = threading.Thread(target=_run_worker, daemon=True)
        self.worker_thread.start()
        print(f"[ROBOFLOW SERVERLESS] Stream running in background from {'Webcam' if self.is_webcam else self.source_input}")
        return self

    def read(self):
        """Thread-safe fetch of latest annotated frame and prediction data."""
        with self.lock:
            frame = self.latest_frame.copy() if self.latest_frame is not None else None
            data = dict(self.latest_data) if self.latest_data else {}
        return frame, data

    def extract_hazard_metrics(self, data, frame_shape=(480, 640)):
        """Extracts 5-dimensional hazard vector for BiGRU integration."""
        frame_h, frame_w = frame_shape[:2]
        frame_area = frame_h * frame_w
        predictions = data.get("predictions", []) if isinstance(data, dict) else []

        boxes = []
        if isinstance(predictions, list):
            for pred in predictions:
                if isinstance(pred, dict):
                    w = pred.get("width", 0)
                    h = pred.get("height", 0)
                    cx = pred.get("x", 0) / frame_w if frame_w else 0
                    cy = pred.get("y", 0) / frame_h if frame_h else 0
                    norm_area = (w * h) / frame_area if frame_area else 0
                    boxes.append((norm_area, cx, cy))

        if boxes:
            boxes.sort(key=lambda x: x[0], reverse=True)
            closest = boxes[0]
            has_close = 1.0 if closest[0] > 0.10 else 0.0
            return [len(boxes), round(closest[0], 4), round(closest[1], 4), round(closest[2], 4), has_close]

        return [0, 0.0, 0.0, 0.0, 0.0]

    def stop(self):
        """Clean shutdown of WebRTC session."""
        self.is_running = False
        if self.session:
            try:
                self.session.close()
            except Exception:
                pass

