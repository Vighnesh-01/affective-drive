import cv2
import time
import argparse
from src.perception.roboflow_workflow import RoboflowWorkflowStreamer

def main():
    parser = argparse.ArgumentParser(description="Test Roboflow WebRTC Workflow Streamer")
    parser.add_argument("--source", default="0", help="Webcam index (e.g. 0) or video file path (e.g. data/sample_road.mp4)")
    parser.add_argument("--api-url", default="https://detect.roboflow.com", help="Roboflow server URL (https://detect.roboflow.com or http://localhost:9001)")
    parser.add_argument("--api-key", default="pbLftKrM5HXO73JtKBTS", help="Roboflow API key")
    args = parser.parse_args()

    # Determine source (integer for webcam, string for video)
    source = int(args.source) if args.source.isdigit() else args.source

    print(f"[*] Initializing Roboflow Streamer for source: {source}")
    streamer = RoboflowWorkflowStreamer(
        source=source,
        api_url=args.api_url,
        api_key=args.api_key
    ).start()

    print("[*] Waiting for WebRTC stream connection (Press 'q' to exit)...")
    try:
        while True:
            frame, data = streamer.read()

            if frame is not None:
                # Extract 5-feature vector
                features = streamer.extract_hazard_metrics(data, frame_shape=frame.shape)

                # Overlay status
                status_text = f"Detections: {features[0]} | Close Hazard: {features[4] == 1.0}"
                cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow("Affect-Drive: Roboflow Workflow Test", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(0.01)

    finally:
        print("[*] Shutting down streamer...")
        streamer.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

