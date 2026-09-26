import cv2
import time
import argparse
from src.perception.roboflow_serverless import RoboflowServerlessStreamer

def main():
    parser = argparse.ArgumentParser(description="Test Roboflow Serverless Cloud WebRTC Streaming")
    parser.add_argument("--source", default="0", help="Webcam index (e.g. 0) or video file path (e.g. data/sample_road.mp4)")
    parser.add_argument("--api-url", default="https://serverless.roboflow.com", help="Roboflow serverless endpoint")
    parser.add_argument("--api-key", default="pbLftKrM5HXO73JtKBTS", help="Roboflow API key")
    parser.add_argument("--plan", default="webrtc-gpu-medium", help="Serverless GPU plan (webrtc-gpu-small, webrtc-gpu-medium, webrtc-gpu-large)")
    parser.add_argument("--region", default="us", help="Serverless region (us, eu, ap)")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source

    print(f"[*] Initializing Serverless Roboflow WebRTC Streamer for: {source}")
    print(f"[*] Endpoint: {args.api_url} | Plan: {args.plan} | Region: {args.region}")

    streamer = RoboflowServerlessStreamer(
        source=source,
        api_url=args.api_url,
        api_key=args.api_key,
        requested_plan=args.plan,
        requested_region=args.region
    ).start()

    print("[*] Connecting to Roboflow Cloud GPU via WebRTC (Press 'q' to exit)...")
    try:
        while True:
            frame, data = streamer.read()

            if frame is not None:
                # Extract hazard / pose metrics
                features = streamer.extract_hazard_metrics(data, frame_shape=frame.shape)

                # Overlay status
                status_text = f"Detections: {features[0]} | Close Hazard: {features[4] == 1.0}"
                cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow("Affect-Drive: Roboflow Serverless WebRTC", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(0.01)

    finally:
        print("[*] Closing WebRTC session...")
        streamer.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

