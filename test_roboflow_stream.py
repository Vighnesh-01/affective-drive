import cv2
import time
import argparse
from src.perception.roboflow_workflow import RoboflowWorkflowStreamer

def run_local_mode(source="data/sample_road.mp4", model_path="best.pt"):
    """Runs ultra-fast local YOLO (best.pt) without any cloud/network dependency."""
    from ultralytics import YOLO
    print(f"[*] Starting LOCAL YOLO mode using weights: {model_path}")
    model = YOLO(model_path)

    is_video_file = isinstance(source, str) and not source.isdigit() and source != "0"
    src = str(source) if is_video_file else (int(source) if str(source).isdigit() else 0)
    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    fps_counter, fps_start = 0, time.time()
    current_fps = 0.0

    print("[*] Running local inference loop (Press 'q' to exit)...")
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret and is_video_file:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            elif not ret:
                break

            t0 = time.time()
            results = model.predict(frame, verbose=False, conf=0.35, device="cpu", imgsz=640)[0]
            infer_ms = (time.time() - t0) * 1000.0

            # Render bounding boxes
            annotated_frame = results.plot()

            # FPS counter
            fps_counter += 1
            now = time.time()
            if now - fps_start >= 1.0:
                current_fps = round(fps_counter / (now - fps_start), 1)
                fps_counter = 0
                fps_start = now

            # Top HUD
            cv2.rectangle(annotated_frame, (0, 0), (640, 35), (20, 20, 20), -1)
            hud = f"MODE: LOCAL YOLO | FPS: {current_fps} | Infer: {infer_ms:.1f}ms | Objects: {len(results.boxes)}"
            cv2.putText(annotated_frame, hud, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            cv2.imshow("Affect-Drive: Local Hazard Detection Test", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


def run_cloud_mode(source="0", api_url="https://detect.roboflow.com", api_key="pbLftKrM5HXO73JtKBTS", workflow="humanrobotpose-2-vhumanrobotpose-2-6a3rj-1-yolo26n-pose-t1-logic"):
    """Runs optimized Roboflow cloud workflow with asynchronous frame streaming."""
    source_val = int(source) if str(source).isdigit() else source

    print(f"[*] Starting OPTIMIZED CLOUD WORKFLOW mode on source: {source_val}")
    print(f"[*] Workflow: {workflow}")

    streamer = RoboflowWorkflowStreamer(
        source=source_val,
        api_url=api_url,
        api_key=api_key,
        workflow=workflow,
        infer_quality=75,
        resolution=(640, 480)
    ).start()

    print("[*] Streaming active (Press 'q' to exit, 's' to save snapshot)...")
    try:
        while True:
            frame, data = streamer.read()

            if frame is not None:
                cv2.imshow("Affect-Drive: Optimized Roboflow Workflow", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and frame is not None:
                out_name = f"snapshot_{int(time.time())}.jpg"
                cv2.imwrite(out_name, frame)
                print(f"[+] Saved snapshot to {out_name}")

            time.sleep(0.005)

    finally:
        print("[*] Shutting down streamer...")
        streamer.stop()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Optimized Roboflow Workflow & Local Hazard Test Runner")
    parser.add_argument("--mode", default="cloud", choices=["cloud", "local"], help="Inference mode: 'cloud' (Roboflow Workflow API) or 'local' (best.pt)")
    parser.add_argument("--source", default="0", help="Webcam index (e.g. 0) or video file path (e.g. data/sample_road.mp4)")
    parser.add_argument("--api-url", default="https://detect.roboflow.com", help="Roboflow server URL")
    parser.add_argument("--api-key", default="pbLftKrM5HXO73JtKBTS", help="Roboflow API key")
    parser.add_argument("--model", default="best.pt", help="Local model weights (for local mode)")
    args = parser.parse_args()

    if args.mode == "local":
        run_local_mode(source=args.source, model_path=args.model)
    else:
        run_cloud_mode(source=args.source, api_url=args.api_url, api_key=args.api_key)


if __name__ == "__main__":
    main()
