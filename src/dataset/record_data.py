import cv2
from src.dataset.feature_logger import FeatureLogger

def record_dataset():
    logger = FeatureLogger(output_csv="data/phase2_dataset.csv")
    
    driver_cap = cv2.VideoCapture(0)
    road_cap = cv2.VideoCapture("data/sample_road.mp4")
    
    current_label = 0  # Default: 0 = Normal
    sequence_id = 1
    
    print("\n--- DATASET RECORDING CONTROLS ---")
    print("Press '0': Set Label to NORMAL (0)")
    print("Press '1': Set Label to ANXIOUS/STRESSED (1)")
    print("Press '2': Set Label to DISTRACTED/ANGRY (2)")
    print("Press 'n': Start New Sequence ID")
    print("Press 'q': Stop Recording\n")

    while True:
        ret_d, d_frame = driver_cap.read()
        ret_r, r_frame = road_cap.read()
        
        if not ret_r:
            road_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
            continue
            
        if not ret_d:
            break

        # Log feature vector
        logger.log_frame(sequence_id, current_label, d_frame, r_frame)

        # UI Overlay
        label_text = {0: "NORMAL", 1: "ANXIOUS", 2: "DISTRACTED"}.get(current_label, "UNKNOWN")
        cv2.putText(d_frame, f"REC | Seq: {sequence_id} | Label: {label_text}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        r_resized = cv2.resize(r_frame, (d_frame.shape[1], d_frame.shape[0]))
        cv2.imshow("Dataset Recorder", cv2.hconcat([d_frame, r_resized]))

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('0'):
            current_label = 0
        elif key == ord('1'):
            current_label = 1
        elif key == ord('2'):
            current_label = 2
        elif key == ord('n'):
            sequence_id += 1
            print(f"[RECORDER] Advanced to Sequence #{sequence_id}")

    driver_cap.release()
    road_cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    record_dataset()