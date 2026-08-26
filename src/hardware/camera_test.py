import cv2

def test_available_cameras(max_tested=5):
    available_indices = []
    print("--- Searching for connected cameras ---")
    for index in range(max_tested):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"[SUCCESS] Camera index {index} is active. Resolution: {frame.shape[1]}x{frame.shape[0]}")
                available_indices.append(index)
            cap.release()
        else:
            print(f"[FAIL] Camera index {index} is not available.")
    
    return available_indices

if __name__ == "__main__":
    indices = test_available_cameras()
    if len(indices) < 2:
        print("\nNote: Fewer than 2 cameras detected. If testing remotely without 2 cameras, use sample video files in data/ for Phase 2 testing.")
    else:
        print(f"\nAssigned Camera Indices: {indices}")