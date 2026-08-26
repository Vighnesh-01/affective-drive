import cv2

def test_available_cameras(max_tested=5):
    available_indices = []
    print("--- Searching for connected cameras ---")
    for index in range(max_tested):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                h, w, _ = frame.shape
                print(f"[SUCCESS] Camera index {index} is active. Resolution: {w}x{h}")
                available_indices.append(index)
            cap.release()
        else:
            print(f"[FAIL] Camera index {index} is not available.")
    
    return available_indices

if __name__ == "__main__":
    indices = test_available_cameras()
    print("\n-------------------------------------------")
    if len(indices) >= 2:
        print(f"Dual setup detected! Assigned Camera Indices: {indices[:2]}")
    elif len(indices) == 1:
        print(f"Single camera detected at index {indices[0]}. (Use dummy video files in data/ for second stream testing)")
    else:
        print("No cameras detected. Ensure webcams are plugged in or use offline test videos.")
    print("-------------------------------------------")