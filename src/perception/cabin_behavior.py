"""
In-Cabin Driver Behavior & Safety Detector for Affective-Drive.

Detects key real-world driving cabin behaviors requested for realistic
Driver Monitoring System (DMS) evaluation:
  - Seatbelt Detection (ON / OFF / NOT VISIBLE) via face-anchored torso ROI & Hough line analysis
  - Talking Detection (rhythmic MAR oscillations without yawning)
  - Eating / Drinking Detection (hand-to-mouth proximity & YOLO bottle/cup/food detections)
  - Phone Use Detection (YOLO cell-phone detection & hand-to-ear proximity)
  - Yawning Detection (sustained wide Mouth Aspect Ratio)
"""

from collections import deque
import math
import cv2
import numpy as np
import mediapipe as mp

# Safe multi-strategy MediaPipe Hands importer (optional lightweight hand tracking)
mp_hands = None
try:
    import mediapipe.python.solutions.hands as mp_hands
except (ModuleNotFoundError, AttributeError):
    try:
        import mediapipe.solutions.hands as mp_hands
    except (ModuleNotFoundError, AttributeError):
        try:
            mp_hands = mp.solutions.hands
        except AttributeError:
            mp_hands = None


class CabinBehaviorDetector:
    """
    Multimodal in-cabin driver activity and seatbelt detector.
    Combines MediaPipe FaceMesh mouth kinematics, optional MediaPipe Hands
    proximity, YOLO cabin object detections, and torso Hough-line seatbelt checks.
    """

    # COCO classes relevant to cabin distraction / eating / drinking
    CABIN_FOOD_DRINK_CLASSES = {
        "bottle", "cup", "wine glass", "fork", "knife", "spoon", "bowl",
        "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
        "hot dog", "pizza", "donut", "cake"
    }
    CABIN_PHONE_CLASSES = {"cell phone"}
    CABIN_TARGET_IDS = [39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 52, 53, 54, 55, 67]

    def __init__(self, mar_window: int = 16, seatbelt_window: int = 25):
        self._mar_history = deque(maxlen=mar_window)
        self._seatbelt_history = deque(maxlen=seatbelt_window)
        self._talking_hold = 0
        self._eating_hold = 0
        self._phone_hold = 0
        self._yawn_hold = 0
        self._frame_idx = 0
        self._cached_hands = []
        self._cached_seatbelt_line = None

        self.hands = None
        if mp_hands is not None:
            try:
                self.hands = mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    model_complexity=0,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.45,
                )
            except Exception as e:
                print(f"[CABIN BEHAVIOR] MediaPipe Hands disabled ({e}).")
                self.hands = None

    def _detect_seatbelt(self, bgr_frame, landmarks):
        """
        Detects diagonal seatbelt strap across the driver's chest ROI anchored
        below the chin landmark (#152) using CLAHE + Canny + HoughLinesP.
        Returns: (seatbelt_status, roi_rect, best_line)
        """
        h, w = bgr_frame.shape[:2]
        chin = landmarks[152]
        forehead = landmarks[10]
        l_cheek = landmarks[234]
        r_cheek = landmarks[454]

        face_h = max(int(abs(chin.y - forehead.y) * h), 20)
        face_w = max(int(abs(r_cheek.x - l_cheek.x) * w), 20)
        face_cx = int(((l_cheek.x + r_cheek.x) / 2.0) * w)
        chin_y = int(chin.y * h)

        # Chest ROI directly below the chin
        y1 = min(h - 1, max(0, chin_y + int(0.12 * face_h)))
        y2 = min(h, chin_y + int(1.45 * face_h))
        x1 = max(0, face_cx - int(1.25 * face_w))
        x2 = min(w, face_cx + int(1.25 * face_w))

        roi_h = y2 - y1
        roi_w = x2 - x1

        # If camera is zoomed tightly on just the face, torso is out of view
        if roi_h < 42 or roi_w < 50:
            return "N/A", None, None

        roi = bgr_frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        edges = cv2.Canny(blurred, 45, 130)

        min_len = max(24, int(min(roi_h, roi_w) * 0.36))
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=28,
            minLineLength=min_len,
            maxLineGap=14,
        )

        best_line = None
        best_len = 0.0
        diagonal_hits = 0

        if lines is not None:
            for line in lines:
                lx1, ly1, lx2, ly2 = line[0]
                dx = lx2 - lx1
                dy = ly2 - ly1
                length = math.hypot(dx, dy)
                if dx == 0:
                    continue
                angle = abs(math.degrees(math.atan2(dy, dx)))
                # Seatbelt straps cross the chest diagonally between ~25 and ~70 degrees
                if 25.0 <= angle <= 70.0 and length >= min_len:
                    diagonal_hits += 1
                    if length > best_len:
                        best_len = length
                        best_line = (x1 + lx1, y1 + ly1, x1 + lx2, y1 + ly2)

        detected_now = 1 if diagonal_hits >= 1 else 0
        self._seatbelt_history.append(detected_now)

        if len(self._seatbelt_history) >= 8:
            ratio = sum(self._seatbelt_history) / len(self._seatbelt_history)
            status = "ON" if ratio >= 0.45 else "OFF"
        else:
            status = "ON" if detected_now else "CHECKING"

        if status == "ON" and best_line is not None:
            self._cached_seatbelt_line = best_line
        elif status == "OFF":
            self._cached_seatbelt_line = None

        return status, (x1, y1, x2, y2), self._cached_seatbelt_line

    def _update_hands(self, rgb_frame):
        """Runs lightweight hand tracking every 3rd frame for eating / phone cues."""
        if self.hands is None:
            return []
        if self._frame_idx % 3 == 0:
            res = self.hands.process(rgb_frame)
            pts = []
            if res.multi_hand_landmarks:
                for hand_lm in res.multi_hand_landmarks:
                    # Index tip (8), middle tip (12), thumb tip (4), wrist (0)
                    pts.append({
                        "index": hand_lm.landmark[8],
                        "middle": hand_lm.landmark[12],
                        "thumb": hand_lm.landmark[4],
                        "wrist": hand_lm.landmark[0],
                    })
            self._cached_hands = pts
        return self._cached_hands

    @staticmethod
    def _dist_norm(p1, p2):
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    def evaluate(self, bgr_frame, rgb_frame, landmarks, mar, cabin_boxes=None):
        """
        Evaluates all in-cabin behaviors for the current frame.

        Args:
            bgr_frame: BGR image (640x480)
            rgb_frame: RGB image (640x480)
            landmarks: 468 FaceMesh landmarks (or None if no face)
            mar: current Mouth Aspect Ratio
            cabin_boxes: list of (x1, y1, x2, y2, label, conf) from cabin YOLO

        Returns:
            dict with keys:
              - seatbelt: "ON" | "OFF" | "N/A" | "CHECKING"
              - seatbelt_roi: (x1, y1, x2, y2) or None
              - seatbelt_line: (x1, y1, x2, y2) or None
              - talking: bool
              - eating_drinking: bool
              - phone_use: bool
              - yawning: bool
              - primary_activity: "ATTENTIVE" | "TALKING" | "EATING/DRINKING" | "PHONE USE" | "YAWNING"
              - activities: list[str]
        """
        self._frame_idx += 1
        cabin_boxes = cabin_boxes or []

        # Object-level signals from YOLO
        yolo_food_drink = any(
            lbl in self.CABIN_FOOD_DRINK_CLASSES for (_, _, _, _, lbl, _) in cabin_boxes
        )
        yolo_phone = any(
            lbl in self.CABIN_PHONE_CLASSES for (_, _, _, _, lbl, _) in cabin_boxes
        )

        if landmarks is None:
            self.reset()
            return {
                "seatbelt": "N/A",
                "seatbelt_roi": None,
                "seatbelt_line": None,
                "talking": False,
                "eating_drinking": yolo_food_drink,
                "phone_use": yolo_phone,
                "yawning": False,
                "primary_activity": "NO DRIVER",
                "activities": [],
            }

        # 1. Seatbelt analysis
        seatbelt_status, sb_roi, sb_line = self._detect_seatbelt(bgr_frame, landmarks)

        # 2. Mouth kinematics (Talking vs Yawning)
        self._mar_history.append(mar)
        is_yawning_now = False
        is_talking_now = False

        if len(self._mar_history) >= 6:
            recent6 = list(self._mar_history)[-6:]
            if sum(1 for m in recent6 if m > 0.44) >= 5:
                is_yawning_now = True

        if not is_yawning_now and len(self._mar_history) >= 10:
            vals = list(self._mar_history)
            mar_std = float(np.std(vals))
            mar_span = float(max(vals) - min(vals))
            mar_mean = float(np.mean(vals))
            # Talking shows clear mouth oscillation without sustained wide yawn
            if mar_std > 0.025 and mar_span > 0.065 and 0.07 <= mar_mean <= 0.42:
                is_talking_now = True

        # 3. Hand proximity cues (Hand-to-mouth -> Eating/Drinking, Hand-to-ear -> Phone call)
        hands = self._update_hands(rgb_frame)
        mouth_pt = landmarks[13]
        left_ear = landmarks[234]
        right_ear = landmarks[454]

        hand_near_mouth = False
        hand_near_ear = False
        for hand in hands:
            for key in ("index", "middle", "thumb"):
                pt = hand[key]
                if self._dist_norm(pt, mouth_pt) < 0.13:
                    hand_near_mouth = True
                if self._dist_norm(pt, left_ear) < 0.14 or self._dist_norm(pt, right_ear) < 0.14:
                    hand_near_ear = True

        # Combine multimodal cues with brief hysteresis to prevent flicker
        eating_now = yolo_food_drink or (hand_near_mouth and mar > 0.09)
        phone_now = yolo_phone or hand_near_ear

        self._yawn_hold = 12 if is_yawning_now else max(0, self._yawn_hold - 1)
        self._talking_hold = 10 if is_talking_now else max(0, self._talking_hold - 1)
        self._eating_hold = 15 if eating_now else max(0, self._eating_hold - 1)
        self._phone_hold = 15 if phone_now else max(0, self._phone_hold - 1)

        yawning = self._yawn_hold > 0
        talking = self._talking_hold > 0 and not eating_now
        eating = self._eating_hold > 0
        phone = self._phone_hold > 0

        activities = []
        if phone:
            activities.append("PHONE USE")
        if eating:
            activities.append("EATING/DRINKING")
        if yawning:
            activities.append("YAWNING")
        if talking:
            activities.append("TALKING")

        if phone:
            primary = "PHONE USE"
        elif eating:
            primary = "EATING/DRINKING"
        elif yawning:
            primary = "YAWNING"
        elif talking:
            primary = "TALKING"
        else:
            primary = "ATTENTIVE"

        return {
            "seatbelt": seatbelt_status,
            "seatbelt_roi": sb_roi,
            "seatbelt_line": sb_line,
            "talking": talking,
            "eating_drinking": eating,
            "phone_use": phone,
            "yawning": yawning,
            "primary_activity": primary,
            "activities": activities,
        }

    def reset(self):
        self._mar_history.clear()
        self._seatbelt_history.clear()
        self._talking_hold = 0
        self._eating_hold = 0
        self._phone_hold = 0
        self._yawn_hold = 0
        self._cached_hands = []
        self._cached_seatbelt_line = None

    def close(self):
        if self.hands is not None:
            try:
                self.hands.close()
            except Exception:
                pass
