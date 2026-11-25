"""
Facial Emotion Detection using MediaPipe Face Mesh Landmarks.

Implements FACS (Facial Action Coding System) geometric analysis to classify
driver emotions from 468 face mesh landmarks — zero extra ML models or
dependencies required (reuses the same FaceMesh already running in the pipeline).

Detected emotions:  neutral · happy · sad · angry · surprise · fear · drowsy

Geometric features computed:
  - EAR  (Eye Aspect Ratio)       -> drowsiness / eye closure
  - MAR  (Mouth Aspect Ratio)     -> surprise / yawning / talking
  - Brow-Eye Distance Ratio       -> surprise / anger
  - Brow Furrow (inner brow gap)  -> anger / stress
  - Lip Corner Ratio & Mouth Width -> happiness / sadness

Temporal smoothing over a configurable sliding window eliminates single-frame
noise and delivers stable, presentation-ready predictions.
"""

from collections import deque


class FacialEmotionDetector:
    """
    Real-time facial emotion detector driven by landmark geometry.

    Usage::

        detector = FacialEmotionDetector()
        # `landmarks` is the list of 468 NormalizedLandmark from MediaPipe FaceMesh
        emotion, scores, metrics = detector.predict(landmarks)
    """

    # ── MediaPipe Face Mesh landmark indices ───────────────────────────────
    # Left eye contour (6 pts for EAR)
    LEFT_EYE  = [33, 160, 158, 133, 153, 144]
    # Right eye contour
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    # Eyebrow key points
    LEFT_BROW_INNER   = 107
    LEFT_BROW_OUTER   = 70
    LEFT_BROW_CENTER  = 105
    RIGHT_BROW_INNER  = 336
    RIGHT_BROW_OUTER  = 300
    RIGHT_BROW_CENTER = 334

    # Eye centre (upper lid midpoint)
    LEFT_EYE_CENTER  = 159
    RIGHT_EYE_CENTER = 386

    # Mouth landmarks
    MOUTH_LEFT   = 61
    MOUTH_RIGHT  = 291
    MOUTH_TOP    = 13
    MOUTH_BOTTOM = 14

    # Face reference
    FOREHEAD    = 10
    CHIN        = 152
    NOSE_TIP    = 1
    LEFT_CHEEK  = 234
    RIGHT_CHEEK = 454

    EMOTIONS = ["neutral", "happy", "sad", "angry", "surprise", "fear", "drowsy"]

    # ── Constructor ────────────────────────────────────────────────────────
    def __init__(self, smoothing_window: int = 12):
        """
        Args:
            smoothing_window: frames to average over for temporal smoothing.
        """
        self.smoothing_window = smoothing_window
        self._score_history = deque(maxlen=smoothing_window)
        self._ear_history   = deque(maxlen=smoothing_window)

    # ── Low-level geometry helpers ─────────────────────────────────────────
    @staticmethod
    def _dist(p1, p2):
        return ((p1.x - p2.x) ** 2 +
                (p1.y - p2.y) ** 2 +
                (p1.z - p2.z) ** 2) ** 0.5

    @staticmethod
    def _dist_2d(p1, p2):
        return ((p1.x - p2.x) ** 2 +
                (p1.y - p2.y) ** 2) ** 0.5

    # ── Feature extractors ─────────────────────────────────────────────────
    def compute_ear(self, lm):
        """Eye Aspect Ratio. Normal ≈ 0.24-0.35 · Drowsy < 0.19"""
        def _one_eye(idx):
            p = [lm[i] for i in idx]
            v1 = self._dist_2d(p[1], p[5])
            v2 = self._dist_2d(p[2], p[4])
            h  = self._dist_2d(p[0], p[3])
            return (v1 + v2) / (2.0 * h + 1e-6)
        return (_one_eye(self.LEFT_EYE) + _one_eye(self.RIGHT_EYE)) / 2.0

    def compute_mar(self, lm):
        """Mouth Aspect Ratio. Closed ≈ 0.01-0.08 · Talking 0.10-0.35 · Yawn/Surprise > 0.45"""
        v = self._dist_2d(lm[self.MOUTH_TOP], lm[self.MOUTH_BOTTOM])
        h = self._dist_2d(lm[self.MOUTH_LEFT], lm[self.MOUTH_RIGHT])
        return v / (h + 1e-6)

    def compute_brow_eye_ratio(self, lm):
        """Brow-Eye height normalised by face height.
        Neutral ≈ 0.060-0.105 · High (>0.115) -> surprise/fear · Low (<0.052) -> anger."""
        fh = self._dist_2d(lm[self.FOREHEAD], lm[self.CHIN]) + 1e-6
        ld = abs(lm[self.LEFT_EYE_CENTER].y - lm[self.LEFT_BROW_CENTER].y)
        rd = abs(lm[self.RIGHT_EYE_CENTER].y - lm[self.RIGHT_BROW_CENTER].y)
        return ((ld + rd) / 2.0) / fh

    def compute_brow_furrow(self, lm):
        """Inner-brow gap / outer-brow gap. Neutral ≈ 0.21-0.32 · Very low (<0.20) -> furrowed."""
        inner = self._dist_2d(lm[self.LEFT_BROW_INNER], lm[self.RIGHT_BROW_INNER])
        outer = self._dist_2d(lm[self.LEFT_BROW_OUTER],  lm[self.RIGHT_BROW_OUTER])
        return inner / (outer + 1e-6)

    def compute_lip_corner_ratio(self, lm):
        """Lip-corner elevation relative to mouth center midpoint.
        Neutral ≈ -0.010 to +0.010 · Positive (>0.014) -> smile · Negative (<-0.016) -> frown."""
        corner_y = (lm[self.MOUTH_LEFT].y + lm[self.MOUTH_RIGHT].y) / 2.0
        mouth_mid_y = (lm[self.MOUTH_TOP].y + lm[self.MOUTH_BOTTOM].y) / 2.0
        fh = self._dist_2d(lm[self.FOREHEAD], lm[self.CHIN]) + 1e-6
        # In image coords y increases downward: mouth_mid_y > corner_y when corners lift (smile)
        return (mouth_mid_y - corner_y) / fh

    def compute_mouth_width_ratio(self, lm):
        """Mouth width relative to face width. Neutral ≈ 0.36-0.43 · Smile > 0.45."""
        mw = self._dist_2d(lm[self.MOUTH_LEFT], lm[self.MOUTH_RIGHT])
        fw = self._dist_2d(lm[self.LEFT_CHEEK], lm[self.RIGHT_CHEEK]) + 1e-6
        return mw / fw

    # ── Core scoring logic ─────────────────────────────────────────────────
    def _raw_scores(self, lm):
        ear         = self.compute_ear(lm)
        mar         = self.compute_mar(lm)
        brow_eye    = self.compute_brow_eye_ratio(lm)
        brow_fur    = self.compute_brow_furrow(lm)
        lip_corner  = self.compute_lip_corner_ratio(lm)
        mouth_width = self.compute_mouth_width_ratio(lm)

        metrics = {
            "ear":         round(ear, 4),
            "mar":         round(mar, 4),
            "brow_eye":    round(brow_eye, 4),
            "brow_fur":    round(brow_fur, 4),
            "lip_corner":  round(lip_corner, 4),
            "mouth_width": round(mouth_width, 4),
        }

        s = {e: 0.0 for e in self.EMOTIONS}

        # ── Drowsy (sustained low EAR handled in predict()) ─────────────
        if ear < 0.165:
            s["drowsy"] = 0.75
        elif ear < 0.19:
            s["drowsy"] = 0.35

        # ── Happy (corners lifted OR wide smile) ───────────────────────
        if lip_corner > 0.015 or (lip_corner > 0.008 and mouth_width > 0.46):
            s["happy"] = min(max(lip_corner * 30.0, (mouth_width - 0.42) * 8.0), 0.90)
        elif lip_corner > 0.011:
            s["happy"] = 0.35

        # ── Sad (clear downward lip corners without scowl) ─────────────
        if lip_corner < -0.018:
            s["sad"] = min(abs(lip_corner + 0.012) * 25.0, 0.75)

        # ── Surprise (high brows + wide open mouth) ────────────────────
        if brow_eye > 0.115 and mar > 0.35:
            s["surprise"] = 0.85
        elif brow_eye > 0.115 and mar > 0.22:
            s["surprise"] = 0.50

        # ── Angry (genuinely lowered brows + tight furrow) ─────────────
        if brow_fur < 0.20 and brow_eye < 0.055:
            s["angry"] = 0.80
        elif brow_fur < 0.225 and brow_eye < 0.052:
            s["angry"] = 0.60
        elif brow_eye < 0.048 and lip_corner < -0.010:
            s["angry"] = 0.55

        # ── Fear (raised brows + wide eyes + tense mid-open mouth) ─────
        if brow_eye > 0.112 and ear > 0.31 and 0.14 < mar < 0.38:
            s["fear"] = 0.60

        # ── Neutral (strong baseline when relaxed) ─────────────────────
        peak = max(s.values())
        if peak < 0.20:
            s["neutral"] = 0.85
        elif peak < 0.35:
            s["neutral"] = 0.55
        elif peak < 0.50:
            s["neutral"] = 0.30
        else:
            s["neutral"] = 0.08

        # Normalise
        total = sum(s.values()) + 1e-6
        s = {k: round(v / total, 3) for k, v in s.items()}
        return s, metrics

    # ── Public API ─────────────────────────────────────────────────────────
    def predict(self, landmarks):
        """
        Predict the dominant facial emotion with temporal smoothing.

        Args:
            landmarks: list[NormalizedLandmark] — 468 landmarks from FaceMesh.

        Returns:
            (emotion_label, smoothed_scores, raw_metrics)
        """
        raw, metrics = self._raw_scores(landmarks)
        self._score_history.append(raw)
        self._ear_history.append(metrics["ear"])

        # Temporal average
        if len(self._score_history) >= 3:
            smoothed = {}
            for em in self.EMOTIONS:
                vals = [h[em] for h in self._score_history]
                smoothed[em] = round(sum(vals) / len(vals), 3)
        else:
            smoothed = dict(raw)

        # Sustained drowsiness check (avoids false positives on quick 3-4 frame blinks)
        if len(self._ear_history) >= 10:
            recent = list(self._ear_history)[-10:]
            if sum(1 for e in recent if e < 0.19) >= 8:
                smoothed["drowsy"] = max(smoothed["drowsy"], 0.78)
        if len(self._ear_history) >= 6:
            recent6 = list(self._ear_history)[-6:]
            if all(e < 0.16 for e in recent6):
                smoothed["drowsy"] = max(smoothed["drowsy"], 0.85)

        emotion = max(smoothed, key=smoothed.get)
        return emotion, smoothed, metrics

    def reset(self):
        """Clear temporal buffers (call when face is lost / reacquired)."""
        self._score_history.clear()
        self._ear_history.clear()
