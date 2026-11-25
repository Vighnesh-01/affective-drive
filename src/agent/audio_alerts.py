import math
import os
import struct
import threading


class AudioAlertEngine:
    """
    Instant ADAS alert chime engine for Affect-Drive.

    Plays brief synthesized vehicle cockpit warning chimes (<5ms latency)
    without overlapping spoken voice recordings over the AI Co-Pilot TTS.
    Also preserves named WAV playback for backward compatibility.
    """

    ALERT_MAP = {
        ("DISTRACTED", "Clear"):         "chime_warning",
        ("ANXIOUS",    "Clear"):         "chime_soft",
        ("STRESSED",   "Clear"):         "chime_soft",
        ("DROWSY",     "Clear"):         "chime_critical",
        ("NO FACE",    "Clear"):         "chime_soft",
        ("DISTRACTED", "Close Vehicle"): "chime_critical",
        ("ANXIOUS",    "Close Vehicle"): "chime_critical",
        ("DROWSY",     "Close Vehicle"): "chime_critical",
        ("NORMAL",     "Close Vehicle"): "chime_warning",
    }

    ALERT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "alerts")

    def __init__(self, volume: float = 0.55):
        self.volume = max(0.0, min(1.0, volume))
        self._lock = threading.Lock()
        self._initialized = False
        self._sounds: dict = {}
        self._init_pygame()

    @staticmethod
    def _build_tone_bytes(freqs, duration_ms=160, sample_rate=22050, amplitude=0.35):
        """Synthesizes a clean multi-tone ADAS chime with smooth fade-out."""
        n_samples = int(sample_rate * (duration_ms / 1000.0))
        buf = bytearray()
        for i in range(n_samples):
            t = i / sample_rate
            # Smooth exponential envelope to eliminate clicks
            env = math.exp(-3.5 * (i / max(n_samples, 1)))
            if i < 200:
                env *= (i / 200.0)
            val = sum(math.sin(2.0 * math.pi * f * t) for f in freqs) / max(len(freqs), 1)
            sample = int(max(-1.0, min(1.0, val * env * amplitude)) * 32767)
            buf.extend(struct.pack("<h", sample))
        return bytes(buf)

    def _init_pygame(self):
        """Initialize pygame.mixer and synthesize ADAS alert chimes on a background thread."""
        def _init():
            try:
                import pygame
                pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)

                # Synthesize non-speech ADAS cockpit chimes in RAM
                chime_specs = {
                    "chime_soft":     ([587.33, 880.0], 160, 0.30),
                    "chime_warning":  ([783.99, 1046.5], 190, 0.38),
                    "chime_critical": ([987.77, 1318.5], 240, 0.45),
                }
                for name, (freqs, dur, amp) in chime_specs.items():
                    pcm = self._build_tone_bytes(freqs, duration_ms=dur, amplitude=amp)
                    snd = pygame.mixer.Sound(buffer=pcm)
                    snd.set_volume(self.volume)
                    self._sounds[name] = snd

                # Optionally load custom WAVs under their stem names
                alert_dir = os.path.abspath(self.ALERT_DIR)
                if os.path.isdir(alert_dir):
                    for fname in os.listdir(alert_dir):
                        if fname.endswith(".wav"):
                            stem = fname[:-4]
                            if stem not in self._sounds:
                                wav_path = os.path.join(alert_dir, fname)
                                snd = pygame.mixer.Sound(wav_path)
                                snd.set_volume(self.volume)
                                self._sounds[stem] = snd

                self._initialized = len(self._sounds) > 0
                print(f"[AUDIO] ADAS chime engine initialized ({len(self._sounds)} sounds ready).")
            except Exception as e:
                print(f"[AUDIO] Initialization failed: {e}. Audio chimes disabled.")
                self._initialized = False

        threading.Thread(target=_init, daemon=True).start()

    def play(self, alert_name: str):
        """Non-blocking playback of a named chime or alert."""
        if not self._initialized:
            return

        sound = self._sounds.get(alert_name) or self._sounds.get("chime_warning")
        if sound is None:
            return

        def _play():
            with self._lock:
                try:
                    sound.play()
                except Exception as e:
                    print(f"[AUDIO] Playback error for '{alert_name}': {e}")

        threading.Thread(target=_play, daemon=True).start()

    def play_for_payload(self, payload: dict):
        """Plays an appropriate short ADAS chime based on driver state + road hazard."""
        ds = payload.get("driver_state", {})
        driver_status = ds.get("driver_status", "NORMAL")
        activity = ds.get("activity", "ATTENTIVE")
        seatbelt = ds.get("seatbelt", "N/A")
        hazard = payload.get("road_state", {}).get("high_risk_hazard", "Clear")

        has_close = "Close" in str(hazard) or "Vehicle" in str(hazard) or "Pedestrian" in str(hazard)
        hazard_key = "Close Vehicle" if has_close else "Clear"

        if activity in ("PHONE USE", "YAWNING") or driver_status == "DROWSY":
            self.play("chime_critical")
            return
        if activity == "EATING/DRINKING" or seatbelt == "OFF":
            self.play("chime_warning")
            return

        alert_name = self.ALERT_MAP.get((driver_status, hazard_key), "chime_warning")
        self.play(alert_name)

    def stop(self):
        """Stops any currently playing audio and uninitializes mixer."""
        try:
            import pygame
            if pygame.mixer.get_init():
                pygame.mixer.stop()
                pygame.mixer.quit()
        except Exception:
            pass
