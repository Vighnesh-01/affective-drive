import threading
import os


class AudioAlertEngine:
    """
    Instant pre-recorded audio alert engine for Affect-Drive.

    Replaces pyttsx3 TTS synthesis (200-400ms latency) with instant
    pre-recorded WAV file playback via pygame.mixer (<5ms latency).

    Alert WAV files are generated once using generate_alerts.py and stored
    in the alerts/ directory. The engine is thread-safe and non-blocking.

    Usage:
        audio = AudioAlertEngine()
        audio.play("distracted")      # Plays alerts/distracted.wav
        audio.play("hazard_close")    # Plays alerts/hazard_close.wav
        audio.play_for_payload(payload)  # Auto-selects correct alert
    """

    ALERT_MAP = {
        # Maps (driver_status, hazard_key) → alert filename stem
        ("DISTRACTED", "Clear"):         "distracted",
        ("ANXIOUS",    "Clear"):         "anxious",
        ("DISTRACTED", "Close Vehicle"): "distracted_hazard",
        ("ANXIOUS",    "Close Vehicle"): "anxious_hazard",
        ("NORMAL",     "Close Vehicle"): "hazard_close",
    }

    ALERT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "alerts")

    def __init__(self, volume: float = 0.9):
        self.volume = max(0.0, min(1.0, volume))
        self._lock = threading.Lock()
        self._initialized = False
        self._sounds: dict = {}
        self._init_pygame()

    def _init_pygame(self):
        """Initialize pygame.mixer on a background thread to avoid startup blocking."""
        def _init():
            try:
                import pygame
                pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
                pygame.mixer.music.set_volume(self.volume)

                alert_dir = os.path.abspath(self.ALERT_DIR)
                loaded = 0
                for stem in set(self.ALERT_MAP.values()):
                    wav_path = os.path.join(alert_dir, f"{stem}.wav")
                    if os.path.isfile(wav_path):
                        self._sounds[stem] = pygame.mixer.Sound(wav_path)
                        self._sounds[stem].set_volume(self.volume)
                        loaded += 1
                    else:
                        print(f"[AUDIO] WARNING: Alert file not found: {wav_path}")

                self._initialized = loaded > 0
                status = f"{loaded} alert(s) loaded" if loaded > 0 else "NO alerts found — run generate_alerts.py"
                print(f"[AUDIO] pygame.mixer initialized. {status}")
            except Exception as e:
                print(f"[AUDIO] Initialization failed: {e}. Audio alerts will be disabled.")
                self._initialized = False

        threading.Thread(target=_init, daemon=True).start()

    def play(self, alert_name: str):
        """
        Non-blocking playback of a named alert.
        alert_name: stem name like 'distracted', 'hazard_close', etc.
        """
        if not self._initialized:
            print(f"[AUDIO MOCK] Would play: {alert_name}")
            return

        sound = self._sounds.get(alert_name)
        if sound is None:
            print(f"[AUDIO] Alert '{alert_name}' not loaded.")
            return

        def _play():
            with self._lock:
                try:
                    sound.play()
                except Exception as e:
                    print(f"[AUDIO] Playback error for '{alert_name}': {e}")

        threading.Thread(target=_play, daemon=True).start()

    def play_for_payload(self, payload: dict):
        """
        Auto-selects and plays the correct alert based on driver state + road hazard.
        Call this on every HIGH-risk event instead of TTS engine.
        """
        driver_status = payload.get("driver_state", {}).get("driver_status", "NORMAL")
        hazard = payload.get("road_state", {}).get("high_risk_hazard", "Clear")
        hazard_key = "Close Vehicle" if ("Close" in hazard or "Vehicle" in hazard or "Pedestrian" in hazard or "Pothole" in hazard) else "Clear"

        alert_name = self.ALERT_MAP.get((driver_status, hazard_key))
        if alert_name:
            self.play(alert_name)
        else:
            # Fallback to generic hazard tone
            self.play("hazard_close")

    def stop(self):
        """Stops any currently playing audio and uninitializes mixer."""
        try:
            import pygame
            if pygame.mixer.get_init():
                pygame.mixer.stop()
                pygame.mixer.quit()
        except Exception:
            pass
