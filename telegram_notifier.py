import threading
import time
import requests
import io
import cv2


class TelegramNotifier:
    """
    Optimized, Asynchronous Telegram Safety Alert Bot for Affect-Drive.

    Sends HIGH-risk events as clean HTML alerts + JPEG snapshots to
    the driver/fleet manager's Telegram chat in a non-blocking background thread.

    Features:
    - HTML parse mode (immune to markdown escaping syntax errors).
    - Lightweight JPEG snapshot compression for instant (<100ms) dispatch.
    - Configurable cooldown to prevent notification flooding.
    - Thread-safe non-blocking execution.
    """

    def __init__(
        self,
        bot_token: str = "8602864132:AAEwK3rR4PMDcmdK67tnI7bngS-e0QKJigs",
        chat_id: str = "6167292056",
        cooldown_seconds: float = 6.0,
        mock_mode: bool = False
    ):
        self.bot_token = bot_token.strip()
        self.chat_id = str(chat_id).strip()
        self.cooldown_seconds = cooldown_seconds
        self._last_sent_time = 0.0
        self._session = requests.Session()
        self._session.timeout = 8.0

        # Enable live mode if credentials exist and mock_mode is not explicitly forced
        self.mock_mode = mock_mode or not self.bot_token or not self.chat_id
        if self.mock_mode:
            print("[TELEGRAM] Running in MOCK MODE — alerts will be printed to console only.")
        else:
            print(f"[TELEGRAM] Live Bot Connected. Target Chat ID: {self.chat_id}")

    def send_alert(self, payload: dict, snapshot_frame=None):
        """Dispatches alert asynchronously in a background thread."""
        now = time.time()
        if now - self._last_sent_time < self.cooldown_seconds:
            return  # Within cooldown window
        self._last_sent_time = now

        # Fire and forget in daemon thread
        threading.Thread(
            target=self._dispatch,
            args=(payload, snapshot_frame.copy() if snapshot_frame is not None else None),
            daemon=True
        ).start()

    def _build_html_message(self, payload: dict) -> str:
        """Constructs a clean HTML alert message that never breaks on special chars."""
        driver_status = payload.get("driver_state", {}).get("driver_status", "UNKNOWN")
        road_hazard = payload.get("road_state", {}).get("high_risk_hazard", "Unknown")
        risk_level = payload.get("system_risk_level", "HIGH")
        timestamp = time.strftime("%d %b %Y | %H:%M:%S")

        status_emoji = "👁️" if driver_status == "DISTRACTED" else ("😰" if driver_status == "ANXIOUS" else "✅")
        hazard_emoji = "🚨" if ("Close" in road_hazard or "Vehicle" in road_hazard or "Pothole" in road_hazard) else "⚠️"

        html = (
            f"<b>🚗 Affect-Drive Safety Alert</b>\n"
            f"────────────────────\n"
            f"{status_emoji} <b>Driver State:</b> <code>{driver_status}</code>\n"
            f"{hazard_emoji} <b>Road Hazard:</b> <code>{road_hazard}</code>\n"
            f"🔴 <b>System Risk:</b> <code>{risk_level}</code>\n"
            f"🕐 <b>Time:</b> {timestamp}\n"
            f"────────────────────\n"
            f"<i>Automated alert from Affect-Drive AI Cockpit</i>"
        )
        return html

    def _encode_snapshot(self, frame) -> bytes | None:
        """Downsamples and compresses frame to fast JPEG bytes (~35KB)."""
        if frame is None:
            return None
        try:
            h, w = frame.shape[:2]
            if w > 640:
                scale = 640.0 / w
                frame = cv2.resize(frame, (640, int(h * scale)), interpolation=cv2.INTER_AREA)

            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
            return buf.tobytes() if ok else None
        except Exception as e:
            print(f"[TELEGRAM ENCODE ERROR] {e}")
            return None

    def _dispatch(self, payload: dict, snapshot_frame):
        """Background worker: sends snapshot photo + caption or text message."""
        message = self._build_html_message(payload)

        if self.mock_mode:
            clean = message.replace("<b>", "").replace("</b>", "").replace("<code>", "").replace("</code>", "").replace("<i>", "").replace("</i>", "")
            clean = clean.encode("ascii", errors="replace").decode("ascii")
            print(f"\n[TELEGRAM MOCK ALERT]\n{clean}\n")
            return

        base_url = f"https://api.telegram.org/bot{self.bot_token}"

        # 1. Try sending photo with HTML caption
        if snapshot_frame is not None:
            jpeg_bytes = self._encode_snapshot(snapshot_frame)
            if jpeg_bytes:
                try:
                    resp = self._session.post(
                        f"{base_url}/sendPhoto",
                        data={
                            "chat_id": self.chat_id,
                            "caption": message,
                            "parse_mode": "HTML"
                        },
                        files={"photo": ("alert_snapshot.jpg", io.BytesIO(jpeg_bytes), "image/jpeg")},
                        timeout=8.0
                    )
                    if resp.status_code == 200:
                        print(f"[TELEGRAM] Alert + Snapshot sent successfully to Chat {self.chat_id}")
                        return
                    else:
                        print(f"[TELEGRAM WARNING] Photo send failed ({resp.status_code}: {resp.text}), attempting text fallback...")
                except Exception as e:
                    print(f"[TELEGRAM DISPATCH ERROR] {e}")

        # 2. Fallback: send text message
        try:
            resp = self._session.post(
                f"{base_url}/sendMessage",
                json={
                    "chat_id": self.chat_id,
                    "text": message,
                    "parse_mode": "HTML"
                },
                timeout=6.0
            )
            if resp.status_code == 200:
                print(f"[TELEGRAM] Text alert sent to Chat {self.chat_id}")
            else:
                print(f"[TELEGRAM ERROR] Message send failed ({resp.status_code}: {resp.text})")
        except Exception as e:
            print(f"[TELEGRAM DISPATCH ERROR] {e}")
