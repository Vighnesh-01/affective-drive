import os
import csv
import time
import textwrap
import numpy as np
import cv2

from src.perception.fusion_pipeline import AffectivePerceptionEngine
from src.agent.llm_intervention import InterventionAgent
from src.agent.tts_engine import TTSEngine
from telegram_notifier import TelegramNotifier
from src.agent.audio_alerts import AudioAlertEngine


class DatasetActuator:
    """
    Software-only replacement for HardwareController (Arduino).
    Logs all actuation signals, emotions, cabin activities, AI decisions, and
    driver/road state payloads to a timestamped CSV file for offline analysis
    and demo presentation — zero serial hardware required.
    """
    def __init__(self, output_csv="data/actuation_log.csv"):
        self.output_csv = output_csv
        self.signal_count = 0
        self.high_count = 0
        self.low_count = 0
        self.start_time = time.time()

        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        with open(self.output_csv, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "frame_id",
                "timestamp",
                "elapsed_sec",
                "risk_level",
                "signal",
                "driver_status",
                "emotion",
                "seatbelt",
                "activity",
                "road_hazard",
                "ai_intervention",
            ])
        print(f"[DATASET ACTUATOR] Logging actuation signals to: {self.output_csv}")

    def send_actuation_signal(self, risk_level, payload=None, ai_response=None):
        """Logs each actuation event to CSV instead of sending serial data."""
        signal = "H" if risk_level == "HIGH" else "L"
        self.signal_count += 1
        if risk_level == "HIGH":
            self.high_count += 1
        else:
            self.low_count += 1

        elapsed = round(time.time() - self.start_time, 3)

        ds = payload.get("driver_state", {}) if payload else {}
        driver_status = ds.get("driver_status", "")
        emotion = ds.get("emotion", "")
        seatbelt = ds.get("seatbelt", "")
        activity = ds.get("activity", "")
        road_hazard = payload.get("road_state", {}).get("high_risk_hazard", "") if payload else ""

        with open(self.output_csv, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                self.signal_count,
                time.strftime("%H:%M:%S"),
                elapsed,
                risk_level,
                signal,
                driver_status,
                emotion,
                seatbelt,
                activity,
                road_hazard,
                ai_response or "",
            ])

    def get_stats(self):
        """Returns session statistics for the summary report."""
        total_time = round(time.time() - self.start_time, 1)
        return {
            "total_frames": self.signal_count,
            "high_risk_count": self.high_count,
            "low_risk_count": self.low_count,
            "high_risk_pct": round(self.high_count / max(self.signal_count, 1) * 100, 1),
            "session_duration_sec": total_time,
        }

    def close(self):
        """Prints session summary on shutdown."""
        stats = self.get_stats()
        print("\n" + "=" * 60)
        print("  DATASET ACTUATOR — SESSION SUMMARY")
        print("=" * 60)
        print(f"  Total Frames Processed : {stats['total_frames']}")
        print(f"  HIGH Risk Events       : {stats['high_risk_count']} ({stats['high_risk_pct']}%)")
        print(f"  LOW  Risk Events       : {stats['low_risk_count']}")
        print(f"  Session Duration       : {stats['session_duration_sec']}s")
        print(f"  Output File            : {self.output_csv}")
        print("=" * 60 + "\n")


def _draw_footer(canvas, risk_level, ai_mode_label, ai_text):
    h, w = canvas.shape[:2]
    footer_h = 64
    cv2.rectangle(canvas, (0, h - footer_h), (w, h), (12, 12, 12), -1)
    cv2.line(canvas, (0, h - footer_h), (w, h - footer_h), (60, 60, 60), 1)
    text_color = (80, 90, 255) if risk_level == "HIGH" else (80, 240, 120)

    wrapped = textwrap.wrap(f"AI Co-Pilot [{ai_mode_label}]: {ai_text}", width=105)
    line1 = wrapped[0] if wrapped else f"AI Co-Pilot: {ai_text}"
    line2 = wrapped[1] if len(wrapped) > 1 else ""

    cv2.putText(canvas, line1, (12, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.54, text_color, 2, cv2.LINE_AA)
    if line2:
        cv2.putText(canvas, line2, (12, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, text_color, 1, cv2.LINE_AA)

    ctrl = "[CAR DATASET MODE — No Arduino] | [Q] Quit"
    (cw, _), _ = cv2.getTextSize(ctrl, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
    cv2.putText(canvas, ctrl, (w - cw - 12, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 200, 255), 1, cv2.LINE_AA)


def main():
    print("=" * 60)
    print("  AFFECT-DRIVE — CAR SCENARIO DATASET MODE (No Arduino)")
    print("=" * 60)

    perception = AffectivePerceptionEngine(
        driver_src="data/sample_driver.mp4",
        road_src="data/sample_road.mp4",
        skip_road_frames=2,
        target_resolution=(640, 480),
    )

    agent = InterventionAgent(mode="auto")
    actuator = DatasetActuator(output_csv="data/actuation_log.csv")
    tts = TTSEngine(voice_gender="female", rate=165, pitch=0)
    notifier = TelegramNotifier(
        bot_token="8602864132:AAEwK3rR4PMDcmdK67tnI7bngS-e0QKJigs",
        chat_id="6167292056",
        cooldown_seconds=6.0,
        mock_mode=False,
    )
    audio = AudioAlertEngine(volume=0.55)

    last_intervention_time = 0.0
    cooldown_seconds = 5.0
    last_risk_state = "LOW"
    last_driver_status = "NORMAL"
    last_activity = "ATTENTIVE"
    latest_ai_warning = "All systems nominal. Safe driving conditions ahead."

    cockpit_canvas = np.zeros((544, 1280, 3), dtype=np.uint8)

    def on_ai_response(response_text, speak=True):
        nonlocal latest_ai_warning
        latest_ai_warning = response_text
        print(f"\n[AI CO-PILOT INTERVENTION]: {response_text}\n")
        if speak:
            tts.speak_async(response_text)

    try:
        while True:
            d_frame, r_frame, payload = perception.step()

            if d_frame is None or r_frame is None or payload is None:
                time.sleep(0.01)
                continue

            current_time = time.time()
            risk_level = payload.get("system_risk_level", "LOW")
            ds = payload.get("driver_state", {})
            driver_status = ds.get("driver_status", "NORMAL")
            activity = ds.get("activity", "ATTENTIVE")
            ai_text = None

            state_changed = (driver_status != last_driver_status) or (activity != last_activity)
            cooldown_elapsed = (current_time - last_intervention_time) > cooldown_seconds

            # 1. Dynamic Intervention Trigger
            if risk_level == "HIGH" and (last_risk_state == "LOW" or state_changed or cooldown_elapsed):
                def _on_high(txt):
                    nonlocal ai_text
                    ai_text = txt
                    on_ai_response(txt, speak=True)

                agent.evaluate_and_respond_async(payload, callback=_on_high)
                audio.play_for_payload(payload)
                if driver_status != "NO FACE":
                    notifier.send_alert(payload, snapshot_frame=d_frame)
                last_intervention_time = current_time

            elif risk_level == "LOW" and last_risk_state == "HIGH":
                agent.evaluate_and_respond_async(
                    payload, callback=lambda txt: on_ai_response(txt, speak=False)
                )
            elif risk_level == "LOW" and activity == "TALKING" and last_activity != "TALKING" and cooldown_elapsed:
                agent.evaluate_and_respond_async(
                    payload, callback=lambda txt: on_ai_response(txt, speak=True)
                )
                last_intervention_time = current_time

            last_risk_state = risk_level
            last_driver_status = driver_status
            last_activity = activity

            # 2. Log actuation to CSV dataset
            actuator.send_actuation_signal(risk_level, payload=payload, ai_response=ai_text)

            # 3. Render Dashboard + Footer
            cockpit_canvas[:480, :640] = d_frame
            cockpit_canvas[:480, 640:] = r_frame
            _draw_footer(cockpit_canvas, risk_level, agent.mode.upper(), latest_ai_warning)

            cv2.imshow("Affective-Drive Master Control", cockpit_canvas)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        perception.stop()
        actuator.close()
        tts.stop()
        audio.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
