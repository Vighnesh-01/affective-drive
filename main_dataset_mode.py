import os
import csv
import time
import numpy as np
import cv2
from src.perception.fusion_pipeline import AffectivePerceptionEngine
from src.agent.llm_intervention import InterventionAgent
from src.agent.tts_engine import TTSEngine


class DatasetActuator:
    """
    Software-only replacement for HardwareController (Arduino).
    Logs all actuation signals, AI decisions, and driver/road state payloads
    to a timestamped CSV file for offline analysis, model evaluation, and
    demo presentation — zero serial hardware required.
    """
    def __init__(self, output_csv="data/actuation_log.csv"):
        self.output_csv = output_csv
        self.signal_count = 0
        self.high_count = 0
        self.low_count = 0
        self.start_time = time.time()

        # Create CSV with header
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        with open(self.output_csv, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "frame_id",
                "timestamp",
                "elapsed_sec",
                "risk_level",
                "signal",
                "driver_status",
                "road_hazard",
                "ai_intervention"
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

        driver_status = ""
        road_hazard = ""
        if payload:
            driver_status = payload.get("driver_state", {}).get("driver_status", "")
            road_hazard = payload.get("road_state", {}).get("high_risk_hazard", "")

        with open(self.output_csv, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                self.signal_count,
                time.strftime("%H:%M:%S"),
                elapsed,
                risk_level,
                signal,
                driver_status,
                road_hazard,
                ai_response or ""
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


def main():
    print("=" * 60)
    print("  AFFECT-DRIVE — DATASET MODE (No Arduino Required)")
    print("  Hardware: i5-1235U / 8 GB RAM / Intel UHD iGPU")
    print("=" * 60)

    # Use pre-recorded driver video instead of live webcam (src=0)
    # Change driver_src to 0 for live webcam if needed
    perception = AffectivePerceptionEngine(
        driver_src="data/sample_driver.mp4",
        road_src="data/sample_road.mp4",
        skip_road_frames=2,
        target_resolution=(640, 480)
    )

    agent = InterventionAgent(mode="rule_based")
    actuator = DatasetActuator(output_csv="data/actuation_log.csv")
    tts = TTSEngine(voice_gender="female", rate=165, pitch=2)

    last_intervention_time = 0
    cooldown_seconds = 5.0
    last_risk_state = "LOW"
    latest_ai_warning = "System Normal"

    # Pre-allocate dashboard canvas (640*2 x 480) — zero per-frame allocation
    cockpit_canvas = np.zeros((480, 1280, 3), dtype=np.uint8)

    def on_ai_response(response_text):
        nonlocal latest_ai_warning
        latest_ai_warning = response_text
        print(f"\n[AI CO-PILOT INTERVENTION]: {response_text}\n")
        tts.speak_async(response_text)

    try:
        while True:
            d_frame, r_frame, payload = perception.step()

            if d_frame is None or r_frame is None or payload is None:
                time.sleep(0.01)
                continue

            current_time = time.time()
            risk_level = payload.get("system_risk_level", "LOW")
            ai_text = None

            # 1. Intervention Trigger
            if risk_level == "HIGH" and (last_risk_state == "LOW" or (current_time - last_intervention_time > cooldown_seconds)):
                ai_text = agent._rule_based_response(payload)
                agent.evaluate_and_respond_async(payload, callback=on_ai_response)
                last_intervention_time = current_time

            last_risk_state = risk_level

            # 2. Log actuation to CSV dataset (replaces Arduino serial)
            actuator.send_actuation_signal(risk_level, payload=payload, ai_response=ai_text)

            # 3. Optimized In-Place Dashboard Render
            cockpit_canvas[:, :640] = d_frame
            cockpit_canvas[:, 640:] = r_frame

            banner_color = (0, 0, 255) if risk_level == "HIGH" else (0, 255, 0)
            cv2.rectangle(cockpit_canvas, (0, 440), (1280, 480), (0, 0, 0), -1)
            cv2.putText(
                cockpit_canvas,
                f"AI Co-Pilot: {latest_ai_warning}",
                (10, 468),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                banner_color,
                2
            )

            # Mode indicator
            cv2.putText(cockpit_canvas, "[DATASET MODE — No Arduino]", (880, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

            cv2.imshow("Affective-Drive Master Control", cockpit_canvas)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        perception.stop()
        actuator.close()
        tts.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

