import argparse
import textwrap
import time
import cv2
import numpy as np

from src.perception.fusion_pipeline import AffectivePerceptionEngine
from src.agent.llm_intervention import InterventionAgent
from src.hardware.serial_controller import HardwareController
from src.agent.tts_engine import TTSEngine
from src.agent.audio_alerts import AudioAlertEngine
from telegram_notifier import TelegramNotifier


def _draw_footer(canvas, risk_level, ai_mode_label, source_label, ai_text):
    """Renders a crisp, two-line wrapped AI Co-Pilot & Cockpit Controls footer."""
    h, w = canvas.shape[:2]
    footer_h = 64
    cv2.rectangle(canvas, (0, h - footer_h), (w, h), (12, 12, 12), -1)
    cv2.line(canvas, (0, h - footer_h), (w, h - footer_h), (60, 60, 60), 1)

    text_color = (80, 90, 255) if risk_level == "HIGH" else (80, 240, 120)

    # Top footer line: AI Co-Pilot message (wrapped cleanly if long)
    wrapped = textwrap.wrap(f"AI Co-Pilot [{ai_mode_label}]: {ai_text}", width=105)
    line1 = wrapped[0] if wrapped else f"AI Co-Pilot: {ai_text}"
    line2 = wrapped[1] if len(wrapped) > 1 else ""

    cv2.putText(canvas, line1, (12, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.54, text_color, 2, cv2.LINE_AA)
    if line2:
        cv2.putText(canvas, line2, (12, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, text_color, 1, cv2.LINE_AA)

    # Bottom-right controls & source indicator
    ctrl_text = f"SRC: {source_label} | [C] Switch Car/Webcam | [Q] Quit"
    (cw, _), _ = cv2.getTextSize(ctrl_text, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
    cv2.putText(canvas, ctrl_text, (w - cw - 12, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (190, 190, 190), 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="Affective-Drive AI Cockpit System")
    parser.add_argument(
        "--mode",
        choices=["car", "webcam"],
        default="car",
        help="Driver stream source mode: 'car' (in-cabin video) or 'webcam' (live camera 0)",
    )
    parser.add_argument("--driver-src", default=None, help="Override driver video path or camera index")
    parser.add_argument("--road-src", default="data/sample_road.mp4", help="Road dashcam video path or camera index")
    parser.add_argument(
        "--ai-mode",
        choices=["auto", "gemini", "contextual", "local_llm"],
        default="auto",
        help="AI Co-Pilot engine ('auto' uses Gemini if API key set, else contextual)",
    )
    args = parser.parse_args()

    car_video_path = "data/sample_driver.mp4"
    if args.driver_src is not None:
        driver_src = int(args.driver_src) if args.driver_src.isdigit() else args.driver_src
        active_mode = "webcam" if isinstance(driver_src, int) else "car"
    elif args.mode == "webcam":
        driver_src = 0
        active_mode = "webcam"
    else:
        driver_src = car_video_path
        active_mode = "car"

    print("=== Starting Affective-Drive AI Cockpit System ===")
    print(f"[SYSTEM] Driver Mode: {active_mode.upper()} ({driver_src}) | Road Feed: {args.road_src}")

    # 1. Initialize Perception, AI Co-Pilot & Hardware Modules
    perception = AffectivePerceptionEngine(
        driver_src=driver_src,
        road_src=args.road_src,
        skip_road_frames=2,
        target_resolution=(640, 480),
    )
    agent = InterventionAgent(mode=args.ai_mode)
    actuator = HardwareController()
    tts = TTSEngine(voice_gender="female", rate=165, pitch=0)

    # 2. Safety Interventions & Telemetry Bot
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

    # Pre-allocated canvas: 1280 x (480 + 64 footer)
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

            # 1. Hardware Actuation
            actuator.send_actuation_signal(risk_level)

            # 2. Dynamic AI Co-Pilot Interventions
            state_changed = (driver_status != last_driver_status) or (activity != last_activity)
            cooldown_elapsed = (current_time - last_intervention_time) > cooldown_seconds

            if risk_level == "HIGH" and (last_risk_state == "LOW" or state_changed or cooldown_elapsed):
                agent.evaluate_and_respond_async(
                    payload, callback=lambda txt: on_ai_response(txt, speak=True)
                )
                audio.play_for_payload(payload)
                if driver_status != "NO FACE":
                    notifier.send_alert(payload, snapshot_frame=d_frame)
                last_intervention_time = current_time

            elif risk_level == "LOW" and last_risk_state == "HIGH":
                # Smoothly clear stale high-risk banner once driver returns to normal
                agent.evaluate_and_respond_async(
                    payload, callback=lambda txt: on_ai_response(txt, speak=False)
                )
            elif risk_level == "LOW" and activity == "TALKING" and last_activity != "TALKING" and cooldown_elapsed:
                # Subtle Co-Pilot awareness prompt when conversation begins
                agent.evaluate_and_respond_async(
                    payload, callback=lambda txt: on_ai_response(txt, speak=True)
                )
                last_intervention_time = current_time

            last_risk_state = risk_level
            last_driver_status = driver_status
            last_activity = activity

            # 3. Render Side-by-Side Cockpit Dashboard + Footer
            cockpit_canvas[:480, :640] = d_frame
            cockpit_canvas[:480, 640:] = r_frame
            src_label = "CAR CABIN" if active_mode == "car" else "LIVE WEBCAM"
            _draw_footer(cockpit_canvas, risk_level, agent.mode.upper(), src_label, latest_ai_warning)

            cv2.imshow("Affective-Drive Master Control", cockpit_canvas)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                # Live toggle between realistic Car Cabin video & Webcam
                if active_mode == "car":
                    active_mode = "webcam"
                    perception.switch_driver_source(0)
                else:
                    active_mode = "car"
                    perception.switch_driver_source(car_video_path)
                last_risk_state = "LOW"
                last_driver_status = "NORMAL"

    finally:
        perception.stop()
        actuator.close()
        tts.stop()
        audio.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
