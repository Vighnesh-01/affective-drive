import cv2
import time
import numpy as np
from src.perception.fusion_pipeline import AffectivePerceptionEngine
from src.agent.llm_intervention import InterventionAgent
from src.hardware.serial_controller import HardwareController
from src.agent.tts_engine import TTSEngine
from telegram_notifier import TelegramNotifier
from src.agent.audio_alerts import AudioAlertEngine


def main():
    print("=== Starting Affective-Drive AI Cockpit System ===")

    # 1. Initialize Pipeline & Modules (640x480 native stream resolution)
    perception = AffectivePerceptionEngine(
        driver_src=0,
        road_src="data/sample_road.mp4",
        skip_road_frames=2,
        target_resolution=(640, 480)
    )
    # Default to rule_based mode to save ~2.2 GB RAM and eliminate LLM inference latency
    agent = InterventionAgent(mode="rule_based")
    actuator = HardwareController()
    tts = TTSEngine(voice_gender="female", rate=165, pitch=2)

    # 2. Safety Interventions & Telemetry Bot
    notifier = TelegramNotifier(
        bot_token="8602864132:AAEwK3rR4PMDcmdK67tnI7bngS-e0QKJigs",
        chat_id="6167292056",
        cooldown_seconds=6.0,
        mock_mode=False
    )
    audio = AudioAlertEngine(volume=0.9)

    last_intervention_time = 0
    cooldown_seconds = 5.0
    last_risk_state = "LOW"
    latest_ai_warning = "System Normal"

    # Pre-allocate dashboard canvas once (640*2 x 480) to eliminate per-frame allocations
    cockpit_canvas = np.zeros((480, 1280, 3), dtype=np.uint8)

    def on_ai_response(response_text):
        nonlocal latest_ai_warning
        latest_ai_warning = response_text
        print(f"\n[AI CO-PILOT INTERVENTION]: {response_text}\n")
        # Speak response aloud asynchronously without dropping video frames
        tts.speak_async(response_text)

    try:
        while True:
            d_frame, r_frame, payload = perception.step()

            if d_frame is None or r_frame is None or payload is None:
                time.sleep(0.01)
                continue

            current_time = time.time()
            risk_level = payload.get("system_risk_level", "LOW")

            # 1. Hardware Actuation
            actuator.send_actuation_signal(risk_level)

            # 2. Async Intervention Trigger on High Risk
            if risk_level == "HIGH" and (last_risk_state == "LOW" or (current_time - last_intervention_time > cooldown_seconds)):
                agent.evaluate_and_respond_async(payload, callback=on_ai_response)
                
                # Instant Audio Alert (<5ms)
                audio.play_for_payload(payload)

                # Asynchronous Telegram Alert & Snapshot Dispatch
                notifier.send_alert(payload, snapshot_frame=d_frame)

                last_intervention_time = current_time

            last_risk_state = risk_level

            # 3. Optimized In-Place Dashboard Render (Zero cv2.resize / cv2.hconcat overhead)
            cockpit_canvas[:, :640] = d_frame
            cockpit_canvas[:, 640:] = r_frame

            banner_color = (0, 0, 255) if risk_level == "HIGH" else (0, 255, 0)
            cv2.rectangle(
                cockpit_canvas,
                (0, cockpit_canvas.shape[0] - 40),
                (cockpit_canvas.shape[1], cockpit_canvas.shape[0]),
                (0, 0, 0),
                -1
            )
            cv2.putText(
                cockpit_canvas,
                f"AI Co-Pilot: {latest_ai_warning}",
                (10, cockpit_canvas.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                banner_color,
                2
            )

            cv2.imshow("Affective-Drive Master Control", cockpit_canvas)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        perception.stop()
        actuator.close()
        tts.stop()
        audio.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()