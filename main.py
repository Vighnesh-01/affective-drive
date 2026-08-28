import cv2
import time
from src.perception.fusion_pipeline import AffectivePerceptionEngine
from src.agent.llm_intervention import InterventionAgent
from src.hardware.serial_controller import HardwareController
from src.agent.tts_engine import TTSEngine  # <--- Import TTS Engine

def main():
    print("=== Starting Affective-Drive AI Cockpit System ===")
    
    perception = AffectivePerceptionEngine(driver_src=0, road_src="data/sample_road.mp4")
    agent = InterventionAgent(model_name="llama3.2:3b")
    actuator = HardwareController()
    tts = TTSEngine(voice_gender="female", rate=165, pitch=2)  # <--- Initialize TTS

    last_intervention_time = 0
    cooldown_seconds = 5.0
    last_risk_state = "LOW"
    
    latest_ai_warning = "System Normal"

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

            # 2. Async LLM + TTS Trigger on High Risk
            if risk_level == "HIGH" and (last_risk_state == "LOW" or (current_time - last_intervention_time > cooldown_seconds)):
                agent.evaluate_and_respond_async(payload, callback=on_ai_response)
                last_intervention_time = current_time

            last_risk_state = risk_level

            # 3. Render Dashboard Display
            r_resized = cv2.resize(r_frame, (d_frame.shape[1], d_frame.shape[0]))
            cockpit = cv2.hconcat([d_frame, r_resized])

            banner_color = (0, 0, 255) if risk_level == "HIGH" else (0, 255, 0)
            cv2.rectangle(cockpit, (0, cockpit.shape[0] - 40), (cockpit.shape[1], cockpit.shape[0]), (0, 0, 0), -1)
            cv2.putText(cockpit, f"AI Co-Pilot: {latest_ai_warning}", (10, cockpit.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, banner_color, 2)

            cv2.imshow("Affective-Drive Master Control", cockpit)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        perception.stop()
        actuator.close()
        tts.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()