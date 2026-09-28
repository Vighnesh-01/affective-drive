"""
generate_alerts.py — Pre-record all Affect-Drive safety alert WAV files.

Run this ONCE before launching the cockpit system:
    .\.venv\Scripts\python.exe generate_alerts.py

Uses pyttsx3 (already installed) to synthesize and save each alert as
a standalone WAV file in the alerts/ directory. After generation, the
AudioAlertEngine plays these files instantly via pygame.mixer (<5ms).
"""
import os
import pyttsx3

ALERTS_DIR = os.path.join(os.path.dirname(__file__), "alerts")
os.makedirs(ALERTS_DIR, exist_ok=True)

ALERT_SCRIPTS = {
    "distracted":       "Warning! Driver distraction detected. Eyes on the road.",
    "anxious":          "Stay calm. Take a breath and focus on driving.",
    "distracted_hazard":"Critical alert! Distracted driver with close hazard ahead. Brake now!",
    "anxious_hazard":   "Danger! Vehicle too close and driver is stressed. Reduce speed immediately!",
    "hazard_close":     "Caution. Road hazard detected ahead. Increase following distance.",
}

def generate():
    engine = pyttsx3.init()
    engine.setProperty("rate", 155)

    voices = engine.getProperty("voices")
    if len(voices) > 1:
        engine.setProperty("voice", voices[1].id)  # Female voice

    print(f"Saving alert WAV files to: {ALERTS_DIR}\n")
    for name, text in ALERT_SCRIPTS.items():
        out_path = os.path.join(ALERTS_DIR, f"{name}.wav")
        engine.save_to_file(text, out_path)
        print(f"  [OK] {name}.wav  ->  \"{text}\"")

    engine.runAndWait()
    print(f"\nAll {len(ALERT_SCRIPTS)} alert files generated successfully!")
    print("You can now run main.py or main_dataset_mode.py with instant audio alerts.")

if __name__ == "__main__":
    generate()
