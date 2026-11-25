import re
import threading
import queue
import pyttsx3


class TTSEngine:
    def __init__(self, voice_gender="female", rate=170, pitch=0):
        self.speech_queue = queue.Queue()
        self.is_speaking = False
        self.rate = rate
        self.pitch = pitch
        self.voice_gender = voice_gender.lower()

        self.worker_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.worker_thread.start()

    @staticmethod
    def _clean_text(text: str) -> str:
        """Remove emojis and XML-unsafe chars so Windows SAPI5 speaks smoothly."""
        cleaned = re.sub(r"[^\x20-\x7E]+", " ", text)
        cleaned = cleaned.replace("&", " and ").replace("<", " ").replace(">", " ")
        return re.sub(r"\s+", " ", cleaned).strip()

    def _speech_worker(self):
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)

            voices = engine.getProperty("voices")
            if len(voices) > 1 and self.voice_gender == "female":
                engine.setProperty("voice", voices[1].id)
            elif len(voices) > 0:
                engine.setProperty("voice", voices[0].id)
        except Exception as e:
            print(f"[TTS INIT ERROR]: {e}")
            engine = None

        while True:
            text = self.speech_queue.get()
            if text is None:
                self.speech_queue.task_done()
                break

            if engine is None:
                self.speech_queue.task_done()
                continue

            self.is_speaking = True
            try:
                spoken = self._clean_text(text)
                if spoken:
                    engine.say(spoken)
                    engine.runAndWait()
            except Exception as e:
                print(f"[TTS ERROR]: {e}")
            finally:
                self.is_speaking = False
                self.speech_queue.task_done()

    def speak_async(self, text):
        """Dispatches speech asynchronously without flooding the queue."""
        if not text:
            return
        if not self.is_speaking and self.speech_queue.empty():
            self.speech_queue.put(text)

    def stop(self):
        self.speech_queue.put(None)
