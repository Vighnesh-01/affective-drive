import pyttsx3
import threading
import queue

class TTSEngine:
    def __init__(self, voice_gender="female", rate=170, pitch=0):
        self.speech_queue = queue.Queue()
        self.is_speaking = False
        self.rate = rate
        self.pitch = pitch
        self.voice_gender = voice_gender.lower()
        
        self.worker_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.worker_thread.start()

    def _speech_worker(self):
        engine = pyttsx3.init()
        engine.setProperty('rate', self.rate)
        
        # Select Gender
        voices = engine.getProperty('voices')
        if len(voices) > 1 and self.voice_gender == "female":
            engine.setProperty('voice', voices[1].id)
        elif len(voices) > 0:
            engine.setProperty('voice', voices[0].id)

        while True:
            text = self.speech_queue.get()
            if text is None:
                break
            
            self.is_speaking = True
            try:
                # Apply pitch XML wrapper if pitch modification is requested
                formatted_text = f'<pitch middle="{self.pitch}">{text}</pitch>' if self.pitch != 0 else text
                engine.say(formatted_text)
                engine.runAndWait()
            except Exception as e:
                print(f"[TTS ERROR]: {e}")
            finally:
                self.is_speaking = False
                self.speech_queue.task_done()

    def speak_async(self, text):
        """Dispatches speech asynchronously without flooding the queue."""
        if not self.is_speaking and self.speech_queue.empty():
            self.speech_queue.put(text)

    def stop(self):
        self.speech_queue.put(None)