import requests
import threading

class InterventionAgent:
    def __init__(self, model_name="llama3.2:3b", ollama_url="http://localhost:11434/api/generate"):
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.is_generating = False  # Lock flag to prevent duplicate overlapping requests

    def evaluate_and_respond_async(self, state_payload, callback=None):
        """Dispatches LLM evaluation to a background thread."""
        if self.is_generating:
            return  # Skip if an LLM generation is already in progress

        def worker():
            self.is_generating = True
            text = self.evaluate_and_respond(state_payload)
            self.is_generating = False
            
            if callback and text:
                callback(text)

        # Launch non-blocking background daemon thread
        threading.Thread(target=worker, daemon=True).start()

    def evaluate_and_respond(self, state_payload):
        risk = state_payload.get("system_risk_level", "LOW")
        driver_status = state_payload["driver_state"]["driver_status"]
        hazard = state_payload["road_state"]["high_risk_hazard"]

        if risk == "HIGH":
            prompt = (
                f"You are an AI co-pilot in a smart vehicle. The driver is '{driver_status}' "
                f"and road hazard detected is '{hazard}'. Provide a short, urgent 1-sentence warning."
            )
            return self._query_ollama(prompt) or "CRITICAL WARNING: Eyes on the road!"

        prompt = "System operating normally. Generate a calm 1-sentence status check for the driver."
        return self._query_ollama(prompt) or "All clear ahead. Safe driving!"

    def _query_ollama(self, prompt):
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            }
            response = requests.post(self.ollama_url, json=payload, timeout=3.0)
            if response.status_code == 200:
                return response.json().get("response", "").strip()
        except Exception as e:
            print(f"[OLLAMA ERROR] {e}")
        return None