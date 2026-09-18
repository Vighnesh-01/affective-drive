import requests
import threading

class InterventionAgent:
    def __init__(
        self,
        mode="rule_based",
        model_name="llama3.2:3b",
        ollama_url="http://localhost:11434/api/generate",
        cloud_url=None
    ):
        """
        mode: "rule_based"  → Zero LLM RAM, <1ms latency  (embedded default)
              "local_llm"   → Ollama daemon, ~2.2GB RAM    (high-spec only)
              "cloud_llm"   → REST API to cloud endpoint   (requires network)
        """
        self.mode = mode
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.cloud_url = cloud_url
        self.is_generating = False

        # Pre-compiled deterministic response table (zero RAM overhead)
        self._rule_table = {
            ("DISTRACTED", "Clear"):         "⚠️ Eyes on the road — distraction detected!",
            ("ANXIOUS",    "Clear"):         "Stay calm — take a steady breath and focus.",
            ("DISTRACTED", "Close Vehicle"): "🚨 BRAKE! Distracted driver with close hazard ahead!",
            ("ANXIOUS",    "Close Vehicle"): "🚨 DANGER! Vehicle too close — reduce speed immediately!",
            ("NORMAL",     "Close Vehicle"): "⚠️ Hazard ahead — increase following distance.",
        }

    def _rule_based_response(self, state_payload):
        """Deterministic <1ms response with zero memory overhead."""
        driver_status = state_payload.get("driver_state", {}).get("driver_status", "NORMAL")
        hazard = state_payload.get("road_state", {}).get("high_risk_hazard", "Clear")
        hazard_key = "Close Vehicle" if ("Close" in hazard or "Vehicle" in hazard or "Pedestrian" in hazard) else "Clear"
        key = (driver_status, hazard_key)
        return self._rule_table.get(key, "⚠️ Safety alert: stay alert on the road.")

    def evaluate_and_respond_async(self, state_payload, callback=None):
        """Dispatches response evaluation synchronously (<1ms) or asynchronously (LLM)."""
        if self.mode == "rule_based":
            response = self._rule_based_response(state_payload)
            if callback and response:
                callback(response)
            return

        if self.is_generating:
            return  # Skip duplicate requests while LLM is generating

        def _llm_worker():
            self.is_generating = True
            text = self.evaluate_and_respond(state_payload)
            self.is_generating = False
            if callback and text:
                callback(text)

        threading.Thread(target=_llm_worker, daemon=True).start()

    def evaluate_and_respond(self, state_payload):
        """Evaluates driver state via LLM prompt generation."""
        if self.mode == "rule_based":
            return self._rule_based_response(state_payload)

        risk = state_payload.get("system_risk_level", "LOW")
        driver_status = state_payload.get("driver_state", {}).get("driver_status", "NORMAL")
        hazard = state_payload.get("road_state", {}).get("high_risk_hazard", "Clear")

        if risk == "HIGH":
            prompt = (
                f"You are an AI co-pilot in a smart vehicle. The driver is '{driver_status}' "
                f"and road hazard detected is '{hazard}'. Provide a short, urgent 1-sentence warning."
            )
            return self._query_ollama(prompt) or self._rule_based_response(state_payload)

        prompt = "System operating normally. Generate a calm 1-sentence status check for the driver."
        return self._query_ollama(prompt) or "All clear ahead. Safe driving!"

    def _query_ollama(self, prompt):
        """Queries local Ollama endpoint or cloud URL."""
        endpoint = self.cloud_url if (self.mode == "cloud_llm" and self.cloud_url) else self.ollama_url
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            }
            response = requests.post(endpoint, json=payload, timeout=3.0)
            if response.status_code == 200:
                return response.json().get("response", "").strip()
        except Exception as e:
            print(f"[LLM AGENT ERROR] {e}")
        return None