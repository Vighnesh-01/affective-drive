"""
AI Co-Pilot Intervention Agent for Affective-Drive.

Generates contextual, emotion-aware safety responses for the driver.

Modes
-----
contextual (default)
    Rich template bank with 40+ responses, emotion-aware suffixes,
    and anti-repetition logic.  Zero network, < 1 ms latency.

gemini
    Google Gemini API for truly dynamic natural-language responses.
    Requires ``google-genai`` package + GEMINI_API_KEY env var.
    Falls back to *contextual* on error.

local_llm
    Ollama daemon on localhost:11434.  ~2 GB RAM.

rule_based
    Legacy 5-string dictionary.  Kept for backward compatibility.
"""

import os
import random
import threading
import requests


class InterventionAgent:

    def __init__(
        self,
        mode="contextual",
        model_name="gemini-2.5-flash",
        gemini_api_key=None,
        ollama_url="http://localhost:11434/api/generate",
    ):
        self.mode = mode
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.is_generating = False
        self._recent: list[str] = []
        self._max_recent = 8

        # Optional Gemini initialisation ("auto" uses Gemini if API key is present, else contextual)
        self._gemini_client = None
        if mode in ("gemini", "auto"):
            self._init_gemini(gemini_api_key)
            if self._gemini_client is not None:
                self.mode = "gemini"
            elif mode == "auto":
                self.mode = "contextual"

        # ── Response bank: (driver_status, severity) → list[str] ──────────
        self._bank = {
            # ─── NORMAL ───────────────────────────────────────────────────
            ("NORMAL", "LOW"): [
                "All systems nominal. Safe driving conditions ahead.",
                "Road conditions clear. You're driving attentively.",
                "Everything looks good. Maintaining observation.",
                "Clear road ahead. Keep up the safe driving.",
                "Systems check complete — all parameters normal.",
                "Driving conditions optimal. Stay focused and relaxed.",
                "No concerns detected. Enjoy the drive.",
            ],
            ("NORMAL", "HIGH"): [
                "⚠️ Caution — road hazard detected ahead.  Stay alert.",
                "⚠️ Vehicle in close proximity.  Maintain following distance.",
                "⚠️ Object detected ahead — increase your distance.",
            ],

            # ─── DISTRACTED ───────────────────────────────────────────────
            ("DISTRACTED", "LOW"): [
                "⚠️ Attention alert — please keep your eyes on the road.",
                "⚠️ Distraction detected.  Focus on the road ahead.",
                "⚠️ Driver alert — your attention has drifted.  Look forward.",
                "⚠️ Eyes on the road please.  Stay focused.",
                "⚠️ You seem to be looking away.  The road needs your attention.",
            ],
            ("DISTRACTED", "HIGH"): [
                "🚨 CRITICAL — Distracted driving with hazard ahead!  Look forward NOW!",
                "🚨 DANGER — Eyes on road immediately — close vehicle detected!",
                "🚨 BRAKE ALERT — Distraction + proximity hazard!  Focus now!",
            ],

            # ─── ANXIOUS ──────────────────────────────────────────────────
            ("ANXIOUS", "LOW"): [
                "I sense some tension.  Take a slow, deep breath.",
                "You seem stressed.  The road ahead is clear — ease up.",
                "Elevated stress detected.  Try relaxing your grip on the wheel.",
                "I'm noticing signs of anxiety.  Everything is under control.",
                "You look tense.  Breathe slowly — you're driving safely.",
            ],
            ("ANXIOUS", "HIGH"): [
                "🚨 High stress detected with road hazard!  Reduce speed carefully.",
                "🚨 Anxiety + close vehicle ahead.  Brake gently and stay calm.",
                "🚨 Stay calm — ease off the accelerator.  Vehicle ahead is close.",
            ],

            # ─── DROWSY ───────────────────────────────────────────────────
            ("DROWSY", "LOW"): [
                "🚨 Drowsiness detected!  Consider pulling over for a break.",
                "🚨 Your eyes are closing.  Please find a safe place to rest.",
                "🚨 Fatigue alert — pull over at the next rest stop.",
                "🚨 Driver fatigue detected!  Open a window or take a coffee break.",
                "🚨 You appear drowsy.  Drowsy driving is as dangerous as drunk driving.",
            ],
            ("DROWSY", "HIGH"): [
                "🚨 CRITICAL — Drowsy driving with hazard ahead!  Pull over NOW!",
                "🚨 DANGER — You're falling asleep!  Brake and find a safe stop!",
            ],

            # ─── STRESSED (mild) ─────────────────────────────────────────
            ("STRESSED", "LOW"): [
                "You seem a bit down.  Would some music help?  Stay focused.",
                "I notice you're not feeling great.  The road is clear ahead.",
                "Take it easy — you're doing fine.  Focus on the drive.",
            ],
            ("STRESSED", "HIGH"): [
                "⚠️ You seem stressed and there's a hazard ahead.  Drive carefully.",
                "⚠️ Emotional distress detected with proximity hazard.  Slow down.",
            ],

            # ─── NO FACE ─────────────────────────────────────────────────
            ("NO FACE", "LOW"): [
                "⚠️ Driver face not detected.  Please ensure the camera has a clear view.",
                "⚠️ Camera cannot see the driver.  Adjust your position or camera.",
            ],
            ("NO FACE", "HIGH"): [
                "🚨 Driver not visible and hazard detected!  Ensure camera view is clear.",
                "🚨 Cannot monitor driver — road hazard ahead.  Drive carefully!",
            ],
        }

        # Emotion-specific advice suffixes (appended for HIGH-risk events)
        self._emotion_tips = {
            "angry":    " Try to stay calm — aggressive driving increases accident risk.",
            "fear":     " You're safe.  Focus on steady breathing and driving.",
            "sad":      " Take care of yourself.  Consider stopping if emotions feel overwhelming.",
            "drowsy":   " Drowsy driving is as dangerous as drunk driving.  Rest is essential.",
            "surprise": "",
            "happy":    "",
            "neutral":  "",
            "none":     "",
        }

        # Cabin behavior specific alerts
        self._behavior_bank = {
            "SEATBELT_OFF": [
                "🚨 Safety check: Seatbelt unfastened! Please buckle your seatbelt immediately.",
                "⚠️ Seatbelt not detected — fasten your seatbelt before continuing.",
            ],
            "PHONE USE": [
                "🚨 Phone distraction detected! Put the phone away and keep both hands on the wheel.",
                "🚨 Unsafe phone usage — keep your eyes on the road, not your screen.",
            ],
            "EATING/DRINKING": [
                "⚠️ Eating or drinking detected while driving. Keep both hands free for steering control.",
                "⚠️ Food or drink distraction — pull over safely before eating or drinking.",
            ],
            "YAWNING": [
                "🚨 Frequent yawning detected! Fatigue warning — please consider taking a rest break.",
                "⚠️ Signs of driver fatigue (yawning). Stay alert or stop for fresh air.",
            ],
            "TALKING": [
                "💬 Active conversation detected — keep your eyes scanning the road ahead.",
                "💬 Stay focused while speaking — maintain safe following distance.",
            ],
        }

    # ── Gemini setup ──────────────────────────────────────────────────────
    def _init_gemini(self, api_key=None):
        try:
            key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if key:
                from google import genai
                self._gemini_client = genai.Client(api_key=key)
                print("[AI CO-PILOT] Gemini API initialised.")
            else:
                print("[AI CO-PILOT] No API key found — falling back to contextual mode.")
                self.mode = "contextual"
        except ImportError:
            print("[AI CO-PILOT] google-genai not installed — using contextual mode.")
            self.mode = "contextual"

    # ── Contextual response engine ────────────────────────────────────────
    def _contextual_response(self, payload):
        ds = payload.get("driver_state", {})
        driver_status = ds.get("driver_status", "NORMAL")
        emotion       = ds.get("emotion", "neutral")
        seatbelt      = ds.get("seatbelt", "N/A")
        activity      = ds.get("activity", "ATTENTIVE")
        hazard        = payload.get("road_state", {}).get("high_risk_hazard", "Clear")
        risk          = payload.get("system_risk_level", "LOW")

        severity = "HIGH" if (risk == "HIGH" or "Close" in str(hazard)) else "LOW"

        # Prioritize specific cabin behavior warnings when detected (unless NO FACE or CRITICAL hazard)
        pool = None
        if driver_status != "NO FACE" and not (severity == "HIGH" and "Close" in str(hazard)):
            if activity in ("PHONE USE", "EATING/DRINKING", "YAWNING"):
                pool = self._behavior_bank.get(activity)
            elif seatbelt == "OFF":
                pool = self._behavior_bank.get("SEATBELT_OFF")
            elif activity == "TALKING" and driver_status == "NORMAL":
                pool = self._behavior_bank.get("TALKING")

        if not pool:
            pool = (self._bank.get((driver_status, severity))
                    or self._bank.get((driver_status, "LOW"))
                    or self._bank.get(("NORMAL", "LOW")))

        available = [r for r in pool if r not in self._recent] or pool
        response  = random.choice(available)

        if severity == "HIGH" and activity not in ("PHONE USE", "EATING/DRINKING"):
            tip = self._emotion_tips.get(emotion, "")
            if tip and tip not in response:
                response += tip

        self._recent.append(response)
        if len(self._recent) > self._max_recent:
            self._recent.pop(0)
        return response

    # ── Gemini response ───────────────────────────────────────────────────
    def _gemini_response(self, payload):
        if not self._gemini_client:
            return self._contextual_response(payload)

        ds = payload.get("driver_state", {})
        prompt = (
            f"You are an AI co-pilot safety system in a moving vehicle. "
            f"Driver status: '{ds.get('driver_status')}', facial emotion: "
            f"'{ds.get('emotion')}', cabin activity: '{ds.get('activity', 'ATTENTIVE')}', "
            f"seatbelt: '{ds.get('seatbelt', 'N/A')}', road hazard: "
            f"'{payload.get('road_state',{}).get('high_risk_hazard')}', "
            f"risk: '{payload.get('system_risk_level')}'. "
            f"Give ONE concise, empathetic safety message (1-2 sentences). "
            f"Address the specific driver emotion/behavior first if unsafe. No markdown."
        )
        try:
            resp = self._gemini_client.models.generate_content(
                model=self.model_name, contents=prompt
            )
            text = resp.text.strip()
            if text:
                return text
        except Exception as e:
            print(f"[AI CO-PILOT] Gemini error: {e}")
        return self._contextual_response(payload)

    # ── Legacy rule-based ─────────────────────────────────────────────────
    def _rule_based_response(self, payload):
        ds = payload.get("driver_state", {}).get("driver_status", "NORMAL")
        hz = payload.get("road_state", {}).get("high_risk_hazard", "Clear")
        hk = "Close Vehicle" if ("Close" in hz or "Vehicle" in hz or "Pedestrian" in hz) else "Clear"
        table = {
            ("DISTRACTED", "Clear"):         "⚠️ Eyes on the road — distraction detected!",
            ("ANXIOUS",    "Clear"):         "Stay calm — take a steady breath and focus.",
            ("DISTRACTED", "Close Vehicle"): "🚨 BRAKE! Distracted driver with close hazard!",
            ("ANXIOUS",    "Close Vehicle"): "🚨 DANGER! Vehicle too close — reduce speed!",
            ("NORMAL",     "Close Vehicle"): "⚠️ Hazard ahead — increase following distance.",
        }
        return table.get((ds, hk), "⚠️ Safety alert: stay alert on the road.")

    # ── Ollama helper ─────────────────────────────────────────────────────
    def _query_ollama(self, prompt):
        try:
            r = requests.post(self.ollama_url,
                              json={"model": "llama3.2:3b", "prompt": prompt, "stream": False},
                              timeout=3.0)
            if r.status_code == 200:
                return r.json().get("response", "").strip()
        except Exception as e:
            print(f"[AI CO-PILOT] Ollama error: {e}")
        return None

    # ── Unified public API ────────────────────────────────────────────────
    def evaluate_and_respond(self, payload):
        if self.mode == "contextual":
            return self._contextual_response(payload)
        if self.mode == "gemini":
            return self._gemini_response(payload)
        if self.mode == "local_llm":
            ds = payload.get("driver_state", {})
            prompt = (
                f"You are an AI co-pilot. Driver: {ds.get('driver_status')}, "
                f"emotion: {ds.get('emotion')}, hazard: "
                f"{payload.get('road_state',{}).get('high_risk_hazard')}, "
                f"risk: {payload.get('system_risk_level')}. "
                f"Give a 1-sentence safety warning."
            )
            return self._query_ollama(prompt) or self._contextual_response(payload)
        return self._rule_based_response(payload)

    def evaluate_and_respond_async(self, payload, callback=None):
        """Non-blocking variant.  Instant modes run in-line; LLM modes use a thread."""
        if self.mode in ("contextual", "rule_based"):
            resp = self.evaluate_and_respond(payload)
            if callback and resp:
                callback(resp)
            return
        if self.is_generating:
            return

        def _worker():
            self.is_generating = True
            text = self.evaluate_and_respond(payload)
            self.is_generating = False
            if callback and text:
                callback(text)

        threading.Thread(target=_worker, daemon=True).start()
