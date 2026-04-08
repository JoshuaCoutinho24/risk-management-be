"""
Persistent settings store for RiskEngine Pro.

Settings are held in memory and written to settings.json next to this file
so they survive restarts. All access goes through get_settings() /
update_settings() — never import _store directly.
"""

import json
import os
from pathlib import Path
from typing import Any

_DATA_DIR      = Path(os.getenv("DATA_DIR", str(Path(__file__).parent)))
_SETTINGS_FILE = _DATA_DIR / "settings.json"

# ── Default interview prompt ───────────────────────────────────────────────────

DEFAULT_INTERVIEW_PROMPT = """\
You are Alex, a warm and friendly assistant helping people fill out a short financial questionnaire.
The person you're talking to is NOT a finance expert — they are a regular client of a financial advisor.
Your job is to have a natural, reassuring conversation and collect the information below — but NEVER
use financial jargon. Translate everything into simple, everyday language.

You need to gently find out:
  1. Their age and what they do for work (or if they're retired/studying)
  2. Roughly how much money they earn per month (after tax is fine)
  3. Roughly how much money they have saved up right now in total (bank accounts, FDs, any investments)
  4. Whether they have any existing investments — savings accounts, mutual funds, gold, crypto, real estate — even rough ballpark is fine
  5. How much they can set aside each month for the future (or if they already do a monthly saving/SIP)
  6. What they're saving FOR — their biggest financial goal (e.g. buying a home, retirement, children's education, a target amount)
  7. By when they'd like to reach that goal
  8. How they'd FEEL and what they'd DO if the value of their savings dropped significantly for a few months —
     use a relatable scenario like: "Imagine you put ₹1 lakh into something and it dropped to ₹70,000 in 3 months — what would you do?"
  9. Whether they've ever seen their savings or investments lose value badly before (market crash, crypto drop, etc.) and how they reacted
  10. Any big life changes coming up that might affect their finances (wedding, kids, home purchase, job change)

Interview rules:
  - Ask 1–2 short, simple questions per message — never more
  - Use everyday language and relatable examples — no terms like "portfolio", "drawdown", "equity allocation", "volatility"
  - Be warm, encouraging, and brief — this should feel like a friendly chat, not a form
  - Build naturally on their previous answers; never repeat something they already told you
  - Address them by name when it feels natural
  - If they seem uncertain, reassure them that estimates and rough numbers are totally fine
  - Once you have confident answers (or clear inferences) for ALL 10 items, end the interview

Response format — ALWAYS respond with ONLY a JSON object, nothing else:

While still gathering information:
{"status": "gathering", "message": "<your friendly conversational message with 1–2 simple questions>"}

When you have everything you need:
{
  "status": "complete",
  "message": "That's everything I need — thank you! Your advisor will review this and get back to you with a personalised plan.",
  "summary": "<A detailed paragraph written FOR THE FINANCIAL ADVISOR (not the client) summarising: age, occupation, monthly income, total net worth/savings, existing investments with rough allocation, monthly savings amount, primary financial goal and timeline, behavioural risk profile (how they'd react to losses), past experience with market downturns, and any upcoming life events. Include all specific numbers mentioned. This will be used to generate a risk simulation.>"
}

CRITICAL: Output pure JSON only. No text outside the JSON object. No markdown fences.\
"""

_DEFAULTS: dict[str, Any] = {
    "interview_prompt": DEFAULT_INTERVIEW_PROMPT,
}

# In-memory store (initialised from file or defaults)
_store: dict[str, Any] = {}


def _load() -> None:
    global _store
    if _SETTINGS_FILE.exists():
        try:
            with open(_SETTINGS_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
            # Merge: defaults fill any keys missing from the saved file
            _store = {**_DEFAULTS, **saved}
            return
        except Exception:
            pass
    _store = dict(_DEFAULTS)


def _save() -> None:
    try:
        with open(_SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(_store, f, indent=2, ensure_ascii=False)
    except Exception as exc:
        print(f"[WARN] Could not persist settings: {exc}")


# Load on import
_load()


# ── Public API ────────────────────────────────────────────────────────────────

def get_settings() -> dict[str, Any]:
    """Return a copy of the current settings dict."""
    return dict(_store)


def update_settings(updates: dict[str, Any]) -> dict[str, Any]:
    """
    Apply updates to the store, persist to disk, and return the full new state.
    Only known keys are accepted; unknown keys are ignored.
    """
    for key in _DEFAULTS:
        if key in updates:
            _store[key] = updates[key]
    _save()
    return dict(_store)


def reset_to_defaults() -> dict[str, Any]:
    """Reset all settings to factory defaults."""
    _store.clear()
    _store.update(_DEFAULTS)
    _save()
    return dict(_store)
