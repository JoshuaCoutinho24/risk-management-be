"""
AI-powered client intake questionnaire.

Conducts a friendly, jargon-free back-and-forth interview via OpenAI to
collect enough data to build a financial risk profile. Questions are written
for a regular person — not a finance professional. The AI synthesises answers
into a structured summary that gets passed to the financial advisor.
"""

import json
import os
import re

import openai
from dotenv import load_dotenv
from settings_store import get_settings

load_dotenv()

_client: openai.OpenAI | None = None


def _get_client() -> openai.OpenAI:
    global _client
    if _client is None:
        _client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


# Fallback only — settings_store.DEFAULT_INTERVIEW_PROMPT is the single source of truth.
# This is used only if the store hasn't initialised yet (e.g. import-time errors).
from settings_store import DEFAULT_INTERVIEW_PROMPT as _BASE_SYSTEM_PROMPT


def run_chat_turn(messages: list[dict], client_name: str = "") -> dict:
    """
    Run one turn of the intake interview.

    Parameters
    ----------
    messages : list[dict]
        Conversation history as [{role: "user"|"assistant", content: str}, ...]
        Pass an empty list to get the opening question.
    client_name : str, optional
        If provided, the AI will address the client by this name.

    Returns
    -------
    dict with keys:
        status   : "gathering" | "complete"
        message  : str — AI's next message (always present)
        summary  : str | None — advisor-facing summary (only when complete)
    """
    ai = _get_client()

    # Always read the prompt fresh from the store so advisor edits take effect
    # on the next message without a server restart
    system_prompt = get_settings().get("interview_prompt", _BASE_SYSTEM_PROMPT)
    if client_name:
        system_prompt = f"The client's name is {client_name}.\n\n" + system_prompt

    chat_messages: list[dict] = [{"role": "system", "content": system_prompt}]

    # Opening turn: no history yet — ask the first question
    if not messages:
        chat_messages.append({
            "role": "user",
            "content": "Please start the questionnaire with a warm welcome and the first question.",
        })
    else:
        chat_messages.extend(messages)

    response = ai.chat.completions.create(
        model="gpt-5-mini",
        messages=chat_messages,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content.strip()

    # Strip accidental markdown fences
    raw = re.sub(r"^```(?:json)?\s*\n?", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\n?```\s*$", "", raw)
    raw = raw.strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {
            "status": "gathering",
            "message": raw[:500] if raw else "Hi! Let's start — could you tell me a little about yourself and what brings you here today?",
        }

    return {
        "status":  str(parsed.get("status", "gathering")),
        "message": str(parsed.get("message", "")),
        "summary": parsed.get("summary"),
    }
