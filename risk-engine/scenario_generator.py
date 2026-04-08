"""
AI-powered custom market scenario generator.

Takes a natural-language description from an advisor and uses GPT-4o to
generate realistic monthly return arrays for equity, bond, and alt assets.
The output is compatible with the SCENARIOS dict format used by run_historical().
"""
import json, os, re
import openai
from dotenv import load_dotenv

load_dotenv()

_client = None

def _get_client():
    global _client
    if _client is None:
        _client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client

_SYSTEM_PROMPT = """You are a quantitative financial scenario modelling engine.
Given an advisor's plain-language description of a market scenario, generate
realistic monthly return arrays for equity, bonds, and alternatives.

Rules:
- All three arrays must be the same length (between 6 and 24 months)
- Monthly returns are decimal fractions: -0.05 means -5% that month
- Be realistic. Reference levels: GFC worst equity month ~-17%, COVID worst ~-13%
- Bonds typically move opposite to equities during equity crashes
- Alts (REITs, commodities, crypto) correlate loosely with equity
- Include a realistic recovery tail after the crash if the scenario implies one
- Return ONLY valid JSON, no markdown, no explanation

Output schema:
{
  "name": "<short descriptive scenario name>",
  "equity": [<list of monthly decimal returns>],
  "bond": [<list of monthly decimal returns>],
  "alt": [<list of monthly decimal returns>],
  "description": "<1-2 sentence description of the scenario>"
}
"""

def generate_scenario(prompt: str) -> dict:
    """
    Generate a custom market scenario from a natural-language advisor prompt.
    Returns a dict with keys: name, equity, bond, alt, description.
    Compatible with scenarios.SCENARIOS format.
    """
    c = _get_client()
    response = c.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Generate a scenario for: {prompt.strip()}"}
        ],
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```(?:json)?\s*\n?", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\n?```\s*$", "", raw)
    parsed = json.loads(raw)

    # Validate and normalise
    for key in ("equity", "bond", "alt"):
        if key not in parsed or not isinstance(parsed[key], list):
            raise ValueError(f"Generated scenario missing '{key}' array")

    n = len(parsed["equity"])
    if not (6 <= n <= 24):
        raise ValueError(f"Scenario length {n} out of range [6, 24]")

    # Clamp extreme values to plausible range [-25%, +15%] per month
    for key in ("equity", "bond", "alt"):
        parsed[key] = [max(-0.25, min(0.15, float(v))) for v in parsed[key]]
        # Pad or trim to match equity length
        parsed[key] = parsed[key][:n]
        while len(parsed[key]) < n:
            parsed[key].append(0.0)

    parsed.setdefault("name", "Custom Scenario")
    parsed.setdefault("description", prompt[:200])
    return parsed
