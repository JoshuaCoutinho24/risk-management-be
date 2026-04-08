"""
LLM-powered entity extraction.

Calls Claude claude-sonnet-4-6 with a structured JSON schema prompt and returns
a validated dict containing person, portfolio, risk_profile, life_events,
archetypes, and lived_through_regimes.

All extraction is synchronous (designed for use in a ThreadPoolExecutor).
"""

import json
import os
import re

import openai
from dotenv import load_dotenv

load_dotenv()

_client: openai.OpenAI | None = None


def _get_client() -> openai.OpenAI:
    global _client
    if _client is None:
        _client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


_SYSTEM_PROMPT = """\
You are a financial data extraction engine for an Indian financial advisory firm.
Given a free-text description of a financial client, extract structured data.
Return ONLY valid JSON — no markdown fences, no explanation, no trailing text.
All monetary values are in INR (Indian Rupees). Do NOT convert to USD or any other currency.
"""

_EXTRACTION_PROMPT = """\
Extract the client's financial profile and return exactly this JSON structure.
Fill in reasonable inferences where explicit data is absent.

Schema:
{{
  "person": {{
    "name": <str — client's first name or full name if mentioned, else "">,
    "age": <int>,
    "income": <int — annual gross income in INR (Indian Rupees)>,
    "net_worth": <int — total net worth in INR (Indian Rupees)>,
    "goal_amount": <int — target financial goal amount in INR, 0 if not mentioned>,
    "goal_years": <int — years to reach the goal, 0 if not mentioned>
  }},
  "portfolio": {{
    "eq_pct": <float — equity/stocks allocation, 0.0–1.0>,
    "bond_pct": <float — fixed-income/debt/FD allocation, 0.0–1.0>,
    "alt_pct": <float — alternatives: gold/real-estate/crypto allocation, 0.0–1.0>,
    "monthly_contribution": <int — monthly SIP or savings amount in INR, 0 if not mentioned>
  }},
  "risk_profile": {{
    "score": <int 1–100 — 1=very conservative, 100=very aggressive>,
    "loss_aversion": <float 0.0–5.0 — 5.0=extremely loss averse, 0.0=completely comfortable with losses>,
    "panic_threshold": <float — negative, e.g. -0.30 means panic/sell at -30% drawdown>
  }},
  "life_events": [
    {{"type": <str>, "date": <str, e.g. "2008-10">, "impact": <"low"|"medium"|"high">}}
  ],
  "archetypes": [<str — choose from: cautious_retiree, bold_accumulator, anxious_saver,
                  steady_builder, speculative_trader, income_seeker>],
  "lived_through_regimes": [<str — choose from: dot_com_2000, gfc_2008, covid_2020,
                              rate_shock_2022, ilfs_2018, taper_tantrum_2013,
                              china_selloff_2015, rbi_rate_hike_2022>]
}}

CRITICAL monetary rules — follow exactly:
- ALL monetary values are in INR. Do NOT convert to USD or any other currency.
- Unit parsing: k or K = ×1,000 | lakh or L = ×100,000 | crore or Cr = ×10,000,000
  Examples: "30k" = 30,000 | "5 lakh" = 500,000 | "1.5 Cr" = 15,000,000 | "300k" = 300,000
- If income is stated as monthly (e.g. "earns ₹50k a month"), multiply by 12 to get annual income.
  Example: "₹30k per month" → income = 360,000
- If net worth/savings is stated in lakh, convert correctly: "3 lakh" = 300,000 (NOT 3,000)

CRITICAL panic threshold rule:
- panic_threshold is the drop percentage at which the client would panic and sell.
- If the client says "I would hold even if it dropped by X%" → panic_threshold = -(X/100)
- If the client says "I can't handle more than X% loss" → panic_threshold = -(X/100)
- Use the EXACT number stated. Do NOT add any margin or buffer. "Hold at 30% drop" = -0.30 exactly.
- Typical range: -0.10 (panic at 10% drop) to -0.50 (panic at 50% drop)

Other rules:
- portfolio eq_pct + bond_pct + alt_pct MUST sum to exactly 1.0
- risk_profile.score: infer from stated risk tolerance and portfolio composition
- loss_aversion: 0-5 scale (0=loves risk, 3=average, 5=extremely averse to losses)
- archetypes: pick 1–3 that best fit; use the exact strings listed
- lived_through_regimes: include only regimes the client would realistically have
  experienced given their age and any explicit mentions
- If no life events are mentioned, return an empty array

Client description:
{text}
"""


def extract_entities(text: str) -> dict:
    """
    Extract structured financial entities from raw client text using Claude.

    Parameters
    ----------
    text : str
        Free-text client description submitted to /ingest.

    Returns
    -------
    dict
        Validated entity dict with keys: person, portfolio, risk_profile,
        life_events, archetypes, lived_through_regimes.

    Raises
    ------
    ValueError
        If the LLM response cannot be parsed as valid JSON after stripping
        markdown fences.
    """
    client = _get_client()

    prompt = _EXTRACTION_PROMPT.format(text=text.strip())

    message = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    raw: str = message.choices[0].message.content.strip()

    # Strip markdown code fences (```json … ``` or ``` … ```)
    raw = re.sub(r"^```(?:json)?\s*\n?", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\n?```\s*$", "", raw)
    raw = raw.strip()

    try:
        entities: dict = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"LLM returned non-JSON output: {exc}\n"
            f"Raw response (first 600 chars):\n{raw[:600]}"
        ) from exc

    # ── Normalise portfolio allocations to sum to 1.0 ────────────────────────
    port = entities.setdefault("portfolio", {})
    eq = float(port.get("eq_pct", 0.60))
    bd = float(port.get("bond_pct", 0.30))
    al = float(port.get("alt_pct", 0.10))
    total = eq + bd + al
    if total > 1e-6 and abs(total - 1.0) > 1e-4:
        port["eq_pct"] = round(eq / total, 6)
        port["bond_pct"] = round(bd / total, 6)
        port["alt_pct"] = round(al / total, 6)
    else:
        port["eq_pct"] = round(eq, 6)
        port["bond_pct"] = round(bd, 6)
        port["alt_pct"] = round(al, 6)

    # ── Clamp risk_profile values to legal ranges ─────────────────────────────
    rp = entities.setdefault("risk_profile", {})
    rp["score"] = max(1, min(100, int(rp.get("score", 50))))
    rp["loss_aversion"] = max(0.0, min(5.0, float(rp.get("loss_aversion", 2.5))))
    pt = float(rp.get("panic_threshold", -0.20))
    rp["panic_threshold"] = max(-0.90, min(-0.05, pt))  # clamp to [-90%, -5%]

    # ── Ensure required top-level keys exist ─────────────────────────────────
    entities.setdefault("person", {"name": "", "age": 0, "income": 0, "net_worth": 0,
                                   "goal_amount": 0, "goal_years": 0})
    person = entities["person"]
    person.setdefault("name", "")
    person.setdefault("goal_amount", 0)
    person.setdefault("goal_years", 0)

    # Ensure monthly_contribution on portfolio
    port.setdefault("monthly_contribution", 0)

    entities.setdefault("life_events", [])
    entities.setdefault("archetypes", [])
    entities.setdefault("lived_through_regimes", [])

    return entities
