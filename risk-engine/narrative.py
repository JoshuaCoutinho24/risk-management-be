"""
Narrative + allocation recommendation generation.

Calls OpenAI with the full person subgraph and simulation results to produce:
  1. A plain-language advisor summary (three paragraphs)
  2. A recommended portfolio split across Equities, Debt, Real Estate,
     and Commodities — each with a simple explanation of WHY that number
     was chosen for this specific client.

All calls are synchronous (designed for use in a ThreadPoolExecutor).
"""

import json
import os
import re

import openai
from dotenv import load_dotenv
from fund_metrics import match_funds, load_cache

load_dotenv()

_client: openai.OpenAI | None = None


def _get_client() -> openai.OpenAI:
    global _client
    if _client is None:
        _client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client



_SYSTEM_PROMPT = """\
You are a senior financial risk analyst writing a report for a financial advisor about their client.

Your job is to produce TWO things:
1. A plain-language 3-paragraph summary of the client's risk situation
2. A recommended portfolio allocation split across 4 asset classes, with a simple explanation for each

Audience: the financial advisor reading this. They need clear, actionable numbers and explanations
they can use directly in a client meeting.

Language rules:
- Write like you are explaining to a smart person who is not a finance expert
- No jargon — replace technical terms with everyday explanations
- Be specific — use the actual numbers from the data
- Be warm but direct

You MUST respond with ONLY a valid JSON object in this exact format:
{
  "narrative": "<3 short paragraphs separated by \\n\\n>",
  "allocation": {
    "equities":    { "pct": <int>, "reason": "<2-3 plain sentences>" },
    "debt":        { "pct": <int>, "reason": "<2-3 plain sentences>" },
    "real_estate": { "pct": <int>, "reason": "<2-3 plain sentences>" },
    "commodities": { "pct": <int>, "reason": "<2-3 plain sentences>" },
    "overall_reasoning": "<1-2 sentences>"
  }
}

Rules for allocation:
- The 4 percentages MUST add up to exactly 100
- Higher risk score + longer timeline = more equities
- Lower risk score or near a goal = more debt
- Panic-prone clients get more commodities (gold) as a buffer

CRITICAL: Output only the JSON object. No text before or after it. No markdown fences.
"""


def generate_narrative(subgraph: dict, all_results: dict) -> dict:
    """
    Generate a plain-language advisor report and allocation recommendation.

    Parameters
    ----------
    subgraph : dict
        Full person subgraph from graph.read_subgraph().
    all_results : dict
        Combined output from historical, monte_carlo, and behavioural sims,
        plus a 'scenario_key' entry.

    Returns
    -------
    dict with keys:
        narrative              : str   — three-paragraph advisor summary
        allocation             : dict  — equities/debt/real_estate/commodities
                                         each with pct (int) and reason (str),
                                         plus overall_reasoning (str)
        fund_recommendations   : list  — top 5 funds matched by live metric engine
    """
    client = _get_client()

    person       = subgraph.get("person", {})
    portfolio    = subgraph.get("portfolio", {})
    risk_profile = subgraph.get("risk_profile", {})
    archetypes   = subgraph.get("archetypes", [])
    life_events  = subgraph.get("life_events", [])
    lived_through = subgraph.get("lived_through_regimes", [])

    hist = all_results.get("historical", {})
    mc   = all_results.get("monte_carlo", {})
    beh  = all_results.get("behavioural", {})
    scenario_key = all_results.get("scenario_key", "unknown")

    le_summary = "; ".join(
        f"{e.get('type', '?')} ({e.get('date', '?')}, {e.get('impact', '?')} impact)"
        for e in life_events[:5]
    ) or "none on record"

    age       = person.get("age") or "unknown"
    income    = person.get("income") or 0
    net_worth = person.get("net_worth") or 0

    eq_pct   = portfolio.get("eq_pct", 0) * 100
    bond_pct = portfolio.get("bond_pct", 0) * 100
    alt_pct  = portfolio.get("alt_pct", 0) * 100

    risk_score          = risk_profile.get("score", 50)
    loss_aversion       = risk_profile.get("loss_aversion", 0.5)
    panic_threshold_pct = risk_profile.get("panic_threshold", -0.20) * 100

    max_dd_pct        = hist.get("max_drawdown", 0) * 100
    recovery_months   = hist.get("recovery_months", 0)
    stress_index      = hist.get("stress_index", 0)
    panic_triggered   = hist.get("panic_triggered", False)
    hist_beh_delta_pct = hist.get("behavioral_delta", 0) * 100

    mc_median_pct     = mc.get("mc_median_10yr", 1.0) * 100 - 100
    mc_p5_pct         = mc.get("mc_p5_10yr", 1.0) * 100 - 100
    mc_p95_pct        = mc.get("mc_p95_10yr", 1.0) * 100 - 100
    panic_fraction_pct = mc.get("panic_fraction", 0) * 100
    goal_probability  = mc.get("goal_probability", 0) * 100

    beh_delta_pct  = beh.get("behavioral_delta", 0) * 100
    worst_scenario = beh.get("worst_scenario_key", scenario_key)
    beh_panic      = beh.get("panic_triggered", False)
    regret_score   = beh.get("regret_score", 0)

    archetype_str = ", ".join(archetypes) if archetypes else "unclassified"
    regimes_str   = ", ".join(lived_through) if lived_through else "none specified"

    goal_years           = person.get("goal_years") or "not set"
    goal_amount          = person.get("goal_amount")
    monthly_contribution = person.get("monthly_contribution")

    goal_str         = f"₹{goal_amount:,} in {goal_years} years" if goal_amount else "not specified"
    contribution_str = f"₹{monthly_contribution:,}/month" if monthly_contribution else "not specified"

    prompt = f"""Client Profile:
  Age: {age}
  Annual income: ₹{income:,}
  Net worth: ₹{net_worth:,}
  Monthly contribution / SIP: {contribution_str}
  Current portfolio: {eq_pct:.0f}% equities / {bond_pct:.0f}% bonds / {alt_pct:.0f}% alternatives
  Risk score: {risk_score}/100  (0 = extremely cautious, 100 = very aggressive)
  Loss aversion: {loss_aversion:.2f}/1.0  (higher = more emotionally affected by losses)
  Panic threshold: sells out if portfolio drops {abs(panic_threshold_pct):.0f}%
  Persona: {archetype_str}
  Market crashes lived through: {regimes_str}
  Key life events: {le_summary}
  Financial goal: {goal_str}

Simulation Results — scenario: {scenario_key}
  Worst historical drop: {max_dd_pct:.1f}%
  Time to recover from that drop: {recovery_months} months
  Stress index: {stress_index:.2f} / 1.00
  Did panic-selling trigger in the replay? {panic_triggered}
  Cost of panic-selling vs staying put: {hist_beh_delta_pct:+.1f}% of portfolio
  Worst scenario for this client: {worst_scenario}
  Behavioural loss from panic (worst case): {beh_delta_pct:+.1f}%
  Did panic trigger in worst case? {beh_panic}
  Psychological regret score: {regret_score:.2f}/1.0

Monte Carlo (10-year, 1,000 simulations):
  Median expected growth: {mc_median_pct:+.1f}%
  Worst 5% of outcomes: {mc_p5_pct:+.1f}%
  Best 5% of outcomes: {mc_p95_pct:+.1f}%
  % of simulations where client panics: {panic_fraction_pct:.1f}%
  Probability of reaching their goal: {goal_probability:.1f}%

Now produce the JSON response as instructed."""

    message = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        response_format={"type": "json_object"},
    )

    raw = message.choices[0].message.content.strip()
    raw = re.sub(r"^```(?:json)?\s*\n?", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\n?```\s*$", "", raw)
    raw = raw.strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {
            "narrative": raw[:1000] if raw else "Narrative generation failed.",
            "allocation": {
                "equities":          {"pct": 40, "reason": "Default balanced allocation."},
                "debt":              {"pct": 35, "reason": "Default balanced allocation."},
                "real_estate":       {"pct": 15, "reason": "Default balanced allocation."},
                "commodities":       {"pct": 10, "reason": "Default balanced allocation."},
                "overall_reasoning": "Could not generate personalised reasoning.",
            },
            "fund_recommendations": [],
        }

    # Ensure pcts sum to 100 (clamp rounding errors)
    alloc = parsed.get("allocation", {})
    keys = ["equities", "debt", "real_estate", "commodities"]
    total = sum(alloc.get(k, {}).get("pct", 0) for k in keys)
    if total != 100 and total > 0:
        # Scale proportionally
        for k in keys:
            if k in alloc and "pct" in alloc[k]:
                alloc[k]["pct"] = round(alloc[k]["pct"] * 100 / total)
        # Fix rounding residual on equities
        residual = 100 - sum(alloc[k]["pct"] for k in keys)
        alloc["equities"]["pct"] = alloc.get("equities", {}).get("pct", 40) + residual

    # Get fund recommendations from live metric engine
    try:
        goal_years_val = person.get("goal_years") or 10
        panic_thr = risk_profile.get("panic_threshold", -0.20)
        fund_recs = match_funds(
            client_risk_score=risk_score,
            panic_threshold=panic_thr,
            goal_years=goal_years_val,
            top_n=5,
        )
    except Exception:
        fund_recs = []

    return {
        "narrative":            parsed.get("narrative", ""),
        "allocation":           alloc,
        "fund_recommendations": fund_recs,
    }
