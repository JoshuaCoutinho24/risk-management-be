"""
Behavioural finance simulation.

Identifies the worst historical scenario for the client's specific allocation,
then models the financial cost of panic-selling at the client's stated threshold
versus staying invested throughout.

Improvements over v1
--------------------
* Cash yield during panic period
  Instead of a 0 % cash return while waiting to re-enter, the client's idle cash
  earns a money-market yield (4 % annual, reflecting typical HYSA / T-bill rates
  or RBI repo-rate proximity for INR portfolios).  This slightly reduces the
  cost-of-panic figures and is reported separately so the advisor can see the
  gross vs net impact.

* Regret score
  A normalised [0, 1] measure of how much psychological regret the client would
  likely experience.  It combines:
    - the relative wealth destroyed by panic-selling (vs staying)
    - how quickly the market recovered after the panic (faster recovery = more regret
      because the investor sold at the bottom and watched others bounce back)

Returns (additions to v1)
-------
  cash_yield_benefit    — additional terminal value from earning money-market yield
                          during the panic-out period (6-month delay baseline)
  regret_score          — psychological regret index [0, 1]
  cost_of_panic         — each entry now includes 'cash_yield_benefit' and
                          'panic_final_no_yield' for comparison
"""

import numpy as np
from scenarios import SCENARIOS

_CASH_ANNUAL_YIELD = 0.04          # 4 % annual money-market / HYSA rate
_CASH_MONTHLY_YIELD = _CASH_ANNUAL_YIELD / 12.0


def _build_path(eq_pct: float, bond_pct: float, alt_pct: float, scenario_key: str) -> np.ndarray:
    scenario = SCENARIOS[scenario_key]
    eq  = np.asarray(scenario["equity"], dtype=np.float64)
    bd  = np.asarray(scenario["bond"],   dtype=np.float64)
    alt = np.asarray(scenario["alt"],    dtype=np.float64)
    monthly_returns = eq_pct * eq + bond_pct * bd + alt_pct * alt
    path    = np.empty(len(monthly_returns) + 1, dtype=np.float64)
    path[0] = 1.0
    path[1:] = np.cumprod(1.0 + monthly_returns)
    return path


def _max_drawdown(path: np.ndarray) -> float:
    peak = np.maximum.accumulate(path)
    dd   = (path - peak) / peak
    return float(np.min(dd))


def run_behavioural(subgraph: dict) -> dict:
    """
    Parameters
    ----------
    subgraph : dict
        Full person subgraph from graph.read_subgraph().

    Returns
    -------
    dict — see module docstring.
    """
    portfolio    = subgraph.get("portfolio",    {})
    risk_profile = subgraph.get("risk_profile", {})

    eq_pct          = float(portfolio.get("eq_pct",          0.60))
    bond_pct        = float(portfolio.get("bond_pct",        0.30))
    alt_pct         = float(portfolio.get("alt_pct",         0.10))
    panic_threshold = float(risk_profile.get("panic_threshold", -0.20))

    # ── Find worst historical scenario for this allocation ────────────────────
    worst_scenario_key: str       = "gfc_2008"
    worst_max_dd:       float     = 0.0
    worst_path:         np.ndarray | None = None

    for key in SCENARIOS:
        p  = _build_path(eq_pct, bond_pct, alt_pct, key)
        dd = _max_drawdown(p)
        if dd < worst_max_dd:
            worst_max_dd       = dd
            worst_scenario_key = key
            worst_path         = p

    if worst_path is None:
        worst_path   = _build_path(eq_pct, bond_pct, alt_pct, "gfc_2008")
        worst_max_dd = _max_drawdown(worst_path)

    path      = worst_path
    held_path = path.copy()

    # ── Drawdown series ───────────────────────────────────────────────────────
    running_peak    = np.maximum.accumulate(path)
    drawdown_series = (path - running_peak) / running_peak
    trough_idx      = int(np.argmin(drawdown_series))

    # ── Panic trigger ─────────────────────────────────────────────────────────
    breach_indices   = np.where(drawdown_series <= panic_threshold)[0]
    panic_triggered: bool = len(breach_indices) > 0

    cost_of_panic:    dict  = {}
    behavioral_delta: float = 0.0
    cash_yield_benefit: float = 0.0
    regret_score:     float = 0.0

    if panic_triggered:
        panic_month      = int(breach_indices[0])
        stay_final       = float(path[-1])
        price_at_panic   = float(path[panic_month])
        n_months_total   = len(path) - 1

        for delay in (3, 6, 12):
            reentry_month    = min(panic_month + delay, n_months_total)
            price_at_reentry = float(path[reentry_month])

            if price_at_reentry > 1e-12:
                # ── Baseline: 0 % cash yield ──────────────────────────────
                shares_no_yield  = price_at_panic / price_at_reentry
                panic_final_0    = shares_no_yield * stay_final

                # ── With money-market yield ───────────────────────────────
                # Cash compounds at _CASH_MONTHLY_YIELD for `delay` months.
                cash_at_reentry  = price_at_panic * (1.0 + _CASH_MONTHLY_YIELD) ** delay
                shares_with_yield = cash_at_reentry / price_at_reentry
                panic_final_yield = shares_with_yield * stay_final
            else:
                # Degenerate: price collapsed to zero — stay in cash either way
                panic_final_0     = price_at_panic * (1.0 + _CASH_MONTHLY_YIELD) ** delay
                panic_final_yield = panic_final_0

            delta_yield         = float(stay_final - panic_final_yield)
            delta_no_yield      = float(stay_final - panic_final_0)
            benefit_this_delay  = float(panic_final_yield - panic_final_0)

            cost_of_panic[f"delay_{delay}m"] = {
                "stay_final":          round(stay_final,       6),
                "panic_final":         round(panic_final_yield, 6),
                "panic_final_no_yield": round(panic_final_0,   6),
                "delta":               round(delta_yield,      6),
                "delta_pct":           round(delta_yield * 100, 2),
                "reentry_month":       reentry_month,
                "cash_yield_benefit":  round(benefit_this_delay, 6),
            }

        # ── Canonical delta: 6-month re-entry with money-market yield ──────
        behavioral_delta   = cost_of_panic["delay_6m"]["delta"]
        cash_yield_benefit = cost_of_panic["delay_6m"]["cash_yield_benefit"]

        # ── Regret score ──────────────────────────────────────────────────
        # Two drivers: (a) relative wealth cost, (b) speed of market recovery.
        # Fast recovery after panic = more regret; investor sold at the exact bottom.
        stay_final_ref   = cost_of_panic["delay_6m"]["stay_final"]
        relative_cost    = abs(behavioral_delta) / max(abs(stay_final_ref), 1e-9)

        # Months between panic and trough recovery (shorter = more regret)
        recovery_months_after_panic = max(trough_idx - panic_month, 1)
        # Normalise so 1 month → 1.0 speed, 12 months → ~0.08 speed
        recovery_speed = 1.0 / recovery_months_after_panic

        regret_score = float(np.clip(relative_cost * (1.0 + recovery_speed * 5.0), 0.0, 1.0))

    return {
        # v1 fields (unchanged)
        "behavioral_delta":  behavioral_delta,
        "panic_triggered":   panic_triggered,
        "held_path":         held_path.tolist(),
        "cost_of_panic":     cost_of_panic,
        "worst_scenario_key": worst_scenario_key,
        "worst_max_drawdown": worst_max_dd,
        "trough_idx":        trough_idx,
        # New fields
        "cash_yield_benefit": cash_yield_benefit,
        "regret_score":       regret_score,
    }
