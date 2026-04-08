"""
Historical stress-test simulation.

Replays a named market-regime crash against the client's current portfolio
weights and risk-profile parameters.

Improvements over v1
--------------------
* Inflation-adjusted real path & drawdown
  Real purchasing-power losses are typically worse than nominal figures.
  A 3 % annual CPI is applied month-by-month so the real max drawdown is
  reported alongside the nominal one.

* Sequence-of-returns (SOR) vulnerability score
  Captures how much the *timing* of losses matters for this specific client.
  Near-retirees (< 10 years to retirement) who suffer an early severe
  drawdown face permanent portfolio impairment even if nominal values recover.
  Score formula:
    sor_vulnerability = stress_index × (1 + sor_weight)
  where sor_weight ∈ [0, 1] peaks when the client is at retirement age.

Returns (additions to v1)
-------
  real_max_drawdown   — inflation-adjusted peak-to-trough loss (negative)
  sor_vulnerability   — SOR risk score [0, 1]; 1 = worst possible timing risk
  years_to_retirement — int, used as a diagnostic field downstream
"""

import numpy as np
from scenarios import SCENARIOS

_INFLATION_ANNUAL  = 0.03
_INFLATION_MONTHLY = _INFLATION_ANNUAL / 12.0


def run_historical(scenario_key: str, subgraph: dict) -> dict:
    """
    Parameters
    ----------
    scenario_key : str
        One of the keys in scenarios.SCENARIOS.
    subgraph : dict
        Full person subgraph as returned by graph.read_subgraph().

    Returns
    -------
    dict with keys: max_drawdown, recovery_months, stress_index,
                    panic_triggered, behavioral_delta, path, scenario_key,
                    real_max_drawdown, sor_vulnerability, years_to_retirement.
    """
    scenario     = SCENARIOS[scenario_key]
    portfolio    = subgraph.get("portfolio",    {})
    risk_profile = subgraph.get("risk_profile", {})
    person       = subgraph.get("person",       {})

    eq_pct   = float(portfolio.get("eq_pct",   0.60))
    bond_pct = float(portfolio.get("bond_pct", 0.30))
    alt_pct  = float(portfolio.get("alt_pct",  0.10))

    eq  = np.asarray(scenario["equity"], dtype=np.float64)
    bd  = np.asarray(scenario["bond"],   dtype=np.float64)
    alt = np.asarray(scenario["alt"],    dtype=np.float64)

    monthly_returns: np.ndarray = eq_pct * eq + bond_pct * bd + alt_pct * alt
    n_months = len(monthly_returns)

    # ── Nominal path ──────────────────────────────────────────────────────────
    path    = np.empty(n_months + 1, dtype=np.float64)
    path[0] = 1.0
    path[1:] = np.cumprod(1.0 + monthly_returns)

    # ── Max drawdown (nominal) ────────────────────────────────────────────────
    running_peak    = np.maximum.accumulate(path)
    drawdown_series = (path - running_peak) / running_peak
    max_drawdown    = float(np.min(drawdown_series))
    trough_idx      = int(np.argmin(drawdown_series))
    peak_at_trough  = float(running_peak[trough_idx])

    # ── Recovery months ───────────────────────────────────────────────────────
    recovery_months = 0
    recovered       = False
    for i in range(trough_idx + 1, len(path)):
        if path[i] >= peak_at_trough:
            recovery_months = i - trough_idx
            recovered       = True
            break
    if not recovered and trough_idx > 0:
        # Still underwater at end of scenario window — partial recovery measure
        recovery_months = n_months - trough_idx

    # ── Stress index ──────────────────────────────────────────────────────────
    loss_aversion   = float(risk_profile.get("loss_aversion",   0.50))
    panic_threshold = float(risk_profile.get("panic_threshold", -0.20))

    denom        = abs(panic_threshold) if abs(panic_threshold) > 1e-9 else 1e-9
    stress_index = float(min(1.0, loss_aversion * abs(max_drawdown) / denom))

    # ── Panic trigger ─────────────────────────────────────────────────────────
    panic_triggered: bool = bool(max_drawdown <= panic_threshold)

    # ── Behavioral delta (6-month re-entry rule) ──────────────────────────────
    behavioral_delta = 0.0
    if panic_triggered:
        panic_indices = np.where(drawdown_series <= panic_threshold)[0]
        if len(panic_indices) > 0:
            panic_month     = int(panic_indices[0])
            reentry_month   = min(panic_month + 6, len(path) - 1)
            stay_final      = float(path[-1])
            price_at_panic  = float(path[panic_month])
            price_at_reentry = float(path[reentry_month])

            if price_at_reentry > 1e-12:
                shares_rebought = price_at_panic / price_at_reentry
                panic_final     = shares_rebought * stay_final
            else:
                panic_final = price_at_panic

            behavioral_delta = float(stay_final - panic_final)

    # ── Real (inflation-adjusted) path & drawdown ─────────────────────────────
    # Deflate each month's portfolio value by cumulative CPI since inception.
    month_indices    = np.arange(n_months + 1, dtype=np.float64)
    inflation_factors = (1.0 + _INFLATION_MONTHLY) ** month_indices
    real_path        = path / inflation_factors

    real_running_peak    = np.maximum.accumulate(real_path)
    real_drawdown_series = (real_path - real_running_peak) / real_running_peak
    real_max_drawdown    = float(np.min(real_drawdown_series))

    # ── Sequence-of-returns vulnerability ────────────────────────────────────
    # How badly does it hurt if this crash lands right at / just after retirement?
    # sor_weight peaks at 1.0 when the client is exactly at retirement age (65).
    age                 = float(person.get("age", 40))
    years_to_retirement = max(int(65 - age), 0)

    if years_to_retirement <= 10:
        # Linear weight: 1.0 at retirement, 0.0 ten years before
        sor_weight = float(1.0 - years_to_retirement / 10.0)
    else:
        # Well before retirement — SOR still matters but less acutely
        sor_weight = 0.0

    # Stress already captures loss magnitude; scale up for SOR weight
    sor_vulnerability = float(min(1.0, stress_index * (1.0 + sor_weight)))

    return {
        # v1 fields (unchanged)
        "max_drawdown":    max_drawdown,
        "recovery_months": int(recovery_months),
        "stress_index":    stress_index,
        "panic_triggered": panic_triggered,
        "behavioral_delta": behavioral_delta,
        "path":            path.tolist(),
        "scenario_key":    scenario_key,
        # New fields
        "real_max_drawdown":    real_max_drawdown,
        "sor_vulnerability":    sor_vulnerability,
        "years_to_retirement":  years_to_retirement,
    }
