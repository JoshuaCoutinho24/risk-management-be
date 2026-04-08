"""
Monte Carlo simulation — 1 000 paths over a 10-year horizon.

Improvements over v1
--------------------
* Inflation adjustment   — all terminal values reported in today's purchasing power
                           (3 % annual CPI, configurable via INFLATION_ANNUAL).
* Tax drag               — 15 % LTCG applied annually on real gains above the
                           3 % risk-free rate; produces after-tax terminal values.
* Goal probability       — if person.goal_amount and person.goal_years are set,
                           computes fraction of paths that reach the stated wealth goal,
                           correctly accounting for monthly SIP contributions.
* SOR risk score         — measures dispersion of outcomes in the first 3 years
                           weighted by proximity to retirement; near-retirees with
                           high early-period variance score close to 1.0.

Returns (additions to v1)
-------
  mc_real_median_10yr   — inflation-adjusted median terminal value
  mc_real_p5_10yr       — inflation-adjusted 5th-percentile terminal value
  mc_real_p95_10yr      — inflation-adjusted 95th-percentile terminal value
  goal_probability      — fraction of paths that reach person.goal_amount (0 if not set)
  tax_drag_cost         — nominal median minus after-tax median (drag in portfolio units)
  sor_risk_score        — sequence-of-returns risk [0, 1]
  inflation_rate_used   — CPI assumption used (float)
"""

import numpy as np
from scipy.linalg import cholesky

# ── Capital-market assumptions ────────────────────────────────────────────────
_ANN_MUS  = np.array([0.09, 0.04, 0.06], dtype=np.float64)   # eq, bond, alt
_ANN_SIGS = np.array([0.17, 0.06, 0.10], dtype=np.float64)

_CORR = np.array(
    [
        [ 1.00, -0.20,  0.30],
        [-0.20,  1.00,  0.10],
        [ 0.30,  0.10,  1.00],
    ],
    dtype=np.float64,
)

_N_RUNS   = 1_000
_N_MONTHS = 120          # 10 years

# ── Macro assumptions ─────────────────────────────────────────────────────────
_INFLATION_ANNUAL  = 0.03    # 3 % annual CPI
_TAX_RATE          = 0.15    # 15 % LTCG / qualified-dividend rate
_RISK_FREE_ANNUAL  = 0.03    # 3 % nominal risk-free (for tax-drag calculation)


def run_monte_carlo(subgraph: dict) -> dict:
    """
    Parameters
    ----------
    subgraph : dict
        Full person subgraph from graph.read_subgraph().

    Returns
    -------
    dict — see module docstring for all keys.
    """
    rng = np.random.default_rng(seed=42)

    portfolio    = subgraph.get("portfolio",    {})
    risk_profile = subgraph.get("risk_profile", {})
    person       = subgraph.get("person",       {})

    weights = np.array(
        [
            float(portfolio.get("eq_pct",   0.60)),
            float(portfolio.get("bond_pct", 0.30)),
            float(portfolio.get("alt_pct",  0.10)),
        ],
        dtype=np.float64,
    )
    panic_threshold      = float(risk_profile.get("panic_threshold", -0.20))
    monthly_contribution = float(portfolio.get("monthly_contribution", 0.0))

    # ── Monthly parameters ────────────────────────────────────────────────────
    monthly_mus  = _ANN_MUS  / 12.0
    monthly_sigs = _ANN_SIGS / np.sqrt(12.0)

    cov = _CORR * np.outer(monthly_sigs, monthly_sigs)
    L   = cholesky(cov, lower=True)

    z             = rng.standard_normal((_N_RUNS, _N_MONTHS, 3))
    asset_returns = z @ L.T + monthly_mus          # (N, M, 3)
    port_returns  = asset_returns @ weights         # (N, M)

    # ── Nominal paths ─────────────────────────────────────────────────────────
    paths = np.ones((_N_RUNS, _N_MONTHS + 1), dtype=np.float64)
    paths[:, 1:] = np.cumprod(1.0 + port_returns, axis=1)

    final_values = paths[:, -1]

    mc_median_10yr = float(np.median(final_values))
    mc_p5_10yr     = float(np.percentile(final_values, 5))
    mc_p95_10yr    = float(np.percentile(final_values, 95))
    mc_mean_10yr   = float(np.mean(final_values))

    # ── Panic fraction ────────────────────────────────────────────────────────
    running_peak  = np.maximum.accumulate(paths, axis=1)
    drawdowns     = (paths - running_peak) / running_peak
    min_drawdowns = drawdowns.min(axis=1)
    panic_fraction = float(np.mean(min_drawdowns <= panic_threshold))

    # ── Inflation adjustment ──────────────────────────────────────────────────
    cumulative_inflation = (1.0 + _INFLATION_ANNUAL) ** 10
    real_final_values    = final_values / cumulative_inflation

    mc_real_median_10yr = float(np.median(real_final_values))
    mc_real_p5_10yr     = float(np.percentile(real_final_values, 5))
    mc_real_p95_10yr    = float(np.percentile(real_final_values, 95))

    # ── Tax drag ──────────────────────────────────────────────────────────────
    # Apply LTCG tax annually on gains above the risk-free rate.
    # Convert paths to annual returns at year-end snapshots.
    annual_snapshots = paths[:, ::12]                         # (N, 11) — years 0..10
    annual_gross_ret = annual_snapshots[:, 1:] / annual_snapshots[:, :-1] - 1.0  # (N, 10)
    taxable_gain     = np.maximum(0.0, annual_gross_ret - _RISK_FREE_ANNUAL)
    after_tax_ret    = annual_gross_ret - _TAX_RATE * taxable_gain
    after_tax_terminal = np.prod(1.0 + after_tax_ret, axis=1)   # (N,)
    tax_drag_cost = float(mc_median_10yr - float(np.median(after_tax_terminal)))

    # ── Goal probability ──────────────────────────────────────────────────────
    goal_amount   = float(person.get("goal_amount",  0.0))
    goal_years    = int(person.get("goal_years",    10))
    net_worth     = float(person.get("net_worth",    0.0))

    goal_probability = 0.0
    if goal_amount > 0.0 and (net_worth > 0.0 or monthly_contribution > 0.0):
        goal_horizon = min(goal_years * 12, _N_MONTHS)
        terminal     = paths[:, goal_horizon]              # (N,)

        # Portfolio component: existing wealth compounds along the path
        portfolio_terminal = net_worth * terminal          # (N,)

        if monthly_contribution > 0.0 and goal_horizon > 0:
            # Each monthly SIP at time t grows by path[T] / path[t] to reach horizon T.
            # monthly_paths[:, t] is the portfolio multiplier at month t (1-indexed).
            monthly_paths      = paths[:, 1 : goal_horizon + 1]   # (N, T)
            # Avoid division by near-zero early path values
            safe_monthly       = np.where(monthly_paths > 1e-12, monthly_paths, 1e-12)
            contribution_fv    = monthly_contribution * np.sum(
                terminal[:, np.newaxis] / safe_monthly, axis=1
            )
            total_value = portfolio_terminal + contribution_fv
        else:
            total_value = portfolio_terminal

        goal_probability = float(np.mean(total_value >= goal_amount))

    # ── Sequence-of-returns risk score ────────────────────────────────────────
    # Measures dispersion in the first 3 years, weighted by proximity to retirement.
    # Close-to-retirement clients with high early-period variance are most vulnerable.
    age                = float(person.get("age", 40))
    years_to_retirement = max(65.0 - age, 0.0)

    sor_horizon    = min(36, _N_MONTHS)                    # first 3 years
    early_values   = paths[:, sor_horizon]
    early_p5       = float(np.percentile(early_values, 5))
    early_p95      = float(np.percentile(early_values, 95))
    dispersion     = (early_p95 - early_p5) / max(abs(early_p5), 1e-9)

    # Weight: 1.0 when at retirement (0 years left), 0.0 when 10+ years away
    retirement_weight = float(np.clip(1.0 - years_to_retirement / 10.0, 0.0, 1.0))
    sor_risk_score    = float(np.clip(dispersion * retirement_weight * 0.35, 0.0, 1.0))

    # ── Sample paths (5 representative runs for charting) ─────────────────────
    percentile_targets  = [10, 30, 50, 70, 90]
    sorted_indices      = np.argsort(final_values)
    sample_run_indices  = [
        int(sorted_indices[int(_N_RUNS * p / 100)]) for p in percentile_targets
    ]
    yearly_indices = list(range(0, _N_MONTHS + 1, 12))
    sample_paths   = paths[sample_run_indices][:, yearly_indices].tolist()

    return {
        # v1 fields (unchanged keys, backward-compatible)
        "mc_median_10yr":  mc_median_10yr,
        "mc_p5_10yr":      mc_p5_10yr,
        "mc_p95_10yr":     mc_p95_10yr,
        "mc_mean_10yr":    mc_mean_10yr,
        "panic_fraction":  panic_fraction,
        "sample_paths":    sample_paths,
        # New fields
        "mc_real_median_10yr": mc_real_median_10yr,
        "mc_real_p5_10yr":     mc_real_p5_10yr,
        "mc_real_p95_10yr":    mc_real_p95_10yr,
        "goal_probability":    goal_probability,
        "tax_drag_cost":       tax_drag_cost,
        "sor_risk_score":      sor_risk_score,
        "inflation_rate_used": _INFLATION_ANNUAL,
    }
