"""
Historical market regime data.
Monthly return arrays (equity, bond, alt) for four named stress regimes.
All values are monthly total-return approximations sourced from public index data.
"""

from typing import TypedDict


class Scenario(TypedDict):
    name: str
    equity: list[float]
    bond: list[float]
    alt: list[float]


# ---------------------------------------------------------------------------
# dot_com_2000  — 30 months  (Mar 2000 → Aug 2002)
# S&P 500 fell ~49% peak-to-trough; 10-yr Treasury rallied as flight-to-safety
# ---------------------------------------------------------------------------
_DOT_COM_EQUITY: list[float] = [
    -0.015, -0.031, -0.022,  0.023, -0.016,  0.062,  # Mar–Aug 2000
    -0.053, -0.005, -0.080, -0.001,                    # Sep–Dec 2000
     0.035, -0.092, -0.064,  0.077,  0.005,            # Jan–May 2001
    -0.025, -0.010, -0.065, -0.081,  0.018,            # Jun–Oct 2001
     0.076,  0.008,                                     # Nov–Dec 2001
    -0.015, -0.021,  0.037, -0.061, -0.009,            # Jan–May 2002
    -0.072, -0.079,  0.005,                             # Jun–Aug 2002
]

_DOT_COM_BOND: list[float] = [
     0.004,  0.006,  0.003,  0.002,  0.005,  0.003,
     0.007,  0.008,  0.006,  0.005,
     0.004,  0.009,  0.011,  0.006,  0.004,
     0.007,  0.008,  0.012,  0.015,  0.008,
     0.006,  0.003,
     0.005,  0.007,  0.004,  0.006,  0.008,
     0.009,  0.010,  0.007,
]

_DOT_COM_ALT: list[float] = [
    -0.010, -0.015, -0.010,  0.010, -0.008,  0.025,
    -0.020, -0.002, -0.035, -0.001,
     0.015, -0.040, -0.025,  0.030,  0.002,
    -0.010, -0.005, -0.025, -0.030,  0.010,
     0.030,  0.004,
    -0.008, -0.010,  0.015, -0.025, -0.004,
    -0.030, -0.035,  0.003,
]

# ---------------------------------------------------------------------------
# gfc_2008  — 18 months  (Sep 2008 → Feb 2010)
# S&P 500 lost ~55% peak-to-trough; worst single month Oct 2008 ≈ -16.8%
# ---------------------------------------------------------------------------
_GFC_EQUITY: list[float] = [
    -0.091, -0.168, -0.072,  0.009,   # Sep–Dec 2008
    -0.086, -0.110,  0.085,  0.094,   # Jan–Apr 2009
     0.053,  0.002,  0.074,  0.034,   # May–Aug 2009
     0.036, -0.020,  0.057,  0.018,   # Sep–Dec 2009
    -0.036,  0.029,                    # Jan–Feb 2010
]

_GFC_BOND: list[float] = [
     0.010,  0.025,  0.015,  0.012,
     0.010,  0.008, -0.002, -0.003,
     0.004,  0.005,  0.003,  0.004,
     0.003,  0.002,  0.004,  0.003,
     0.002,  0.003,
]

_GFC_ALT: list[float] = [
    -0.050, -0.085, -0.040,  0.005,
    -0.045, -0.060,  0.045,  0.050,
     0.028,  0.001,  0.040,  0.018,
     0.020, -0.010,  0.030,  0.010,
    -0.018,  0.015,
]

# ---------------------------------------------------------------------------
# covid_2020  — 12 months  (Feb 2020 → Jan 2021)
# Fast crash Feb–Mar 2020; sharp V-shaped recovery through rest of year
# ---------------------------------------------------------------------------
_COVID_EQUITY: list[float] = [
    -0.082, -0.125,  0.127,  0.045,  0.018,  0.055,
     0.070, -0.039, -0.028,  0.108,  0.037, -0.011,
]

_COVID_BOND: list[float] = [
     0.018,  0.012, -0.005, -0.002,  0.008,  0.003,
     0.005, -0.003, -0.004, -0.002,  0.001,  0.003,
]

_COVID_ALT: list[float] = [
    -0.040, -0.065,  0.065,  0.022,  0.009,  0.028,
     0.035, -0.020, -0.014,  0.055,  0.018, -0.006,
]

# ---------------------------------------------------------------------------
# rate_shock_2022  — 12 months  (Jan 2022 → Dec 2022)
# Rare simultaneous equity AND bond drawdown driven by Fed rate hikes
# Bloomberg US Agg Bond fell ~13%; S&P 500 fell ~19%
# ---------------------------------------------------------------------------
_RATE_EQUITY: list[float] = [
    -0.052, -0.030,  0.036, -0.087,  0.000, -0.084,
     0.091, -0.042, -0.093,  0.080,  0.054, -0.059,
]

_RATE_BOND: list[float] = [
    -0.022, -0.011, -0.028, -0.038, -0.001, -0.016,
     0.024, -0.028, -0.043,  0.013,  0.029, -0.005,
]

_RATE_ALT: list[float] = [
    -0.030, -0.020,  0.020, -0.045, -0.005, -0.040,
     0.045, -0.022, -0.048,  0.040,  0.028, -0.030,
]

# ---------------------------------------------------------------------------
# ilfs_2018  — 8 months  (Sep 2018 → Apr 2019)
# India's shadow banking sector seized up. Nifty fell ~15% peak-to-trough.
# Credit spreads exploded. NBFCs froze.
# ---------------------------------------------------------------------------
_ILFS_EQUITY: list[float] = [-0.065, -0.085, -0.030, -0.025, 0.040, 0.030, 0.025, 0.035]
_ILFS_BOND:   list[float] = [-0.025, -0.040, -0.015, -0.010, 0.008, 0.012, 0.010, 0.008]
_ILFS_ALT:    list[float] = [-0.035, -0.050, -0.020, -0.015, 0.020, 0.018, 0.015, 0.020]

# ---------------------------------------------------------------------------
# taper_tantrum_2013  — 5 months  (May 2013 → Sep 2013)
# Fed taper fears: INR/USD collapsed from 54→68. FII outflows. Nifty fell ~10%.
# Bonds crushed by RBI emergency rate hike to 10.25%.
# ---------------------------------------------------------------------------
_TAPER_EQUITY: list[float] = [-0.048, -0.030, -0.025, 0.035, 0.015]
_TAPER_BOND:   list[float] = [-0.035, -0.045, -0.020, 0.010, 0.008]
_TAPER_ALT:    list[float] = [-0.025, -0.018, -0.012, 0.018, 0.010]

# ---------------------------------------------------------------------------
# china_selloff_2015  — 8 months  (Jul 2015 → Feb 2016)
# China's A-share bubble burst. Global EM selloff. Nifty fell ~13%.
# Commodities cratered (oil to $27). INR weakened.
# ---------------------------------------------------------------------------
_CHINA_EQUITY: list[float] = [-0.035, -0.070, -0.045, 0.020, -0.025, -0.020, 0.035, 0.030]
_CHINA_BOND:   list[float] = [0.008,   0.010,  0.012, 0.005,  0.006,  0.008, 0.004, 0.003]
_CHINA_ALT:    list[float] = [-0.055, -0.085, -0.060, 0.010, -0.040, -0.030, 0.015, 0.012]

# ---------------------------------------------------------------------------
# rbi_rate_hike_2022  — 9 months  (Oct 2021 → Jun 2022)
# Global inflation shock + RBI hiking repo from 4%→6.5%. Nifty fell ~15%
# Oct peak to Jun trough. Long-duration debt funds fell 8-12%. IT sector worst hit.
# ---------------------------------------------------------------------------
_RBI_EQUITY: list[float] = [-0.002, -0.018, 0.010, -0.030, -0.025, -0.052, -0.048, -0.030, 0.025]
_RBI_BOND:   list[float] = [-0.012, -0.015, -0.008, -0.018, -0.022, -0.025, -0.015, -0.010, 0.005]
_RBI_ALT:    list[float] = [-0.005, -0.010, 0.008, -0.020, -0.015, -0.030, -0.025, -0.015, 0.015]

# ---------------------------------------------------------------------------
# Public registry
# ---------------------------------------------------------------------------
SCENARIOS: dict[str, Scenario] = {
    "dot_com_2000": {
        "name": "Dot-Com Bust (2000–2002)",
        "equity": _DOT_COM_EQUITY,
        "bond": _DOT_COM_BOND,
        "alt": _DOT_COM_ALT,
    },
    "gfc_2008": {
        "name": "Global Financial Crisis (2008–2009)",
        "equity": _GFC_EQUITY,
        "bond": _GFC_BOND,
        "alt": _GFC_ALT,
    },
    "covid_2020": {
        "name": "COVID-19 Crash (2020)",
        "equity": _COVID_EQUITY,
        "bond": _COVID_BOND,
        "alt": _COVID_ALT,
    },
    "rate_shock_2022": {
        "name": "Rate Shock (2022)",
        "equity": _RATE_EQUITY,
        "bond": _RATE_BOND,
        "alt": _RATE_ALT,
    },
    "ilfs_2018": {
        "name": "IL&FS Credit Crisis (2018)",
        "equity": _ILFS_EQUITY,
        "bond": _ILFS_BOND,
        "alt": _ILFS_ALT,
    },
    "taper_tantrum_2013": {
        "name": "Taper Tantrum — INR Crisis (2013)",
        "equity": _TAPER_EQUITY,
        "bond": _TAPER_BOND,
        "alt": _TAPER_ALT,
    },
    "china_selloff_2015": {
        "name": "China Selloff Contagion (2015–16)",
        "equity": _CHINA_EQUITY,
        "bond": _CHINA_BOND,
        "alt": _CHINA_ALT,
    },
    "rbi_rate_hike_2022": {
        "name": "RBI Rate Hike Cycle (2021–22)",
        "equity": _RBI_EQUITY,
        "bond": _RBI_BOND,
        "alt": _RBI_ALT,
    },
}

# Validate lengths match at import time
for _key, _scenario in SCENARIOS.items():
    for _asset in ("equity", "bond", "alt"):
        assert len(_scenario[_asset]) > 0, f"scenarios.py: {_key}[{_asset}] is empty"
