"""
Live fund risk metrics engine.

Fetches NAV history from MFAPI.in for each fund in fund_catalog.json,
computes risk metrics mathematically, caches to fund_cache.json,
and provides a matching function that scores funds against a client's
risk profile.
"""

import json
import math
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import urllib.request
import urllib.error

# ── Paths ─────────────────────────────────────────────────────────────────────

_BASE_DIR    = Path(__file__).parent
_DATA_DIR    = Path(os.getenv("DATA_DIR", str(_BASE_DIR)))
_CATALOG     = _BASE_DIR / "fund_catalog.json"   # source file — ships with the app
_CACHE_FILE  = _DATA_DIR / "fund_cache.json"     # runtime cache — lives in DATA_DIR


# ── Network helpers ───────────────────────────────────────────────────────────

def _fetch_nav(scheme_code: int) -> list[dict]:
    """
    Fetch NAV history from MFAPI.in for a given scheme code.

    Returns the ``data`` list from the JSON response, or an empty list on any
    network / parse failure.
    """
    url = f"https://api.mfapi.in/mf/{scheme_code}"
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "RiskEnginePro/1.0"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read().decode("utf-8")
        payload = json.loads(body)
        return payload.get("data", [])
    except Exception:
        return []


# ── Parsing ───────────────────────────────────────────────────────────────────

def _parse_navs(raw: list[dict]) -> list[tuple[datetime, float]]:
    """
    Parse raw MFAPI data records into (datetime, nav_float) tuples.

    MFAPI returns dates as ``"DD-MM-YYYY"`` strings and NAVs as strings.
    The returned list is sorted by date ascending (oldest first).
    """
    parsed: list[tuple[datetime, float]] = []
    for record in raw:
        try:
            date_str = record.get("date", "")
            nav_str  = record.get("nav", "")
            dt  = datetime.strptime(date_str, "%d-%m-%Y")
            nav = float(nav_str)
            parsed.append((dt, nav))
        except (ValueError, TypeError):
            continue
    parsed.sort(key=lambda x: x[0])
    return parsed


# ── Metric computation ────────────────────────────────────────────────────────

def _cagr(navs: list[tuple[datetime, float]], years: int) -> Optional[float]:
    """
    Compute the CAGR for the most recent ``years`` years.

    Returns the value as a percentage rounded to 2 dp, or None if there is
    insufficient history.
    """
    if not navs:
        return None

    latest_dt, latest_nav = navs[-1]
    cutoff = datetime(latest_dt.year - years, latest_dt.month, latest_dt.day)

    # Find closest data point on or just before the cutoff date
    past_nav: Optional[float] = None
    past_dt:  Optional[datetime] = None
    for dt, nav in navs:
        if dt <= cutoff:
            past_nav = nav
            past_dt  = dt
        else:
            break

    if past_nav is None or past_nav <= 0:
        return None

    # Actual fraction of years between the two data points
    actual_years = (latest_dt - past_dt).days / 365.25
    if actual_years < years * 0.75:
        # Less than 75% of requested period available — not reliable
        return None

    cagr_val = (latest_nav / past_nav) ** (1.0 / actual_years) - 1.0
    return round(cagr_val * 100, 2)


def _compute_metrics(navs: list[tuple[datetime, float]]) -> dict:
    """
    Compute a full suite of risk metrics from a NAV history.

    Requires at least 60 data points; returns an empty dict otherwise.
    """
    if len(navs) < 60:
        return {}

    values = [nav for _, nav in navs]

    # ── Log returns ───────────────────────────────────────────────────────────
    log_returns = [
        math.log(values[i] / values[i - 1])
        for i in range(1, len(values))
        if values[i - 1] > 0 and values[i] > 0
    ]

    if len(log_returns) < 30:
        return {}

    # ── Annual volatility ─────────────────────────────────────────────────────
    std_daily = statistics.stdev(log_returns)
    annual_vol = round(std_daily * math.sqrt(252) * 100, 2)  # as %

    # ── Max drawdown ──────────────────────────────────────────────────────────
    peak      = values[0]
    max_dd    = 0.0
    for v in values:
        if v > peak:
            peak = v
        dd = (peak - v) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    max_drawdown = round(max_dd * 100, 2)  # as %

    # ── CAGR ──────────────────────────────────────────────────────────────────
    cagr_1y = _cagr(navs, 1)
    cagr_3y = _cagr(navs, 3)
    cagr_5y = _cagr(navs, 5)

    # ── Sharpe ratio ──────────────────────────────────────────────────────────
    sharpe_ratio: Optional[float] = None
    if cagr_1y is not None and annual_vol and annual_vol > 0:
        sharpe_ratio = round((cagr_1y - 6.5) / annual_vol, 3)

    # ── Risk score 0–100 (normalise vol 0–40% → score 0–100) ─────────────────
    # annual_vol is already in % (e.g. 15.39), so divide by 40, not 0.40
    risk_score = min(100, max(0, round(annual_vol / 40.0 * 100)))

    # ── Latest NAV ───────────────────────────────────────────────────────────
    nav_date_dt, nav_latest = navs[-1]
    nav_date = nav_date_dt.strftime("%d-%m-%Y")

    return {
        "annual_volatility": annual_vol,
        "max_drawdown":      max_drawdown,
        "cagr_1y":           cagr_1y,
        "cagr_3y":           cagr_3y,
        "cagr_5y":           cagr_5y,
        "sharpe_ratio":      sharpe_ratio,
        "risk_score":        risk_score,
        "nav_latest":        round(nav_latest, 4),
        "nav_date":          nav_date,
        "data_points":       len(navs),
    }


# ── Cache management ──────────────────────────────────────────────────────────

def refresh_cache() -> int:
    """
    Fetch NAV history for every fund in ``fund_catalog.json``, compute metrics,
    and persist results to ``fund_cache.json``.

    Prints progress to stdout.  Returns the number of funds successfully cached.
    """
    with open(_CATALOG, encoding="utf-8") as fh:
        catalog: list[dict] = json.load(fh)

    results: list[dict] = []
    total = len(catalog)

    for idx, fund in enumerate(catalog, start=1):
        scheme_code = fund["scheme_code"]
        name        = fund["name"]
        category    = fund["category"]

        raw  = _fetch_nav(scheme_code)
        navs = _parse_navs(raw)
        metrics = _compute_metrics(navs)

        status_char = "OK" if metrics else "SKIP"
        print(f"[fund_metrics] {idx}/{total} {name} [{status_char}]")

        if not metrics:
            continue

        results.append({
            "scheme_code": scheme_code,
            "name":        name,
            "category":    category,
            "metrics":     metrics,
            "cached_at":   datetime.now(timezone.utc).isoformat(),
        })

    with open(_CACHE_FILE, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)

    print(f"[fund_metrics] Cache written: {len(results)}/{total} funds cached OK")
    return len(results)


def load_cache() -> list[dict]:
    """
    Load and return the fund metrics cache from ``fund_cache.json``.

    Returns an empty list if the file is missing or contains invalid JSON.
    """
    if not _CACHE_FILE.exists():
        return []
    try:
        with open(_CACHE_FILE, encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return []


def is_cache_stale(max_age_hours: int = 24) -> bool:
    """
    Return True if the cache is empty or the oldest entry is older than
    ``max_age_hours`` hours.
    """
    cache = load_cache()
    if not cache:
        return True

    now = datetime.now(timezone.utc)
    oldest: Optional[datetime] = None

    for entry in cache:
        ts_str = entry.get("cached_at", "")
        if not ts_str:
            continue
        try:
            ts = datetime.fromisoformat(ts_str)
            # Ensure timezone-aware for comparison
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if oldest is None or ts < oldest:
                oldest = ts
        except ValueError:
            continue

    if oldest is None:
        return True

    age_hours = (now - oldest).total_seconds() / 3600
    return age_hours > max_age_hours


# ── Reason builder ────────────────────────────────────────────────────────────

def _build_reason(
    fund: dict,
    metrics: dict,
    match_score: int,
    client_risk: int,
    goal_years: int | str,
) -> str:
    """
    Build a 2-sentence plain-English reason for recommending (or cautioning
    about) a fund relative to a client's risk profile.
    """
    fund_risk   = metrics["risk_score"]
    annual_vol  = metrics["annual_volatility"]
    cagr_3y     = metrics.get("cagr_3y")

    # Risk comparison
    diff = fund_risk - client_risk
    if abs(diff) <= 10:
        risk_line = (
            f"This fund's risk score ({fund_risk}/100) is a close match to your "
            f"client's risk appetite ({client_risk}/100), "
            f"and it moves roughly \u00b1{annual_vol}% per year."
        )
    elif diff > 0:
        risk_line = (
            f"This fund (risk score {fund_risk}/100) is somewhat more aggressive "
            f"than the client's profile ({client_risk}/100) — it moves \u00b1{annual_vol}% per year, "
            f"so it suits investors who can stomach occasional sharp dips."
        )
    else:
        risk_line = (
            f"This fund (risk score {fund_risk}/100) is more conservative than the "
            f"client's risk appetite ({client_risk}/100), offering steadier but "
            f"potentially lower returns, with annual swings of about \u00b1{annual_vol}%."
        )

    # Return / timeline line
    try:
        goal_int = int(goal_years)
    except (TypeError, ValueError):
        goal_int = 10

    if cagr_3y is not None:
        return_part = f"It has grown at {cagr_3y}% per year over the last 3 years"
        if goal_int <= 3 and fund_risk > 60:
            return_part += (
                ", but with a short investment horizon this level of volatility "
                "carries meaningful timing risk — consider a more stable alternative"
            )
        return_line = return_part + "."
    else:
        if goal_int <= 3 and fund_risk > 60:
            return_line = (
                "With a short investment horizon and high fund volatility, "
                "there is meaningful timing risk — a more stable alternative may be prudent."
            )
        else:
            return_line = (
                "Insufficient return history is available, but the volatility profile "
                "aligns with the client's overall investment timeline."
            )

    return f"{risk_line} {return_line}"


# ── Matching ──────────────────────────────────────────────────────────────────

def match_funds(
    client_risk_score: int,
    panic_threshold: float,
    goal_years: int | str,
    top_n: int = 5,
) -> list[dict]:
    """
    Score every cached fund against a client's risk profile and return the
    top ``top_n`` matches sorted by match score descending.

    Parameters
    ----------
    client_risk_score : int
        Client risk score 0–100.
    panic_threshold : float
        Fractional loss that triggers panic selling, e.g. -0.20 for 20%.
    goal_years : int or str
        Investment horizon in years.  Accepts ``"not set"`` — defaults to 10.
    top_n : int
        Maximum number of results to return.
    """
    # Normalise goal_years
    if isinstance(goal_years, str):
        try:
            goal_years_int = int(goal_years)
        except (ValueError, TypeError):
            goal_years_int = 10
    else:
        try:
            goal_years_int = int(goal_years)
        except (TypeError, ValueError):
            goal_years_int = 10

    client_thr_pct = abs(panic_threshold * 100)  # e.g. -0.20 → 20.0

    cache = load_cache()
    scored: list[dict] = []

    # Deduplicate by scheme_code to avoid double-scoring
    seen_codes: set[int] = set()

    for fund in cache:
        scheme_code = fund.get("scheme_code")
        if scheme_code in seen_codes:
            continue
        seen_codes.add(scheme_code)

        metrics = fund.get("metrics", {})
        if not metrics:
            continue

        fund_risk = metrics.get("risk_score")
        fund_dd   = metrics.get("max_drawdown")
        fund_vol  = metrics.get("annual_volatility")

        if fund_risk is None or fund_dd is None or fund_vol is None:
            continue

        # Quality gate: skip funds with implausible data
        # vol > 60% almost always means <1yr of data or pricing errors
        # data_points < 252 = less than 1 year of trading days
        data_points = fund.get("metrics", {}).get("data_points", 0)
        if fund_vol > 60 or data_points < 252:
            continue

        # ── Scoring formula ───────────────────────────────────────────────────
        base             = 100 - abs(client_risk_score - fund_risk)
        drawdown_penalty = max(0.0, fund_dd - client_thr_pct) * 0.8

        if goal_years_int <= 3:
            timeline_penalty = max(0.0, fund_risk - 40) * 0.5
        elif goal_years_int <= 5:
            timeline_penalty = max(0.0, fund_risk - 60) * 0.3
        else:
            timeline_penalty = 0.0

        cagr_3y      = metrics.get("cagr_3y") or 0.0
        return_bonus = min(15.0, max(0.0, cagr_3y * 0.5))

        raw_score  = base - drawdown_penalty - timeline_penalty + return_bonus
        match_score = min(100, max(0, round(raw_score)))

        scored.append({
            "scheme_code":      scheme_code,
            "name":             fund["name"],
            "category":         fund["category"],
            "match_score":      match_score,
            "annual_volatility": metrics["annual_volatility"],
            "max_drawdown":     metrics["max_drawdown"],
            "cagr_1y":          metrics.get("cagr_1y"),
            "cagr_3y":          metrics.get("cagr_3y"),
            "cagr_5y":          metrics.get("cagr_5y"),
            "sharpe_ratio":     metrics.get("sharpe_ratio"),
            "risk_score":       metrics["risk_score"],
            "nav_latest":       metrics["nav_latest"],
            "reason":           _build_reason(
                fund, metrics, match_score, client_risk_score, goal_years_int
            ),
        })

    scored.sort(key=lambda x: x["match_score"], reverse=True)
    return scored[:top_n]
