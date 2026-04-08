"""
Neo4j graph layer — singleton driver, schema setup, and all read/write helpers.

All writes use MERGE so the entire API is idempotent.
The module-level driver is created lazily on first use.

Improvements over v1
--------------------
* Full-text search indexes on PersonaArchetype.name and LifeEvent.type
  enabling substring / fuzzy client discovery.

* search_clients_by_archetype(archetype, limit)
  Returns matching persons with their risk + portfolio summary.

* get_archetype_benchmark(archetype)
  Aggregates risk scores, drawdowns, and MC medians across all clients
  sharing an archetype so advisors can benchmark any individual.

* get_simulation_drift(person_id)
  Returns all SimulationRun nodes for a person in chronological order,
  enabling the frontend to chart how risk metrics have drifted over time.

* Extended write_simulation_run — persists all new simulation fields:
  real_max_drawdown, sor_vulnerability, sor_risk_score, goal_probability,
  tax_drag_cost, mc_real_median_10yr, mc_real_p5_10yr,
  cash_yield_benefit, regret_score, mc_p95_10yr, panic_fraction.

* Extended write_person — persists optional Portfolio.monthly_contribution
  and Person.goal_amount, Person.goal_years for goal-probability calc.
"""

import hashlib
import os
from typing import Optional

from dotenv import load_dotenv
from neo4j import GraphDatabase, Driver

load_dotenv()

# ── Singleton driver ──────────────────────────────────────────────────────────

_driver: Optional[Driver] = None


def get_driver() -> Driver:
    global _driver
    if _driver is None:
        uri      = os.environ.get("NEO4J_URI",      "bolt://localhost:7687")
        user     = os.environ.get("NEO4J_USER",     "neo4j")
        password = os.environ.get("NEO4J_PASSWORD", "password")
        _driver  = GraphDatabase.driver(uri, auth=(user, password))
    return _driver


# ── Schema / startup helpers ──────────────────────────────────────────────────

_CONSTRAINTS = [
    "CREATE CONSTRAINT person_id          IF NOT EXISTS FOR (n:Person)          REQUIRE n.id   IS UNIQUE",
    "CREATE CONSTRAINT portfolio_id        IF NOT EXISTS FOR (n:Portfolio)        REQUIRE n.id   IS UNIQUE",
    "CREATE CONSTRAINT risk_profile_id     IF NOT EXISTS FOR (n:RiskProfile)     REQUIRE n.id   IS UNIQUE",
    "CREATE CONSTRAINT life_event_id       IF NOT EXISTS FOR (n:LifeEvent)       REQUIRE n.id   IS UNIQUE",
    "CREATE CONSTRAINT persona_archetype   IF NOT EXISTS FOR (n:PersonaArchetype) REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT market_regime_key   IF NOT EXISTS FOR (n:MarketRegime)    REQUIRE n.key  IS UNIQUE",
    "CREATE CONSTRAINT simulation_run_id   IF NOT EXISTS FOR (n:SimulationRun)   REQUIRE n.id   IS UNIQUE",
]

# Neo4j 5.x full-text index syntax; wrapped in try/except for version tolerance
_FULLTEXT_INDEXES = [
    "CREATE FULLTEXT INDEX archetype_ft IF NOT EXISTS FOR (n:PersonaArchetype) ON EACH [n.name]",
    "CREATE FULLTEXT INDEX life_event_ft IF NOT EXISTS FOR (n:LifeEvent) ON EACH [n.type]",
]

_MARKET_REGIMES = [
    {"key": "dot_com_2000",    "name": "Dot-Com Bust (2000–2002)"},
    {"key": "gfc_2008",        "name": "Global Financial Crisis (2008–2009)"},
    {"key": "covid_2020",      "name": "COVID-19 Crash (2020)"},
    {"key": "rate_shock_2022", "name": "Rate Shock (2022)"},
]


def create_indexes() -> None:
    """Create uniqueness constraints and full-text indexes on startup."""
    driver = get_driver()
    with driver.session() as session:
        for cypher in _CONSTRAINTS:
            try:
                session.run(cypher)
            except Exception:
                pass  # Constraint already exists or DB version mismatch

        for cypher in _FULLTEXT_INDEXES:
            try:
                session.run(cypher)
            except Exception:
                # Fall back to legacy 4.x procedure if 5.x syntax not available
                pass


def seed_market_regimes() -> None:
    """MERGE the four canonical MarketRegime nodes.  Idempotent."""
    driver = get_driver()
    with driver.session() as session:
        for regime in _MARKET_REGIMES:
            session.run(
                "MERGE (mr:MarketRegime {key: $key}) SET mr.name = $name",
                key=regime["key"],
                name=regime["name"],
            )


# ── Write helpers ─────────────────────────────────────────────────────────────

def _life_event_id(person_id: str, event: dict) -> str:
    """Deterministic ID so re-ingesting the same client is idempotent."""
    key = f"{person_id}:{event.get('type', '')}:{event.get('date', '')}"
    return hashlib.sha256(key.encode()).hexdigest()[:24]


def write_person(person_id: str, entities: dict) -> None:
    """
    Persist all extracted entities for a person.  Every write uses MERGE.

    Extended in v2 to handle optional fields:
      - portfolio.monthly_contribution (monthly SIP amount, any currency)
      - person.goal_amount             (target wealth, same currency as net_worth)
      - person.goal_years              (years until goal target)
    """
    driver = get_driver()

    person       = entities.get("person",       {})
    portfolio    = entities.get("portfolio",    {})
    risk_profile = entities.get("risk_profile", {})
    life_events  = entities.get("life_events",  [])
    archetypes   = entities.get("archetypes",   [])
    lived_through = entities.get("lived_through_regimes", [])

    with driver.session() as session:

        # ── Person ────────────────────────────────────────────────────────────
        session.run(
            """
            MERGE (p:Person {id: $id})
            SET p.name       = $name,
                p.age        = $age,
                p.income     = $income,
                p.net_worth  = $net_worth,
                p.goal_amount = $goal_amount,
                p.goal_years  = $goal_years
            """,
            id=person_id,
            name=str(person.get("name", "")),
            age=int(person.get("age", 0)),
            income=int(person.get("income", 0)),
            net_worth=int(person.get("net_worth", 0)),
            goal_amount=float(person.get("goal_amount", 0.0)),
            goal_years=int(person.get("goal_years", 10)),
        )

        # ── RiskProfile ───────────────────────────────────────────────────────
        rp_id = f"rp_{person_id}"
        session.run(
            """
            MERGE (rp:RiskProfile {id: $rp_id})
            SET rp.score           = $score,
                rp.loss_aversion   = $loss_aversion,
                rp.panic_threshold = $panic_threshold
            WITH rp
            MATCH (p:Person {id: $person_id})
            MERGE (p)-[:HAS_RISK_PROFILE]->(rp)
            """,
            rp_id=rp_id,
            score=int(risk_profile.get("score", 50)),
            loss_aversion=float(risk_profile.get("loss_aversion", 0.5)),
            panic_threshold=float(risk_profile.get("panic_threshold", -0.20)),
            person_id=person_id,
        )

        # ── Portfolio ─────────────────────────────────────────────────────────
        port_id = f"port_{person_id}"
        session.run(
            """
            MERGE (port:Portfolio {id: $port_id})
            SET port.eq_pct               = $eq_pct,
                port.bond_pct             = $bond_pct,
                port.alt_pct              = $alt_pct,
                port.monthly_contribution = $monthly_contribution
            WITH port
            MATCH (p:Person {id: $person_id})
            MERGE (p)-[:HOLDS]->(port)
            """,
            port_id=port_id,
            eq_pct=float(portfolio.get("eq_pct", 0.60)),
            bond_pct=float(portfolio.get("bond_pct", 0.30)),
            alt_pct=float(portfolio.get("alt_pct", 0.10)),
            monthly_contribution=float(portfolio.get("monthly_contribution", 0.0)),
            person_id=person_id,
        )

        # ── LifeEvents ────────────────────────────────────────────────────────
        for event in life_events:
            le_id = _life_event_id(person_id, event)
            session.run(
                """
                MERGE (le:LifeEvent {id: $le_id})
                SET le.type   = $type,
                    le.date   = $date,
                    le.impact = $impact
                WITH le
                MATCH (p:Person {id: $person_id})
                MERGE (p)-[:EXPERIENCED]->(le)
                """,
                le_id=le_id,
                type=str(event.get("type", "")),
                date=str(event.get("date", "")),
                impact=str(event.get("impact", "low")),
                person_id=person_id,
            )

        # ── PersonaArchetypes ─────────────────────────────────────────────────
        for archetype in archetypes:
            session.run(
                """
                MERGE (pa:PersonaArchetype {name: $name})
                WITH pa
                MATCH (p:Person {id: $person_id})
                MERGE (p)-[:MATCHES_ARCHETYPE]->(pa)
                """,
                name=str(archetype),
                person_id=person_id,
            )

        # ── MarketRegimes lived through ───────────────────────────────────────
        for regime_key in lived_through:
            session.run(
                """
                MATCH (mr:MarketRegime {key: $key})
                MATCH (p:Person {id: $person_id})
                MERGE (p)-[:LIVED_THROUGH]->(mr)
                """,
                key=str(regime_key),
                person_id=person_id,
            )


def write_simulation_run(sim_data: dict) -> None:
    """
    Persist a SimulationRun node and wire it to Person and MarketRegime.
    MERGE on id — re-running the same simulation_id is a safe no-op.

    Extended in v2 with new simulation fields from all three engines.
    """
    driver = get_driver()
    with driver.session() as session:
        session.run(
            """
            MERGE (sr:SimulationRun {id: $id})
            SET
                sr.person_id           = $person_id,
                sr.scenario            = $scenario,
                sr.created_at          = $created_at,

                sr.max_drawdown        = $max_drawdown,
                sr.real_max_drawdown   = $real_max_drawdown,
                sr.recovery_months     = $recovery_months,
                sr.stress_index        = $stress_index,
                sr.panic_triggered     = $panic_triggered,
                sr.behavioral_delta    = $behavioral_delta,
                sr.sor_vulnerability   = $sor_vulnerability,

                sr.mc_median_10yr      = $mc_median_10yr,
                sr.mc_p5_10yr          = $mc_p5_10yr,
                sr.mc_p95_10yr         = $mc_p95_10yr,
                sr.mc_mean_10yr        = $mc_mean_10yr,
                sr.mc_real_median_10yr = $mc_real_median_10yr,
                sr.mc_real_p5_10yr     = $mc_real_p5_10yr,
                sr.panic_fraction      = $panic_fraction,
                sr.goal_probability    = $goal_probability,
                sr.tax_drag_cost       = $tax_drag_cost,
                sr.sor_risk_score      = $sor_risk_score,
                sr.inflation_rate_used = $inflation_rate_used,

                sr.cash_yield_benefit  = $cash_yield_benefit,
                sr.regret_score        = $regret_score
            WITH sr
            MATCH (p:Person {id: $person_id})
            MERGE (p)-[:HAS_SIMULATION]->(sr)
            WITH sr
            OPTIONAL MATCH (mr:MarketRegime {key: $scenario})
            FOREACH (r IN CASE WHEN mr IS NOT NULL THEN [mr] ELSE [] END |
                MERGE (sr)-[:USES_REGIME]->(r)
            )
            """,
            id=sim_data["id"],
            person_id=sim_data["person_id"],
            scenario=str(sim_data["scenario"]),
            created_at=str(sim_data["created_at"]),
            # Historical
            max_drawdown=float(sim_data.get("max_drawdown", 0.0)),
            real_max_drawdown=float(sim_data.get("real_max_drawdown", 0.0)),
            recovery_months=int(sim_data.get("recovery_months", 0)),
            stress_index=float(sim_data.get("stress_index", 0.0)),
            panic_triggered=bool(sim_data.get("panic_triggered", False)),
            behavioral_delta=float(sim_data.get("behavioral_delta", 0.0)),
            sor_vulnerability=float(sim_data.get("sor_vulnerability", 0.0)),
            # Monte Carlo
            mc_median_10yr=float(sim_data.get("mc_median_10yr", 0.0)),
            mc_p5_10yr=float(sim_data.get("mc_p5_10yr", 0.0)),
            mc_p95_10yr=float(sim_data.get("mc_p95_10yr", 0.0)),
            mc_mean_10yr=float(sim_data.get("mc_mean_10yr", 0.0)),
            mc_real_median_10yr=float(sim_data.get("mc_real_median_10yr", 0.0)),
            mc_real_p5_10yr=float(sim_data.get("mc_real_p5_10yr", 0.0)),
            panic_fraction=float(sim_data.get("panic_fraction", 0.0)),
            goal_probability=float(sim_data.get("goal_probability", 0.0)),
            tax_drag_cost=float(sim_data.get("tax_drag_cost", 0.0)),
            sor_risk_score=float(sim_data.get("sor_risk_score", 0.0)),
            inflation_rate_used=float(sim_data.get("inflation_rate_used", 0.03)),
            # Behavioural
            cash_yield_benefit=float(sim_data.get("cash_yield_benefit", 0.0)),
            regret_score=float(sim_data.get("regret_score", 0.0)),
        )


# ── Read helpers ──────────────────────────────────────────────────────────────

def read_subgraph(person_id: str) -> Optional[dict]:
    """
    Fetch the full person subgraph from Neo4j.

    Returns None if the person does not exist.
    """
    driver = get_driver()
    with driver.session() as session:
        result = session.run(
            """
            MATCH (p:Person {id: $id})
            OPTIONAL MATCH (p)-[:HAS_RISK_PROFILE]->(rp:RiskProfile)
            OPTIONAL MATCH (p)-[:HOLDS]->(port:Portfolio)
            OPTIONAL MATCH (p)-[:EXPERIENCED]->(le:LifeEvent)
            OPTIONAL MATCH (p)-[:MATCHES_ARCHETYPE]->(pa:PersonaArchetype)
            OPTIONAL MATCH (p)-[:LIVED_THROUGH]->(mr:MarketRegime)
            OPTIONAL MATCH (p)-[:HAS_SIMULATION]->(sr:SimulationRun)
            RETURN
                properties(p)    AS person,
                properties(rp)   AS risk_profile,
                properties(port) AS portfolio,
                collect(DISTINCT properties(le)) AS life_events,
                collect(DISTINCT pa.name)         AS archetypes,
                collect(DISTINCT mr.key)          AS lived_through_regimes,
                collect(DISTINCT properties(sr))  AS simulation_runs
            """,
            id=person_id,
        )

        record = result.single()
        if record is None or record["person"] is None:
            return None

        return {
            "person":               dict(record["person"]),
            "risk_profile":         dict(record["risk_profile"]) if record["risk_profile"] else {},
            "portfolio":            dict(record["portfolio"])    if record["portfolio"]    else {},
            "life_events":          [e for e in record["life_events"]           if e],
            "archetypes":           [a for a in record["archetypes"]            if a],
            "lived_through_regimes":[r for r in record["lived_through_regimes"] if r],
            "simulation_runs":      [s for s in record["simulation_runs"]       if s],
        }


def search_clients_by_archetype(archetype: str, limit: int = 20) -> list[dict]:
    """
    Return persons whose archetype name contains the given substring (case-insensitive).
    Each result includes a lightweight risk + portfolio summary.

    Tries the full-text index first (Neo4j 5+); falls back to a CONTAINS scan.
    """
    driver = get_driver()
    with driver.session() as session:
        # Try full-text index (Neo4j 5+)
        try:
            result = session.run(
                """
                CALL db.index.fulltext.queryNodes('archetype_ft', $query)
                YIELD node AS pa
                MATCH (p:Person)-[:MATCHES_ARCHETYPE]->(pa)
                OPTIONAL MATCH (p)-[:HAS_RISK_PROFILE]->(rp:RiskProfile)
                OPTIONAL MATCH (p)-[:HOLDS]->(port:Portfolio)
                OPTIONAL MATCH (p)-[:MATCHES_ARCHETYPE]->(all_pa:PersonaArchetype)
                RETURN
                    p.id            AS person_id,
                    p.age           AS age,
                    p.income        AS income,
                    p.net_worth     AS net_worth,
                    rp.score        AS risk_score,
                    rp.loss_aversion AS loss_aversion,
                    rp.panic_threshold AS panic_threshold,
                    port.eq_pct     AS eq_pct,
                    port.bond_pct   AS bond_pct,
                    port.alt_pct    AS alt_pct,
                    collect(DISTINCT all_pa.name) AS archetypes
                ORDER BY rp.score DESC
                LIMIT $limit
                """,
                query=archetype,
                limit=limit,
            )
            rows = [dict(r) for r in result]
            if rows:
                return rows
        except Exception:
            pass  # Full-text index not available — fall through to CONTAINS scan

        # Fallback: plain CONTAINS scan
        result = session.run(
            """
            MATCH (p:Person)-[:MATCHES_ARCHETYPE]->(pa:PersonaArchetype)
            WHERE toLower(pa.name) CONTAINS toLower($archetype)
            OPTIONAL MATCH (p)-[:HAS_RISK_PROFILE]->(rp:RiskProfile)
            OPTIONAL MATCH (p)-[:HOLDS]->(port:Portfolio)
            OPTIONAL MATCH (p)-[:MATCHES_ARCHETYPE]->(all_pa:PersonaArchetype)
            RETURN
                p.id            AS person_id,
                p.age           AS age,
                p.income        AS income,
                p.net_worth     AS net_worth,
                rp.score        AS risk_score,
                rp.loss_aversion AS loss_aversion,
                rp.panic_threshold AS panic_threshold,
                port.eq_pct     AS eq_pct,
                port.bond_pct   AS bond_pct,
                port.alt_pct    AS alt_pct,
                collect(DISTINCT all_pa.name) AS archetypes
            ORDER BY rp.score DESC
            LIMIT $limit
            """,
            archetype=archetype,
            limit=limit,
        )
        return [dict(r) for r in result]


def get_archetype_benchmark(archetype: str) -> dict:
    """
    Aggregate risk and simulation statistics across all clients who share
    the given archetype.  Returns averages, counts, and percentile bands
    so an advisor can benchmark any individual client against their cohort.
    """
    driver = get_driver()
    with driver.session() as session:
        result = session.run(
            """
            MATCH (p:Person)-[:MATCHES_ARCHETYPE]->(pa:PersonaArchetype)
            WHERE toLower(pa.name) CONTAINS toLower($archetype)
            OPTIONAL MATCH (p)-[:HAS_RISK_PROFILE]->(rp:RiskProfile)
            OPTIONAL MATCH (p)-[:HAS_SIMULATION]->(sr:SimulationRun)
            WITH
                pa.name                      AS archetype_name,
                count(DISTINCT p)            AS client_count,
                count(DISTINCT sr)           AS simulation_count,
                avg(p.age)                   AS avg_age,
                avg(rp.score)                AS avg_risk_score,
                avg(rp.loss_aversion)        AS avg_loss_aversion,
                avg(rp.panic_threshold)      AS avg_panic_threshold,
                avg(sr.max_drawdown)         AS avg_max_drawdown,
                avg(sr.mc_median_10yr)       AS avg_mc_median_10yr,
                avg(sr.mc_real_median_10yr)  AS avg_mc_real_median,
                avg(sr.goal_probability)     AS avg_goal_probability,
                avg(sr.sor_risk_score)       AS avg_sor_risk_score,
                avg(sr.regret_score)         AS avg_regret_score,
                avg(sr.panic_fraction)       AS avg_panic_fraction,
                avg(sr.stress_index)         AS avg_stress_index,
                min(sr.max_drawdown)         AS worst_drawdown,
                max(sr.mc_median_10yr)       AS best_mc_median
            RETURN *
            """,
            archetype=archetype,
        )
        record = result.single()
        if record is None:
            return {"error": f"No clients found matching archetype '{archetype}'"}

        return {
            "archetype":          record["archetype_name"],
            "client_count":       record["client_count"],
            "simulation_count":   record["simulation_count"],
            "avg_age":            record["avg_age"],
            "avg_risk_score":     record["avg_risk_score"],
            "avg_loss_aversion":  record["avg_loss_aversion"],
            "avg_panic_threshold":record["avg_panic_threshold"],
            "avg_max_drawdown":   record["avg_max_drawdown"],
            "avg_mc_median_10yr": record["avg_mc_median_10yr"],
            "avg_mc_real_median": record["avg_mc_real_median"],
            "avg_goal_probability":record["avg_goal_probability"],
            "avg_sor_risk_score": record["avg_sor_risk_score"],
            "avg_regret_score":   record["avg_regret_score"],
            "avg_panic_fraction": record["avg_panic_fraction"],
            "avg_stress_index":   record["avg_stress_index"],
            "worst_drawdown":     record["worst_drawdown"],
            "best_mc_median":     record["best_mc_median"],
        }


def get_simulation_drift(person_id: str) -> list[dict]:
    """
    Return all SimulationRun nodes for a person in chronological order.

    Each entry contains the key risk metrics so the caller can chart
    how the client's risk profile has drifted across successive simulations:
    e.g. improving max_drawdown as the portfolio de-risks toward retirement,
    or a worsening stress_index after a life event.
    """
    driver = get_driver()
    with driver.session() as session:
        result = session.run(
            """
            MATCH (p:Person {id: $person_id})-[:HAS_SIMULATION]->(sr:SimulationRun)
            RETURN
                sr.id               AS simulation_id,
                sr.scenario         AS scenario,
                sr.created_at       AS created_at,
                sr.max_drawdown     AS max_drawdown,
                sr.real_max_drawdown AS real_max_drawdown,
                sr.stress_index     AS stress_index,
                sr.sor_vulnerability AS sor_vulnerability,
                sr.sor_risk_score   AS sor_risk_score,
                sr.mc_median_10yr   AS mc_median_10yr,
                sr.mc_real_median_10yr AS mc_real_median_10yr,
                sr.mc_p5_10yr       AS mc_p5_10yr,
                sr.goal_probability AS goal_probability,
                sr.panic_triggered  AS panic_triggered,
                sr.panic_fraction   AS panic_fraction,
                sr.regret_score     AS regret_score,
                sr.behavioral_delta AS behavioral_delta,
                sr.tax_drag_cost    AS tax_drag_cost,
                sr.recovery_months  AS recovery_months
            ORDER BY sr.created_at ASC
            """,
            person_id=person_id,
        )
        return [dict(r) for r in result]


def list_clients(limit: int = 50, offset: int = 0) -> list[dict]:
    """
    Return a paginated list of all Person nodes with their latest simulation run.
    Used by the dashboard to display the full client book.
    """
    driver = get_driver()
    with driver.session() as session:
        result = session.run(
            """
            MATCH (p:Person)
            OPTIONAL MATCH (p)-[:HAS_RISK_PROFILE]->(rp:RiskProfile)
            OPTIONAL MATCH (p)-[:HOLDS]->(port:Portfolio)
            OPTIONAL MATCH (p)-[:MATCHES_ARCHETYPE]->(pa:PersonaArchetype)
            OPTIONAL MATCH (p)-[:HAS_SIMULATION]->(sr:SimulationRun)
            WITH p, rp, port,
                 collect(DISTINCT pa.name)        AS archetypes,
                 collect(DISTINCT properties(sr)) AS sim_runs
            RETURN
                p.id            AS person_id,
                p.name          AS name,
                p.age           AS age,
                p.income        AS income,
                p.net_worth     AS net_worth,
                p.goal_amount   AS goal_amount,
                p.goal_years    AS goal_years,
                rp.score        AS risk_score,
                rp.loss_aversion AS loss_aversion,
                rp.panic_threshold AS panic_threshold,
                port.eq_pct     AS eq_pct,
                port.bond_pct   AS bond_pct,
                port.alt_pct    AS alt_pct,
                port.monthly_contribution AS monthly_contribution,
                archetypes,
                sim_runs
            ORDER BY p.id
            SKIP $offset
            LIMIT $limit
            """,
            offset=offset,
            limit=limit,
        )

        clients = []
        for row in result:
            raw_sims = [s for s in (row.get("sim_runs") or []) if s]
            # Sort by created_at descending (ISO strings sort lexicographically)
            sorted_sims = sorted(raw_sims, key=lambda s: s.get("created_at", ""), reverse=True)
            latest_sim = sorted_sims[0] if sorted_sims else {}

            clients.append({
                "person_id":    row["person_id"],
                "name":         row["name"],
                "age":          row["age"],
                "income":       row["income"],
                "net_worth":    row["net_worth"],
                "goal_amount":  row["goal_amount"],
                "goal_years":   row["goal_years"],
                "risk_score":   row["risk_score"],
                "loss_aversion": row["loss_aversion"],
                "panic_threshold": row["panic_threshold"],
                "eq_pct":       row["eq_pct"],
                "bond_pct":     row["bond_pct"],
                "alt_pct":      row["alt_pct"],
                "monthly_contribution": row["monthly_contribution"],
                "archetypes":   [a for a in (row["archetypes"] or []) if a],
                "simulation_count": len(sorted_sims),
                "latest_sim":   latest_sim,
            })

        return clients
