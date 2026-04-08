# Financial Risk Engine

A production-quality FastAPI backend for persona-aware financial risk management.
Extracts client profiles from free text via Claude, stores them in a Neo4j knowledge
graph, and runs three parallel simulations (historical replay, Monte Carlo, behavioural
panic model) to generate plain-language advisor narratives.

---

## File structure

```
risk-engine/
├── main.py                  FastAPI app, all endpoints, asyncio.gather parallel sims
├── extractor.py             Claude API → structured JSON entities
├── graph.py                 Neo4j driver singleton, MERGE helpers, subgraph read/write
├── narrative.py             Claude API → plain-language advisor suitability report
├── scenarios.py             Historical monthly return data for 4 market regimes
├── simulations/
│   ├── __init__.py
│   ├── historical.py        Replay real crash against persona portfolio
│   ├── monte_carlo.py       1 000-path 10-year projection (scipy Cholesky correlations)
│   └── behavioural.py       Panic sell-off model vs stay-invested delta
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup

### 1. Neo4j AuraDB (free tier)

Create a free instance at **https://neo4j.com/cloud/aura-free/**. After provisioning,
copy the connection URI (`neo4j+s://…`), username (`neo4j`), and generated password.

### 2. Anthropic API key

Get a key at **https://console.anthropic.com/**.

### 3. Environment variables

```bash
cp .env.example .env
# Edit .env and fill in your three secrets
```

`.env`:
```
ANTHROPIC_API_KEY=sk-ant-...
NEO4J_URI=neo4j+s://xxxxxxxx.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_aura_password
```

### 4. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 5. Run the server

```bash
uvicorn main:app --reload --port 8000
```

On first startup the app creates Neo4j uniqueness constraints and seeds the four
`MarketRegime` nodes automatically.

Interactive API docs: **http://localhost:8000/docs**

---

## Endpoints

| Method | Path                  | Description                                              |
|--------|-----------------------|----------------------------------------------------------|
| POST   | `/ingest`             | Extract entities from raw text, persist to Neo4j         |
| POST   | `/simulate/{id}`      | Run 3 parallel simulations, write results, return report |
| GET    | `/person/{id}`        | Fetch full person subgraph (all nodes + simulation runs) |
| GET    | `/scenarios`          | List available historical stress scenarios               |
| GET    | `/health`             | Liveness check                                           |

Optional query param on `/simulate/{id}`:
`?scenario=dot_com_2000 | gfc_2008 | covid_2020 | rate_shock_2022`
Defaults to the first regime the client lived through, then falls back to `gfc_2008`.

---

## curl examples

### POST /ingest

```bash
curl -s -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "text": "I am a 58-year-old retired teacher. My annual pension income is $65,000 and my total net worth is about $850,000. My current portfolio is roughly 40% equities, 50% bonds, and 10% in REITs and other alternatives. I lived through the 2008 financial crisis and watched my retirement savings fall nearly 40% — I admit I sold some funds near the bottom in a panic. I also stayed invested during the COVID crash in 2020 which helped. My main concern now is capital preservation since I plan to start drawing down in two years for living expenses. I really could not stomach losing more than 15% of my portfolio value at any point."
  }' | python -m json.tool
```

Example response:
```json
{
  "person_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "entities": {
    "person": { "age": 58, "income": 65000, "net_worth": 850000 },
    "portfolio": { "eq_pct": 0.4, "bond_pct": 0.5, "alt_pct": 0.1 },
    "risk_profile": { "score": 35, "loss_aversion": 0.82, "panic_threshold": -0.15 },
    "life_events": [
      { "type": "panic_sell", "date": "2008-10", "impact": "high" },
      { "type": "stay_invested_covid", "date": "2020-03", "impact": "medium" }
    ],
    "archetypes": ["cautious_retiree", "anxious_saver"],
    "lived_through_regimes": ["gfc_2008", "covid_2020"]
  }
}
```

### POST /simulate/{person_id}

Replace `<PERSON_ID>` with the `person_id` returned by `/ingest`:

```bash
curl -s -X POST \
  "http://localhost:8000/simulate/<PERSON_ID>?scenario=gfc_2008" \
  | python -m json.tool
```

Without a scenario query param the engine picks the first lived-through regime:

```bash
curl -s -X POST http://localhost:8000/simulate/<PERSON_ID> | python -m json.tool
```

Example response (abbreviated):
```json
{
  "person_id": "3fa85f64-...",
  "simulation_id": "a1b2c3d4-...",
  "scenario": "gfc_2008",
  "max_drawdown": -0.312,
  "recovery_months": 14,
  "stress_index": 0.85,
  "panic_triggered": true,
  "behavioral_delta": 0.094,
  "mc_median_10yr": 1.87,
  "mc_p5_10yr": 0.93,
  "mc_p95_10yr": 3.61,
  "mc_mean_10yr": 2.01,
  "panic_fraction": 0.312,
  "worst_scenario_key": "gfc_2008",
  "cost_of_panic": {
    "delay_3m":  { "stay_final": 0.98, "panic_final": 0.87, "delta": 0.11, "delta_pct": 11.0, "reentry_month": 9 },
    "delay_6m":  { "stay_final": 0.98, "panic_final": 0.89, "delta": 0.09, "delta_pct":  9.4, "reentry_month": 12 },
    "delay_12m": { "stay_final": 0.98, "panic_final": 0.92, "delta": 0.06, "delta_pct":  6.3, "reentry_month": 18 }
  },
  "narrative": "This client is a 58-year-old retired teacher ...",
  "created_at": "2024-11-15T10:32:00.000Z"
}
```

### GET /person/{person_id}

```bash
curl -s http://localhost:8000/person/<PERSON_ID> | python -m json.tool
```

### GET /scenarios

```bash
curl -s http://localhost:8000/scenarios | python -m json.tool
```

### GET /health

```bash
curl -s http://localhost:8000/health
```

---

## Neo4j graph schema

**Nodes**

| Label            | Key properties                                                              |
|------------------|-----------------------------------------------------------------------------|
| Person           | id, age, income, net_worth                                                  |
| RiskProfile      | id, score (1–100), loss_aversion (0–1), panic_threshold (negative float)   |
| Portfolio        | id, eq_pct, bond_pct, alt_pct                                               |
| LifeEvent        | id, type, date, impact                                                      |
| PersonaArchetype | name                                                                        |
| MarketRegime     | key, name                                                                   |
| SimulationRun    | id, person_id, max_drawdown, recovery_months, stress_index, mc_median_10yr, mc_p5_10yr, behavioral_delta, panic_triggered, scenario, created_at |

**Edges**

```
(Person)-[:HAS_RISK_PROFILE]->(RiskProfile)
(Person)-[:HOLDS]->(Portfolio)
(Person)-[:EXPERIENCED]->(LifeEvent)
(Person)-[:MATCHES_ARCHETYPE]->(PersonaArchetype)
(Person)-[:LIVED_THROUGH]->(MarketRegime)
(Person)-[:HAS_SIMULATION]->(SimulationRun)
(SimulationRun)-[:USES_REGIME]->(MarketRegime)
```

---

## Architecture notes

- **Parallelism** — `/simulate` dispatches historical, Monte Carlo, and behavioural
  workers simultaneously via `asyncio.gather` + `ThreadPoolExecutor`, typically
  completing all three in under 200 ms.
- **Idempotency** — every Neo4j write uses `MERGE`. Re-ingesting the same client
  description or re-running a simulation with the same ID is a safe no-op.
- **Reproducibility** — Monte Carlo uses `numpy.random.default_rng(seed=42)`.
- **Correlations** — Monte Carlo uses Cholesky-decomposed correlated asset returns
  (equity/bond −0.20 correlation; equity/alt +0.30) via `scipy.linalg.cholesky`.
