"""
Financial Risk Engine — FastAPI application.

Endpoints
---------
POST /ingest                      Extract entities from raw text, persist to Neo4j
POST /simulate/{id}               Run 3 parallel simulations, write results, return narrative
GET  /person/{id}                 Fetch full person subgraph
GET  /person/{id}/drift           Time-series of all simulation runs (risk drift chart)
GET  /clients/search              Search clients by archetype substring
GET  /benchmark/{archetype}       Aggregate cohort stats for an archetype
GET  /scenarios                   List available stress-test scenarios
GET  /health                      Liveness check
"""

import asyncio
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, status
from fastapi.security import OAuth2PasswordRequestForm

from auth import Token, authenticate_user, create_access_token, get_current_user
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from extractor import extract_entities
from fund_metrics import match_funds, refresh_cache, load_cache, is_cache_stale
from graph import (
    create_indexes,
    get_archetype_benchmark,
    get_simulation_drift,
    list_clients,
    read_subgraph,
    search_clients_by_archetype,
    seed_market_regimes,
    write_person,
    write_simulation_run,
)
from narrative import generate_narrative
from questionnaire import run_chat_turn
from scenario_generator import generate_scenario
from scenarios import SCENARIOS
from settings_store import get_settings, update_settings, reset_to_defaults, DEFAULT_INTERVIEW_PROMPT
from simulations.behavioural import run_behavioural
from simulations.historical import run_historical
from simulations.monte_carlo import run_monte_carlo
from audit_store import log_action, get_log, get_log_for_person
from notes_store import get_notes, add_note, update_note, delete_note

load_dotenv()

# ── Thread-pool for CPU-bound sims and blocking I/O ───────────────────────────
_executor = ThreadPoolExecutor(max_workers=12)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    import threading
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(_executor, create_indexes)
    await loop.run_in_executor(_executor, seed_market_regimes)
    # Refresh fund metrics cache if stale (non-blocking)
    if is_cache_stale():
        threading.Thread(target=refresh_cache, daemon=True).start()
    yield
    _executor.shutdown(wait=False)


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Financial Risk Engine",
    description="Persona-aware financial risk management backend.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Shared response models needed before route registration ──────────────────

class HealthResponse(BaseModel):
    status:    str
    timestamp: str


# ── Public routes (no auth required) ─────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.post("/auth/login", response_model=Token, tags=["auth"])
async def login(form: OAuth2PasswordRequestForm = Depends()) -> Token:
    username = authenticate_user(form.username, form.password)
    if not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return Token(access_token=create_access_token(username))


# ── Protected router (all routes below require a valid JWT) ───────────────────
_router = APIRouter(dependencies=[Depends(get_current_user)])


# ── Request / Response models ──────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(
        default=[],
        description="Conversation history. Pass empty list to get the opening question.",
    )
    client_name: str = Field(default="", description="Optional client name for personalised interview")


class CustomSimRequest(BaseModel):
    prompt: str = Field(..., min_length=10, description="Natural-language description of the market scenario to simulate")


class ChatResponse(BaseModel):
    status:  str           = Field(description="'gathering' or 'complete'")
    message: str           = Field(description="AI's next conversational message")
    summary: str | None    = Field(default=None, description="Synthesised client description when status='complete'")


class IngestRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=20,
        description="Free-text description of the client's financial situation.",
        examples=[
            "I'm a 58-year-old retired teacher with $850k net worth, "
            "65k pension income, 40/50/10 portfolio, lived through 2008 and panicked."
        ],
    )


class IngestResponse(BaseModel):
    person_id: str
    entities: dict[str, Any]


class SimulationResponse(BaseModel):
    person_id:     str
    simulation_id: str
    scenario:      str

    # ── Historical replay ──────────────────────────────────────────────────────
    max_drawdown:      float = Field(description="Nominal peak-to-trough loss (negative)")
    real_max_drawdown: float = Field(description="Inflation-adjusted peak-to-trough loss")
    recovery_months:   int
    stress_index:      float = Field(description="[0, 1] composite stress score")
    panic_triggered:   bool
    behavioral_delta:  float = Field(description="Stay-invested minus panic-sell (6-month re-entry, with money-market yield)")
    sor_vulnerability: float = Field(description="Sequence-of-returns risk [0, 1]; peaks near retirement")

    # ── Monte Carlo ────────────────────────────────────────────────────────────
    mc_median_10yr:      float = Field(description="Nominal median terminal value (starts at 1.0)")
    mc_p5_10yr:          float
    mc_p95_10yr:         float
    mc_mean_10yr:        float
    mc_real_median_10yr: float = Field(description="Inflation-adjusted median terminal value")
    mc_real_p5_10yr:     float = Field(description="Inflation-adjusted 5th-percentile terminal value")
    panic_fraction:      float = Field(description="Fraction of MC paths that triggered panic threshold")
    goal_probability:    float = Field(description="Fraction of paths reaching person.goal_amount by goal_years (0 if not set)")
    tax_drag_cost:       float = Field(description="Median terminal value lost to LTCG tax drag")
    sor_risk_score:      float = Field(description="MC-based SOR risk score [0, 1]")
    inflation_rate_used: float = Field(description="CPI assumption applied to real-return calculations")

    # ── Behavioural ────────────────────────────────────────────────────────────
    worst_scenario_key: str
    cost_of_panic:      dict[str, Any]
    cash_yield_benefit: float = Field(description="Terminal value gained by earning money-market yield during panic-out period")
    regret_score:       float = Field(description="Psychological regret index [0, 1]")

    # ── Charting ───────────────────────────────────────────────────────────────
    sample_mc_paths:  list[list[float]]
    historical_path:  list[float]

    # ── Narrative + allocation + fund picks ───────────────────────────────────
    narrative:                 str
    allocation_recommendation: dict[str, Any] = Field(
        description="Recommended split across Equities, Debt, Real Estate, Commodities with plain-English reasoning"
    )
    fund_recommendations:      list[dict[str, Any]] = Field(
        default=[],
        description="Top 5 mutual funds scored 0-100 for fit with this client's risk profile"
    )
    created_at: str


class ScenarioSummary(BaseModel):
    key:             str
    name:            str
    duration_months: int


class NoteRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=5000)

class NoteUpdateRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=5000)

class GoalPlanResponse(BaseModel):
    expected_return_pct:    float
    risk_label:             str
    current_corpus:         float
    monthly_sip:            float
    goal_amount:            float
    years_to_goal:          int
    projected_corpus:       float
    gap:                    float
    gap_pct:                float
    on_track:               bool
    required_monthly_sip:   float
    additional_sip_needed:  float
    projection_by_year:     list[dict]

class RebalanceAction(BaseModel):
    asset:      str
    action:     str   # "buy" | "sell" | "hold"
    amount:     float
    pct_change: float

class RebalanceResponse(BaseModel):
    net_worth:   float
    current:     dict[str, float]
    recommended: dict[str, float]
    actions:     list[RebalanceAction]
    source:      str

class ReinterviewRequest(BaseModel):
    messages: list[ChatMessage] = Field(default=[])


# ── Protected endpoints ────────────────────────────────────────────────────────

@_router.get("/scenarios", response_model=list[ScenarioSummary], tags=["reference"])
async def list_scenarios() -> list[ScenarioSummary]:
    return [
        ScenarioSummary(
            key=key,
            name=val["name"],
            duration_months=len(val["equity"]),
        )
        for key, val in SCENARIOS.items()
    ]


@_router.post("/ingest", response_model=IngestResponse, status_code=201, tags=["data"])
async def ingest(req: IngestRequest) -> IngestResponse:
    """
    Extract structured financial entities from raw client text and persist
    the full person subgraph to Neo4j.
    """
    loop = asyncio.get_running_loop()

    try:
        entities: dict = await loop.run_in_executor(_executor, extract_entities, req.text)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    person_id = str(uuid.uuid4())

    try:
        await loop.run_in_executor(_executor, write_person, person_id, entities)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Neo4j write failed: {exc}") from exc

    log_action("ingest", person_id=person_id, details={"archetypes": entities.get("archetypes", [])})
    return IngestResponse(person_id=person_id, entities=entities)


@_router.post("/simulate/{person_id}", response_model=SimulationResponse, tags=["simulation"])
async def simulate(
    person_id: str,
    scenario: Optional[str] = Query(
        default=None,
        description=(
            "Scenario key to replay (dot_com_2000 | gfc_2008 | covid_2020 | rate_shock_2022). "
            "Defaults to the first regime the client lived through, falling back to gfc_2008."
        ),
    ),
) -> SimulationResponse:
    """
    Run three simulations in parallel (historical replay, Monte Carlo,
    behavioural model), persist results to Neo4j, and return a plain-language
    advisor narrative alongside all numeric outputs.
    """
    loop = asyncio.get_running_loop()

    # ── Load subgraph ─────────────────────────────────────────────────────────
    try:
        subgraph: Optional[dict] = await loop.run_in_executor(
            _executor, read_subgraph, person_id
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Neo4j read failed: {exc}") from exc

    if subgraph is None:
        raise HTTPException(
            status_code=404,
            detail=f"Person '{person_id}' not found. Call POST /ingest first.",
        )

    # ── Resolve scenario ──────────────────────────────────────────────────────
    if scenario is None:
        lived    = subgraph.get("lived_through_regimes", [])
        scenario = lived[0] if lived else "gfc_2008"

    if scenario not in SCENARIOS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown scenario '{scenario}'. Valid keys: {list(SCENARIOS.keys())}",
        )

    # ── Run three simulations in parallel ─────────────────────────────────────
    hist_future = loop.run_in_executor(_executor, run_historical, scenario, subgraph)
    mc_future   = loop.run_in_executor(_executor, run_monte_carlo, subgraph)
    beh_future  = loop.run_in_executor(_executor, run_behavioural, subgraph)

    try:
        hist_result, mc_result, beh_result = await asyncio.gather(
            hist_future, mc_future, beh_future
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Simulation error: {exc}") from exc

    all_results = {
        "historical":  hist_result,
        "monte_carlo": mc_result,
        "behavioural": beh_result,
        "scenario_key": scenario,
    }

    # ── Generate narrative + allocation recommendation ────────────────────────
    try:
        narrative_result: dict = await loop.run_in_executor(
            _executor, generate_narrative, subgraph, all_results
        )
        narrative_text  = narrative_result.get("narrative", "")
        allocation_rec  = narrative_result.get("allocation", {})
        fund_recs       = narrative_result.get("fund_recommendations", [])
    except Exception as exc:
        narrative_text = (
            f"Narrative generation failed: {exc}. "
            "Review the numeric results directly."
        )
        allocation_rec = {}
        fund_recs      = []

    # ── Persist SimulationRun ─────────────────────────────────────────────────
    sim_id     = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    sim_data = {
        "id":         sim_id,
        "person_id":  person_id,
        "scenario":   scenario,
        "created_at": created_at,
        # Historical
        "max_drawdown":     hist_result["max_drawdown"],
        "real_max_drawdown": hist_result.get("real_max_drawdown", hist_result["max_drawdown"]),
        "recovery_months":  hist_result["recovery_months"],
        "stress_index":     hist_result["stress_index"],
        "panic_triggered":  hist_result["panic_triggered"],
        "behavioral_delta": hist_result["behavioral_delta"],
        "sor_vulnerability": hist_result.get("sor_vulnerability", 0.0),
        # Monte Carlo
        "mc_median_10yr":      mc_result["mc_median_10yr"],
        "mc_p5_10yr":          mc_result["mc_p5_10yr"],
        "mc_p95_10yr":         mc_result["mc_p95_10yr"],
        "mc_mean_10yr":        mc_result["mc_mean_10yr"],
        "mc_real_median_10yr": mc_result.get("mc_real_median_10yr", 0.0),
        "mc_real_p5_10yr":     mc_result.get("mc_real_p5_10yr", 0.0),
        "panic_fraction":      mc_result["panic_fraction"],
        "goal_probability":    mc_result.get("goal_probability", 0.0),
        "tax_drag_cost":       mc_result.get("tax_drag_cost", 0.0),
        "sor_risk_score":      mc_result.get("sor_risk_score", 0.0),
        "inflation_rate_used": mc_result.get("inflation_rate_used", 0.03),
        # Behavioural
        "cash_yield_benefit": beh_result.get("cash_yield_benefit", 0.0),
        "regret_score":       beh_result.get("regret_score", 0.0),
    }

    log_action("simulate", person_id=person_id, details={"scenario": scenario, "sim_id": sim_id})

    try:
        await loop.run_in_executor(_executor, write_simulation_run, sim_data)
    except Exception as exc:
        print(f"[WARN] SimulationRun write failed for {sim_id}: {exc}")

    # ── Build response ────────────────────────────────────────────────────────
    return SimulationResponse(
        person_id=person_id,
        simulation_id=sim_id,
        scenario=scenario,
        # Historical
        max_drawdown=hist_result["max_drawdown"],
        real_max_drawdown=hist_result.get("real_max_drawdown", hist_result["max_drawdown"]),
        recovery_months=hist_result["recovery_months"],
        stress_index=hist_result["stress_index"],
        panic_triggered=hist_result["panic_triggered"],
        behavioral_delta=hist_result["behavioral_delta"],
        sor_vulnerability=hist_result.get("sor_vulnerability", 0.0),
        # Monte Carlo
        mc_median_10yr=mc_result["mc_median_10yr"],
        mc_p5_10yr=mc_result["mc_p5_10yr"],
        mc_p95_10yr=mc_result["mc_p95_10yr"],
        mc_mean_10yr=mc_result["mc_mean_10yr"],
        mc_real_median_10yr=mc_result.get("mc_real_median_10yr", 0.0),
        mc_real_p5_10yr=mc_result.get("mc_real_p5_10yr", 0.0),
        panic_fraction=mc_result["panic_fraction"],
        goal_probability=mc_result.get("goal_probability", 0.0),
        tax_drag_cost=mc_result.get("tax_drag_cost", 0.0),
        sor_risk_score=mc_result.get("sor_risk_score", 0.0),
        inflation_rate_used=mc_result.get("inflation_rate_used", 0.03),
        # Behavioural
        worst_scenario_key=beh_result["worst_scenario_key"],
        cost_of_panic=beh_result["cost_of_panic"],
        cash_yield_benefit=beh_result.get("cash_yield_benefit", 0.0),
        regret_score=beh_result.get("regret_score", 0.0),
        # Charting
        sample_mc_paths=mc_result["sample_paths"],
        historical_path=hist_result["path"],
        # Narrative + allocation
        narrative=narrative_text,
        allocation_recommendation=allocation_rec,
        fund_recommendations=fund_recs,
        created_at=created_at,
    )


@_router.get("/person/{person_id}", tags=["data"])
async def get_person(person_id: str) -> dict:
    """Fetch the full person subgraph from Neo4j, including all simulation runs."""
    loop = asyncio.get_running_loop()

    try:
        subgraph: Optional[dict] = await loop.run_in_executor(
            _executor, read_subgraph, person_id
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Neo4j read failed: {exc}") from exc

    if subgraph is None:
        raise HTTPException(status_code=404, detail=f"Person '{person_id}' not found.")

    return subgraph


@_router.get("/person/{person_id}/drift", tags=["data"])
async def get_drift(person_id: str) -> list[dict]:
    """
    Return all SimulationRun records for a person in chronological order.

    Use this to render a risk-drift chart showing how key metrics (stress_index,
    max_drawdown, goal_probability, sor_risk_score) have evolved across
    successive simulations — e.g. as the portfolio de-risks toward retirement
    or after a significant life event is recorded.
    """
    loop = asyncio.get_running_loop()

    try:
        drift: list[dict] = await loop.run_in_executor(
            _executor, get_simulation_drift, person_id
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Neo4j read failed: {exc}") from exc

    if not drift:
        raise HTTPException(
            status_code=404,
            detail=f"No simulation runs found for person '{person_id}'.",
        )

    return drift


@_router.get("/clients/search", tags=["data"])
async def search_clients(
    archetype: str = Query(..., description="Archetype substring to search for (e.g. 'cautious', 'accumulator')"),
    limit: int = Query(default=20, ge=1, le=100, description="Maximum results to return"),
) -> list[dict]:
    """
    Search for clients whose persona archetype contains the given substring.

    Uses the Neo4j full-text index for fast lookup; falls back to a CONTAINS
    scan on older Neo4j versions.  Returns a lightweight risk + portfolio summary
    for each matching person so the advisor can quickly compare clients.
    """
    loop = asyncio.get_running_loop()

    try:
        results: list[dict] = await loop.run_in_executor(
            _executor, search_clients_by_archetype, archetype, limit
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Neo4j search failed: {exc}") from exc

    return results


@_router.get("/benchmark/{archetype}", tags=["data"])
async def get_benchmark(archetype: str) -> dict:
    """
    Return cohort-level aggregate statistics for all clients sharing an archetype.

    Advisors use this to benchmark any individual client against their peers:
    e.g. "how does this cautious_retiree's stress index compare to the average
    cautious_retiree in our book?"  Includes average risk scores, drawdowns,
    Monte Carlo medians, goal probabilities, and SOR risk scores.
    """
    loop = asyncio.get_running_loop()

    try:
        benchmark: dict = await loop.run_in_executor(
            _executor, get_archetype_benchmark, archetype
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Neo4j benchmark failed: {exc}") from exc

    if "error" in benchmark:
        raise HTTPException(status_code=404, detail=benchmark["error"])

    return benchmark


@_router.post("/simulate-from-prompt/{person_id}", response_model=SimulationResponse, tags=["simulation"])
async def simulate_from_prompt(person_id: str, req: CustomSimRequest) -> SimulationResponse:
    """
    Generate a custom market scenario from a natural-language prompt, then run
    it as a historical replay against this client's portfolio. Monte Carlo and
    behavioural simulations run with standard parameters alongside it.
    """
    loop = asyncio.get_running_loop()

    # Load subgraph
    try:
        subgraph = await loop.run_in_executor(_executor, read_subgraph, person_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Neo4j read failed: {exc}") from exc
    if subgraph is None:
        raise HTTPException(status_code=404, detail=f"Person '{person_id}' not found.")

    # Generate custom scenario
    try:
        custom_scenario = await loop.run_in_executor(_executor, generate_scenario, req.prompt)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Scenario generation failed: {exc}") from exc

    # Inject custom scenario into a temp key so run_historical can use it
    from scenarios import SCENARIOS as _SCENARIOS
    _temp_key = "__custom__"
    _SCENARIOS[_temp_key] = custom_scenario

    try:
        hist_future = loop.run_in_executor(_executor, run_historical, _temp_key, subgraph)
        mc_future   = loop.run_in_executor(_executor, run_monte_carlo, subgraph)
        beh_future  = loop.run_in_executor(_executor, run_behavioural, subgraph)
        hist_result, mc_result, beh_result = await asyncio.gather(hist_future, mc_future, beh_future)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Simulation error: {exc}") from exc
    finally:
        _SCENARIOS.pop(_temp_key, None)

    all_results = {"historical": hist_result, "monte_carlo": mc_result, "behavioural": beh_result, "scenario_key": _temp_key}

    try:
        narrative_result = await loop.run_in_executor(_executor, generate_narrative, subgraph, all_results)
        narrative_text   = narrative_result.get("narrative", "")
        allocation_rec   = narrative_result.get("allocation", {})
        fund_recs        = narrative_result.get("fund_recommendations", [])
    except Exception as exc:
        narrative_text = f"Narrative generation failed: {exc}."
        allocation_rec = {}
        fund_recs      = []

    sim_id     = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    scenario_label = f"custom: {custom_scenario['name'][:40]}"

    sim_data = {
        "id": sim_id, "person_id": person_id, "scenario": scenario_label,
        "created_at": created_at,
        "max_drawdown":      hist_result["max_drawdown"],
        "real_max_drawdown": hist_result.get("real_max_drawdown", hist_result["max_drawdown"]),
        "recovery_months":   hist_result["recovery_months"],
        "stress_index":      hist_result["stress_index"],
        "panic_triggered":   hist_result["panic_triggered"],
        "behavioral_delta":  hist_result["behavioral_delta"],
        "sor_vulnerability": hist_result.get("sor_vulnerability", 0.0),
        "mc_median_10yr":      mc_result["mc_median_10yr"],
        "mc_p5_10yr":          mc_result["mc_p5_10yr"],
        "mc_p95_10yr":         mc_result["mc_p95_10yr"],
        "mc_mean_10yr":        mc_result["mc_mean_10yr"],
        "mc_real_median_10yr": mc_result.get("mc_real_median_10yr", 0.0),
        "mc_real_p5_10yr":     mc_result.get("mc_real_p5_10yr", 0.0),
        "panic_fraction":      mc_result["panic_fraction"],
        "goal_probability":    mc_result.get("goal_probability", 0.0),
        "tax_drag_cost":       mc_result.get("tax_drag_cost", 0.0),
        "sor_risk_score":      mc_result.get("sor_risk_score", 0.0),
        "inflation_rate_used": mc_result.get("inflation_rate_used", 0.03),
        "cash_yield_benefit":  beh_result.get("cash_yield_benefit", 0.0),
        "regret_score":        beh_result.get("regret_score", 0.0),
    }

    try:
        await loop.run_in_executor(_executor, write_simulation_run, sim_data)
    except Exception as exc:
        print(f"[WARN] SimulationRun write failed for {sim_id}: {exc}")

    return SimulationResponse(
        person_id=person_id, simulation_id=sim_id, scenario=scenario_label,
        max_drawdown=hist_result["max_drawdown"],
        real_max_drawdown=hist_result.get("real_max_drawdown", hist_result["max_drawdown"]),
        recovery_months=hist_result["recovery_months"],
        stress_index=hist_result["stress_index"],
        panic_triggered=hist_result["panic_triggered"],
        behavioral_delta=hist_result["behavioral_delta"],
        sor_vulnerability=hist_result.get("sor_vulnerability", 0.0),
        mc_median_10yr=mc_result["mc_median_10yr"],
        mc_p5_10yr=mc_result["mc_p5_10yr"],
        mc_p95_10yr=mc_result["mc_p95_10yr"],
        mc_mean_10yr=mc_result["mc_mean_10yr"],
        mc_real_median_10yr=mc_result.get("mc_real_median_10yr", 0.0),
        mc_real_p5_10yr=mc_result.get("mc_real_p5_10yr", 0.0),
        panic_fraction=mc_result["panic_fraction"],
        goal_probability=mc_result.get("goal_probability", 0.0),
        tax_drag_cost=mc_result.get("tax_drag_cost", 0.0),
        sor_risk_score=mc_result.get("sor_risk_score", 0.0),
        inflation_rate_used=mc_result.get("inflation_rate_used", 0.03),
        worst_scenario_key=beh_result["worst_scenario_key"],
        cost_of_panic=beh_result["cost_of_panic"],
        cash_yield_benefit=beh_result.get("cash_yield_benefit", 0.0),
        regret_score=beh_result.get("regret_score", 0.0),
        sample_mc_paths=mc_result["sample_paths"],
        historical_path=hist_result["path"],
        narrative=narrative_text,
        allocation_recommendation=allocation_rec,
        fund_recommendations=fund_recs,
        created_at=created_at,
    )

@_router.post("/chat", response_model=ChatResponse, tags=["questionnaire"])
async def chat_turn(req: ChatRequest) -> ChatResponse:
    """
    Run one turn of the AI-powered client intake questionnaire.

    Send an empty messages list to receive the opening question.
    Keep appending user and assistant turns until status == 'complete',
    then pass summary to POST /ingest.
    """
    loop = asyncio.get_running_loop()

    raw_messages = [{"role": m.role, "content": m.content} for m in req.messages]

    try:
        result = await loop.run_in_executor(_executor, run_chat_turn, raw_messages, req.client_name)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chat error: {exc}") from exc

    return ChatResponse(
        status=result["status"],
        message=result["message"],
        summary=result.get("summary"),
    )


@_router.get("/clients", tags=["data"])
async def get_clients(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> list[dict]:
    """
    Return a paginated list of all ingested clients with their latest simulation summary.
    Used by the dashboard to display the full client book.
    """
    loop = asyncio.get_running_loop()

    try:
        clients = await loop.run_in_executor(_executor, list_clients, limit, offset)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Neo4j read failed: {exc}") from exc

    return clients


# ── Settings endpoints ────────────────────────────────────────────────────────

class SettingsPayload(BaseModel):
    interview_prompt: Optional[str] = Field(
        default=None,
        description="Custom system prompt for the client intake interview"
    )


@_router.get("/settings", tags=["settings"])
async def read_settings() -> dict:
    """Return current advisor-configurable settings."""
    s = get_settings()
    s["default_interview_prompt"] = DEFAULT_INTERVIEW_PROMPT
    return s


@_router.patch("/settings", tags=["settings"])
async def write_settings(payload: SettingsPayload) -> dict:
    """Update one or more settings. Only supplied fields are changed."""
    updates = {k: v for k, v in payload.model_dump().items() if v is not None}
    return update_settings(updates)


@_router.post("/settings/reset", tags=["settings"])
async def reset_settings() -> dict:
    """Reset all settings back to factory defaults."""
    return reset_to_defaults()


# ── Fund matching endpoints ───────────────────────────────────────────────────

@_router.post("/funds/refresh", tags=["funds"])
async def trigger_fund_refresh() -> dict:
    """Trigger a background refresh of the fund metrics cache."""
    loop = asyncio.get_running_loop()
    asyncio.ensure_future(loop.run_in_executor(_executor, refresh_cache))
    return {"status": "refreshing", "message": "Fund cache refresh started in background"}


@_router.get("/funds/match", tags=["funds"])
async def get_matched_funds(
    risk_score: int = Query(..., ge=0, le=100),
    panic_threshold: float = Query(default=-0.20),
    goal_years: int = Query(default=10, ge=1),
    top_n: int = Query(default=5, ge=1, le=20),
) -> list[dict]:
    """Return top-N funds matched to a client's risk profile using live metrics."""
    loop = asyncio.get_running_loop()
    try:
        funds = await loop.run_in_executor(
            _executor, match_funds, risk_score, panic_threshold, goal_years, top_n
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Fund matching failed: {exc}")
    return funds


# ── Audit log ─────────────────────────────────────────────────────────────────

@_router.get("/audit", tags=["audit"])
async def get_audit_log(
    limit:  int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> list[dict]:
    """Return the advisor audit log, newest first."""
    return get_log(limit=limit, offset=offset)


@_router.get("/person/{person_id}/audit", tags=["audit"])
async def get_person_audit(person_id: str) -> list[dict]:
    """Return all audit entries for a specific client."""
    return get_log_for_person(person_id)


# ── Advisor Notes ─────────────────────────────────────────────────────────────

@_router.get("/person/{person_id}/notes", tags=["notes"])
async def list_notes(person_id: str) -> list[dict]:
    return get_notes(person_id)


@_router.post("/person/{person_id}/notes", status_code=201, tags=["notes"])
async def create_note(person_id: str, req: NoteRequest) -> dict:
    note = add_note(person_id, req.content)
    log_action("note_added", person_id=person_id, details={"note_id": note["id"]})
    return note


@_router.patch("/person/{person_id}/notes/{note_id}", tags=["notes"])
async def edit_note(person_id: str, note_id: str, req: NoteUpdateRequest) -> dict:
    note = update_note(person_id, note_id, req.content)
    if note is None:
        raise HTTPException(status_code=404, detail="Note not found")
    log_action("note_updated", person_id=person_id, details={"note_id": note_id})
    return note


@_router.delete("/person/{person_id}/notes/{note_id}", tags=["notes"])
async def remove_note(person_id: str, note_id: str) -> dict:
    ok = delete_note(person_id, note_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Note not found")
    log_action("note_deleted", person_id=person_id, details={"note_id": note_id})
    return {"ok": True}


# ── Goal Planning Calculator ──────────────────────────────────────────────────

@_router.get("/person/{person_id}/goal-plan", response_model=GoalPlanResponse, tags=["planning"])
async def goal_plan(person_id: str) -> GoalPlanResponse:
    """
    Project whether the client will reach their financial goal given their
    current corpus, monthly SIP, investment horizon, and risk-adjusted
    expected return rate.
    """
    loop = asyncio.get_running_loop()
    try:
        subgraph = await loop.run_in_executor(_executor, read_subgraph, person_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Neo4j read failed: {exc}")
    if subgraph is None:
        raise HTTPException(status_code=404, detail=f"Person '{person_id}' not found.")

    person    = subgraph.get("person", {})
    portfolio = subgraph.get("portfolio", {})
    risk_prof = subgraph.get("risk_profile", {})

    net_worth   = float(person.get("net_worth", 0) or 0)
    goal_amount = float(person.get("goal_amount", 0) or 0)
    goal_years  = int(person.get("goal_years", 10) or 10)
    monthly_sip = float(portfolio.get("monthly_contribution", 0) or 0)
    risk_score  = int(risk_prof.get("score", 50) or 50)

    # Expected annual return based on risk score
    if risk_score <= 20:
        er, rl = 0.07, "Conservative"
    elif risk_score <= 40:
        er, rl = 0.09, "Moderate"
    elif risk_score <= 60:
        er, rl = 0.11, "Balanced"
    elif risk_score <= 80:
        er, rl = 0.13, "Growth"
    else:
        er, rl = 0.15, "Aggressive"

    # FV of lump-sum (current corpus)
    fv_corpus = net_worth * ((1 + er) ** goal_years)

    # FV of SIP (monthly, end-of-period)
    monthly_rate = er / 12
    n_months     = goal_years * 12
    if monthly_rate > 0:
        fv_sip = monthly_sip * (((1 + monthly_rate) ** n_months - 1) / monthly_rate) * (1 + monthly_rate)
    else:
        fv_sip = monthly_sip * n_months

    projected = fv_corpus + fv_sip
    gap        = goal_amount - projected if goal_amount > 0 else 0.0
    gap_pct    = (gap / goal_amount * 100) if goal_amount > 0 else 0.0
    on_track   = gap <= 0 if goal_amount > 0 else True

    # Required SIP to hit goal (if goal set)
    if goal_amount > 0 and monthly_rate > 0:
        target_from_sip  = max(0.0, goal_amount - fv_corpus)
        req_sip = target_from_sip / ((((1 + monthly_rate) ** n_months - 1) / monthly_rate) * (1 + monthly_rate))
    else:
        req_sip = 0.0

    additional_sip = max(0.0, req_sip - monthly_sip)

    # Year-by-year projection
    projection = []
    corpus = net_worth
    for yr in range(1, goal_years + 1):
        corpus = corpus * (1 + er) + monthly_sip * 12 * ((1 + er) ** 0.5)
        projection.append({"year": yr, "corpus": round(corpus)})

    log_action("goal_plan_viewed", person_id=person_id)

    return GoalPlanResponse(
        expected_return_pct=round(er * 100, 1),
        risk_label=rl,
        current_corpus=round(net_worth),
        monthly_sip=round(monthly_sip),
        goal_amount=round(goal_amount),
        years_to_goal=goal_years,
        projected_corpus=round(projected),
        gap=round(gap),
        gap_pct=round(gap_pct, 1),
        on_track=on_track,
        required_monthly_sip=round(req_sip),
        additional_sip_needed=round(additional_sip),
        projection_by_year=projection,
    )


# ── Portfolio Rebalancing ─────────────────────────────────────────────────────

@_router.get("/person/{person_id}/rebalance", response_model=RebalanceResponse, tags=["planning"])
async def rebalance(person_id: str) -> RebalanceResponse:
    """
    Compare the client's current portfolio allocation against the latest
    simulation's AI-recommended allocation, and return specific buy/sell actions.
    """
    loop = asyncio.get_running_loop()
    try:
        subgraph = await loop.run_in_executor(_executor, read_subgraph, person_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Neo4j read failed: {exc}")
    if subgraph is None:
        raise HTTPException(status_code=404, detail=f"Person '{person_id}' not found.")

    person    = subgraph.get("person", {})
    portfolio = subgraph.get("portfolio", {})
    sim_runs  = subgraph.get("simulation_runs", [])

    net_worth = float(person.get("net_worth", 0) or 0)
    curr_eq   = float(portfolio.get("eq_pct", 0.6) or 0.6)
    curr_bond = float(portfolio.get("bond_pct", 0.3) or 0.3)
    curr_alt  = float(portfolio.get("alt_pct", 0.1) or 0.1)

    # Get recommended allocation from latest simulation
    rec_eq = rec_bond = rec_alt = None
    source = "no_simulation"

    for run in reversed(sim_runs):
        alloc = run.get("allocation_recommendation") or {}
        if alloc:
            eq_pct    = (alloc.get("equities",   {}).get("pct", 0) or 0) / 100.0
            bond_pct  = (alloc.get("debt",        {}).get("pct", 0) or 0) / 100.0
            re_pct    = (alloc.get("real_estate", {}).get("pct", 0) or 0) / 100.0
            comm_pct  = (alloc.get("commodities", {}).get("pct", 0) or 0) / 100.0
            rec_eq    = eq_pct
            rec_bond  = bond_pct
            rec_alt   = re_pct + comm_pct  # combine real_estate + commodities → alternatives
            source    = "latest_simulation"
            break

    if rec_eq is None:
        # Fallback: simple rule based on risk score
        rs = int(subgraph.get("risk_profile", {}).get("score", 50) or 50)
        rec_eq   = min(0.9, max(0.1, rs / 100.0))
        rec_bond = (1 - rec_eq) * 0.7
        rec_alt  = (1 - rec_eq) * 0.3
        source   = "risk_score_fallback"

    actions = []
    for asset, curr, rec, label in [
        ("Equity",       curr_eq,   rec_eq,   "Equity"),
        ("Debt",         curr_bond, rec_bond, "Debt / Bonds"),
        ("Alternatives", curr_alt,  rec_alt,  "Alternatives"),
    ]:
        delta_pct = rec - curr
        delta_amt = delta_pct * net_worth
        if abs(delta_pct) < 0.01:
            action = "hold"
        elif delta_pct > 0:
            action = "buy"
        else:
            action = "sell"
        actions.append(RebalanceAction(
            asset=label,
            action=action,
            amount=round(abs(delta_amt)),
            pct_change=round(delta_pct * 100, 1),
        ))

    log_action("rebalance_viewed", person_id=person_id)

    return RebalanceResponse(
        net_worth=round(net_worth),
        current={
            "equity_pct":  round(curr_eq * 100, 1),
            "debt_pct":    round(curr_bond * 100, 1),
            "alt_pct":     round(curr_alt * 100, 1),
        },
        recommended={
            "equity_pct":  round(rec_eq * 100, 1),
            "debt_pct":    round(rec_bond * 100, 1),
            "alt_pct":     round(rec_alt * 100, 1),
        },
        actions=actions,
        source=source,
    )


# ── Re-Interview ──────────────────────────────────────────────────────────────

@_router.post("/person/{person_id}/reinterview", response_model=ChatResponse, tags=["questionnaire"])
async def reinterview_turn(person_id: str, req: ReinterviewRequest) -> ChatResponse:
    """
    Run one turn of a re-interview for an existing client.

    Works identically to POST /chat. When status='complete', call
    POST /person/{person_id}/update with the returned summary to patch
    the client's entities.
    """
    loop = asyncio.get_running_loop()

    # Verify person exists
    try:
        subgraph = await loop.run_in_executor(_executor, read_subgraph, person_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Neo4j read failed: {exc}")
    if subgraph is None:
        raise HTTPException(status_code=404, detail=f"Person '{person_id}' not found.")

    raw_messages = [{"role": m.role, "content": m.content} for m in req.messages]
    person_name  = (subgraph.get("person") or {}).get("name") or ""

    try:
        result = await loop.run_in_executor(_executor, run_chat_turn, raw_messages, person_name)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chat error: {exc}")

    if result["status"] == "complete":
        log_action("reinterview_complete", person_id=person_id)

    return ChatResponse(
        status=result["status"],
        message=result["message"],
        summary=result.get("summary"),
    )


class UpdateProfileRequest(BaseModel):
    text: str = Field(..., min_length=20, description="Re-interview summary text to re-extract entities from")


@_router.post("/person/{person_id}/update", tags=["data"])
async def update_person(person_id: str, req: UpdateProfileRequest) -> IngestResponse:
    """
    Re-extract entities from a re-interview summary and overwrite the
    existing person's data in Neo4j.
    """
    loop = asyncio.get_running_loop()

    try:
        subgraph = await loop.run_in_executor(_executor, read_subgraph, person_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Neo4j read failed: {exc}")
    if subgraph is None:
        raise HTTPException(status_code=404, detail=f"Person '{person_id}' not found.")

    try:
        entities = await loop.run_in_executor(_executor, extract_entities, req.text)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    try:
        await loop.run_in_executor(_executor, write_person, person_id, entities)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Neo4j write failed: {exc}")

    log_action("profile_updated", person_id=person_id, details={"via": "reinterview"})
    return IngestResponse(person_id=person_id, entities=entities)


# ── Mount protected router ────────────────────────────────────────────────────
app.include_router(_router)
