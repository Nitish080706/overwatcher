"""
OVERWATCHER FastAPI Application
=================================
Exposes the OVERWATCHER pipeline as a REST API.
The agent calls POST /verify before executing any action.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from overwatcher.config import get_settings
from overwatcher.models import (
    OverwatcherRequest,
    OverwatcherResponse,
    PolicyDecision,
)
from overwatcher.pipeline import OverwatcherPipeline

settings = get_settings()

# Pipeline is initialized once at startup (loads spaCy, embedding model, etc.)
_pipeline: OverwatcherPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    _pipeline = OverwatcherPipeline()
    yield
    # Cleanup on shutdown if needed


app = FastAPI(
    title="OVERWATCHER",
    description=(
        "Risk-tiered, intent-aware trust verification framework "
        "for autonomous AI agents in smart healthcare."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
async def health_check():
    """Liveness probe."""
    return {"status": "ok", "service": "OVERWATCHER"}


@app.post(
    "/verify",
    response_model=OverwatcherResponse,
    tags=["Pipeline"],
    summary="Verify an agent action before execution",
)
async def verify_action(request: OverwatcherRequest) -> OverwatcherResponse:
    """
    Main endpoint. The AI agent calls this BEFORE executing any action.

    - **ALLOW**: Agent may proceed immediately.
    - **STEP_UP_VERIFICATION**: Agent must re-confirm identity/context.
    - **HUMAN_APPROVAL**: A clinician must explicitly approve before execution.
    - **BLOCK**: Action is halted — do not execute.
    """
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    response = _pipeline.run(request)
    return response


@app.get(
    "/trust/{agent_id}",
    tags=["Trust Memory"],
    summary="Get current trust profile for an agent",
)
async def get_trust_profile(agent_id: str):
    """Retrieve the persistent trust score for a given agent."""
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    profile = _pipeline._trust_memory.get_trust_profile(agent_id)
    return profile
