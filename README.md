# OVERWATCHER

> A risk-tiered, intent-aware trust verification framework for autonomous AI agents in smart healthcare.

## Architecture Overview

```
Clinician Request
      │
      ▼
[1. Intent Parser]              → Extracts structured intent (spaCy NER)
      │
      ▼
[2. Identity & Device Verify]   → IAM + device trust check
      │
      ▼
[3. Risk Scoring Engine]        → Reversibility + Sensitivity + Urgency score
      │
      ▼
[4. Consistency Checker]        → Intent vs Proposed Action comparison
      │
      ▼
[5. Behavioral Trust Memory]    → Per-agent trust score (persistent)
      │
      ▼
[6. Transformer Behavior Analyzer] → Sequence anomaly detection
      │
      ▼
[7. Dynamic Policy Engine]      → Allow / Step-up / Human Approval / Block
      │
      ▼
[8. Audit Logger]               → Immutable, hash-chained audit log
```

## Tech Stack

- **Language**: Python 3.11+
- **NLP**: spaCy + custom clinical NER
- **Embeddings**: sentence-transformers (ClinicalBERT, encoder only — no generation)
- **ML/Anomaly**: PyTorch encoder OR hmmlearn
- **Auth**: authlib + python-jose + ldap3
- **Crypto**: PyCA cryptography + hashlib
- **Data**: Pydantic + SQLAlchemy + NumPy
- **Storage**: PostgreSQL + Redis
- **API**: FastAPI + Uvicorn
- **Logging**: structlog
- **Infra**: Docker + Docker Compose

## Modules

| Module | Path |
|---|---|
| Intent Parser | `overwatcher/modules/intent_parser/` |
| Identity & Device | `overwatcher/modules/identity/` |
| Risk Scoring | `overwatcher/modules/risk_scoring/` |
| Consistency Checker | `overwatcher/modules/consistency/` |
| Trust Memory | `overwatcher/modules/trust_memory/` |
| Behavior Analyzer | `overwatcher/modules/behavior_analyzer/` |
| Policy Engine | `overwatcher/modules/policy_engine/` |
| Audit Logger | `overwatcher/modules/audit_logger/` |

## Quick Start

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download spaCy model
python -m spacy download en_core_web_trf

# 4. Set up environment variables
cp .env.example .env
# Edit .env with your DB credentials

# 5. Start infrastructure
docker-compose up -d

# 6. Run database migrations
alembic upgrade head

# 7. Start the API server
uvicorn overwatcher.api.main:app --reload --port 8000
```

## No LLM Policy

This framework does **not** use any generative LLM (no OpenAI, Anthropic, or hosted inference APIs).
All models run **locally and offline** for HIPAA compliance.
Transformers are used as **discriminative encoders only** — no text generation.
