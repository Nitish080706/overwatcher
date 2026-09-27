"""
Global application settings loaded from environment variables.
All thresholds and weights are configurable here — no hardcoding.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Application
    app_name: str = "OVERWATCHER"
    app_env: str = "development"
    app_port: int = 8000
    secret_key: str = "change-this-to-a-strong-random-secret-key"

    # Database
    database_url: str = "postgresql://overwatcher:overwatcher@localhost:5432/overwatcher_db"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Auth
    jwt_algorithm: str = "HS256"
    jwt_expiry_minutes: int = 60
    ldap_server: str = ""
    ldap_base_dn: str = ""

    # Trust Memory
    default_trust_score: float = 0.5
    trust_decay_factor: float = 0.995
    trust_reward_alpha: float = 0.05
    trust_penalty_beta: float = 0.15

    # Risk Scoring Weights
    risk_weight_reversibility: float = 0.5
    risk_weight_sensitivity: float = 0.35
    risk_weight_urgency: float = 0.15

    # Policy Engine Thresholds
    threshold_block_consistency: float = 0.9
    threshold_human_approval_risk: float = 0.85
    threshold_human_approval_trust: float = 0.4
    threshold_anomaly_block: float = 0.9
    threshold_anomaly_human: float = 0.7
    threshold_stepup_risk: float = 0.5
    threshold_stepup_anomaly: float = 0.4

    # Behavior Analyzer
    behavior_analyzer_backend: str = "hmm"  # hmm | transformer
    behavior_sequence_window: int = 10

    # Consistency Checker
    consistency_embedding_model: str = "pritamdeka/S-PubMedBert-MS-MARCO"
    consistency_similarity_threshold: float = 0.60

    # Audit Logger
    audit_log_table: str = "audit_logs"

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
