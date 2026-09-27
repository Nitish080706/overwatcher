"""
Module 6: Behavior Analyzer
=============================
Detects sequence-level anomalies in agent behavior.
Flags actions that are individually plausible but collectively suspicious.

Backend: HMM (Hidden Markov Model) — default, fully classical, no neural network
  - Trains a Categorical HMM on normal action sequences
  - Anomaly score = negative log-likelihood of observed sequence
  - Uses hmmlearn library

Output: AnomalyResult with anomaly_score in [0, 1].
"""

from __future__ import annotations

import numpy as np
from hmmlearn.hmm import CategoricalHMM

from overwatcher.config import get_settings
from overwatcher.models import AnomalyResult, OperationType, ResourceType

settings = get_settings()

# ---------------------------------------------------------------------------
# Action token vocabulary
# ---------------------------------------------------------------------------
ACTION_VOCAB: dict[tuple[OperationType, ResourceType], int] = {}
_token = 0
for op in OperationType:
    for res in ResourceType:
        ACTION_VOCAB[(op, res)] = _token
        _token += 1

VOCAB_SIZE = len(ACTION_VOCAB)
UNKNOWN_TOKEN = VOCAB_SIZE


def encode_action(operation: OperationType, resource_type: ResourceType) -> int:
    """Convert an (operation, resource_type) pair to an integer token."""
    return ACTION_VOCAB.get((operation, resource_type), UNKNOWN_TOKEN)


# ---------------------------------------------------------------------------
# HMM Backend
# ---------------------------------------------------------------------------

class HMMBehaviorAnalyzer:
    """HMM-based sequence anomaly detector. Fully classical — no neural networks."""

    def __init__(self, n_components: int = 4) -> None:
        self._n_components = n_components
        self._model: CategoricalHMM | None = None
        self._log_prob_min: float = -100.0
        self._log_prob_max: float = 0.0
        self._trained = False

    def fit(self, sequences: list[list[int]]) -> None:
        """Train the HMM on normal action sequences (each must be >= 3 tokens)."""
        sequences = [s for s in sequences if len(s) >= 3]
        if not sequences:
            return

        X = np.concatenate([np.array(seq).reshape(-1, 1) for seq in sequences])
        lengths = [len(seq) for seq in sequences]
        n_features = VOCAB_SIZE + 1  # +1 for UNKNOWN_TOKEN

        model = CategoricalHMM(
            n_components=self._n_components,
            n_iter=200,
            tol=1e-4,
            random_state=42,
        )
        # Must set n_features BEFORE fit() so emission matrix is correct size
        model.n_features = n_features

        try:
            model.fit(X, lengths)

            # Validate transition matrix rows sum to 1
            row_sums = model.transmat_.sum(axis=1)
            if not np.allclose(row_sums, 1.0, atol=0.05):
                return  # degenerate matrix — skip

            self._model = model
            self._trained = True

            # Calibrate scoring range on training sequences
            log_probs = []
            for seq in sequences:
                try:
                    lp = model.score(np.array(seq).reshape(-1, 1))
                    log_probs.append(lp)
                except Exception:
                    continue

            if log_probs:
                self._log_prob_min = min(log_probs)
                self._log_prob_max = max(log_probs)

        except Exception:
            self._trained = False

    def score_sequence(self, token_sequence: list[int]) -> float:
        """
        Returns anomaly score in [0, 1].
        Returns 0.0 (no anomaly) if model is untrained or sequence too short.
        """
        if not self._trained or self._model is None or len(token_sequence) < 2:
            return 0.0
        try:
            X = np.array(token_sequence).reshape(-1, 1)
            log_prob = self._model.score(X)
        except Exception:
            return 0.0

        range_ = self._log_prob_max - self._log_prob_min
        if range_ == 0:
            return 0.0
        normalized = (log_prob - self._log_prob_min) / range_
        return round(1.0 - max(0.0, min(1.0, normalized)), 4)


# ---------------------------------------------------------------------------
# Behavior Analyzer — main interface
# ---------------------------------------------------------------------------

class BehaviorAnalyzer:
    """Public interface. Manages HMM backend and per-session sliding windows."""

    def __init__(self) -> None:
        self._window = settings.behavior_sequence_window
        self._backend = settings.behavior_analyzer_backend
        self._hmm = HMMBehaviorAnalyzer()
        self._session_windows: dict[str, list[int]] = {}

    def load_or_train(self, training_sequences: list[list[int]] | None = None) -> None:
        """Train the backend model. Called once at startup."""
        if training_sequences is None:
            training_sequences = self._generate_synthetic_normal()
        if self._backend == "hmm":
            self._hmm.fit(training_sequences)

    def analyze(
        self,
        session_id: str,
        operation: OperationType,
        resource_type: ResourceType,
    ) -> AnomalyResult:
        """Add the current action to session window and score the sequence."""
        token = encode_action(operation, resource_type)
        window = self._session_windows.get(session_id, [])
        window.append(token)
        if len(window) > self._window:
            window = window[-self._window:]
        self._session_windows[session_id] = window

        anomaly_score = (
            self._hmm.score_sequence(window)
            if self._backend == "hmm"
            else 0.0
        )
        return AnomalyResult(
            anomaly_score=anomaly_score,
            is_anomalous=anomaly_score > settings.threshold_stepup_anomaly,
            sequence_length=len(window),
        )

    def clear_session(self, session_id: str) -> None:
        self._session_windows.pop(session_id, None)

    @staticmethod
    def _generate_synthetic_normal() -> list[list[int]]:
        """
        Synthetic normal clinical action sequences for HMM training.
        All sequences are >= 3 tokens and cover diverse clinical workflows.
        Replace with real hospital action logs in production.
        """
        def tok(op: OperationType, res: ResourceType) -> int:
            return encode_action(op, res)

        R = OperationType.READ
        W = OperationType.WRITE
        S = OperationType.SEND
        U = OperationType.UPDATE
        C = OperationType.CREATE

        LAB = ResourceType.LAB_REPORT
        RX  = ResourceType.PRESCRIPTION
        MRI = ResourceType.MRI
        REC = ResourceType.PATIENT_RECORD
        MSG = ResourceType.MESSAGE
        APT = ResourceType.APPOINTMENT

        base_patterns = [
            # Doctor: read record → read lab → send message
            [tok(R,REC), tok(R,LAB), tok(S,MSG)],
            # Doctor: read record → read rx → update rx
            [tok(R,REC), tok(R,RX),  tok(U,RX)],
            # Nurse: read lab → send message → read appointment
            [tok(R,LAB), tok(S,MSG), tok(R,APT)],
            # Admin: read appointment → update appointment → send message
            [tok(R,APT), tok(U,APT), tok(S,MSG)],
            # Lab tech: read lab → write lab → send message
            [tok(R,LAB), tok(W,LAB), tok(S,MSG)],
            # Doctor: read MRI → read record → send message
            [tok(R,MRI), tok(R,REC), tok(S,MSG)],
            # Create appointment → send message → read appointment
            [tok(C,APT), tok(S,MSG), tok(R,APT)],
            # Full read workflow
            [tok(R,REC), tok(R,LAB), tok(R,MRI), tok(S,MSG)],
            # Doctor: read + update prescription + send
            [tok(R,REC), tok(R,RX),  tok(U,RX),  tok(S,MSG)],
            # Full consult
            [tok(R,REC), tok(R,LAB), tok(R,MRI), tok(R,RX), tok(S,MSG)],
        ]

        # Repeat 30x to give HMM sufficient training data
        return base_patterns * 30
