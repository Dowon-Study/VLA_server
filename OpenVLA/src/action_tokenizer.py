"""OpenVLA-compatible action tokenizer.

OpenVLA does not add new ``<ACTION_*>`` tokens. It reuses the least-used
tokens at the end of the base LLM vocabulary, which preserves the pretrained
action prediction head when fine-tuning on LIBERO.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np


N_BINS_DEFAULT = 256


class ActionTokenizer:
    """Converts continuous actions to OpenVLA action token IDs."""

    def __init__(
        self,
        tokenizer,
        n_bins: int = N_BINS_DEFAULT,
        min_action: float = -1.0,
        max_action: float = 1.0,
    ):
        self.tokenizer = tokenizer
        self.n_bins = n_bins
        self.min_action = min_action
        self.max_action = max_action
        self.bins = np.linspace(min_action, max_action, n_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
        self.base_vocab_size = int(tokenizer.vocab_size)
        self.action_token_begin_idx = int(self.base_vocab_size - (n_bins + 1))
        self._token_ids = list(range(self.base_vocab_size - 1, self.base_vocab_size - n_bins - 1, -1))

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_action_stats(dataset_root: str) -> Dict[str, Any]:
        import json

        stats_path = Path(dataset_root) / "meta" / "stats.json"
        if not stats_path.exists():
            raise FileNotFoundError(f"stats.json not found: {stats_path}")
        with open(stats_path) as f:
            stats = json.load(f)
        return stats["action"]

    @staticmethod
    def load_stats(dataset_root: str, norm_mode: str = "quantile") -> dict:
        """Load action normalization bounds from lerobot stats.json."""
        action_stats = ActionTokenizer._load_action_stats(dataset_root)
        if norm_mode == "minmax":
            lo = np.array(action_stats["min"], dtype=np.float32)
            hi = np.array(action_stats["max"], dtype=np.float32)
        elif norm_mode == "quantile":
            lo = np.array(action_stats.get("q01", action_stats["min"]), dtype=np.float32)
            hi = np.array(action_stats.get("q99", action_stats["max"]), dtype=np.float32)
        else:
            raise ValueError(f"Unknown norm_mode: {norm_mode}")
        return {"lo": lo, "hi": hi}

    @staticmethod
    def load_openvla_norm_stats(dataset_root: str) -> Dict[str, Any]:
        """Return stats in the format expected by OpenVLA ``predict_action``."""
        action_stats = ActionTokenizer._load_action_stats(dataset_root)
        action_min = np.array(action_stats["min"], dtype=np.float32)
        action_max = np.array(action_stats["max"], dtype=np.float32)
        q01 = np.array(action_stats.get("q01", action_stats["min"]), dtype=np.float32)
        q99 = np.array(action_stats.get("q99", action_stats["max"]), dtype=np.float32)
        mask = action_max != action_min
        return {
            "action": {
                "mean": action_stats.get("mean"),
                "std": action_stats.get("std"),
                "min": action_min.tolist(),
                "max": action_max.tolist(),
                "q01": q01.tolist(),
                "q99": q99.tolist(),
                "mask": mask.tolist(),
            }
        }

    @staticmethod
    def normalize(action: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        """Map action from [lo, hi] → [-1, 1]."""
        span = hi - lo
        span = np.where(span < 1e-8, 1.0, span)   # avoid div-by-zero
        return np.clip(2.0 * (action - lo) / span - 1.0, -1.0, 1.0)

    @staticmethod
    def denormalize(action_norm: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        """Map action from [-1, 1] → [lo, hi]."""
        return (action_norm + 1.0) / 2.0 * (hi - lo) + lo

    # ------------------------------------------------------------------
    # Discretization
    # ------------------------------------------------------------------

    def discretize(self, action_norm: np.ndarray) -> np.ndarray:
        """Map normalized action [-1, 1] to OpenVLA bin IDs [1, n_bins]."""
        action_norm = np.clip(action_norm, self.min_action, self.max_action)
        return np.digitize(action_norm, self.bins)

    # ------------------------------------------------------------------
    # Token conversion
    # ------------------------------------------------------------------

    def encode(self, action_norm: np.ndarray) -> List[int]:
        """Normalized action → list of token IDs (one per dimension)."""
        discretized_action = self.discretize(action_norm)
        return (self.base_vocab_size - discretized_action).astype(np.int64).tolist()

    def decode(self, token_ids: List[int]) -> np.ndarray:
        """List of token IDs → normalized action array."""
        return self.decode_token_ids_to_actions(np.array(token_ids))

    def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
        """OpenVLA action token IDs → normalized continuous actions."""
        discretized_actions = self.base_vocab_size - action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, 0, self.bin_centers.shape[0] - 1)
        return self.bin_centers[discretized_actions]

    def encode_full(
        self,
        action: np.ndarray,
        lo: np.ndarray,
        hi: np.ndarray,
    ) -> List[int]:
        """Raw action → token IDs (normalize then discretize)."""
        return self.encode(self.normalize(action, lo, hi))

    def decode_full(
        self,
        token_ids: List[int],
        lo: np.ndarray,
        hi: np.ndarray,
    ) -> np.ndarray:
        """Token IDs → raw action (undiscretize then denormalize)."""
        return self.denormalize(self.decode(token_ids), lo, hi)

    @property
    def token_ids(self) -> List[int]:
        return self._token_ids

    @property
    def vocab_size(self) -> int:
        return self.n_bins
