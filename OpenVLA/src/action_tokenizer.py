"""
Action tokenizer for OpenVLA.

Discretizes continuous robot actions into 256 bins and maps them to
special tokens added to the LLM vocabulary.

Reference: Kim et al., "OpenVLA" (2024), §3.2
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np


ACTION_TOKEN_PREFIX = "<ACTION_"
N_BINS_DEFAULT = 256


class ActionTokenizer:
    """Converts 7-dim continuous actions ↔ discrete vocabulary tokens."""

    def __init__(self, tokenizer, n_bins: int = N_BINS_DEFAULT):
        self.tokenizer = tokenizer
        self.n_bins = n_bins

        # Add <ACTION_0> … <ACTION_255> to the vocabulary (skip if already added)
        new_tokens = [f"{ACTION_TOKEN_PREFIX}{i}>" for i in range(n_bins)]
        added = tokenizer.add_tokens(new_tokens, special_tokens=False)
        if added > 0:
            print(f"[ActionTokenizer] Added {added} action tokens to vocabulary.")

        # Cache token IDs for fast lookup
        self._token_ids: List[int] = tokenizer.convert_tokens_to_ids(new_tokens)

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def load_stats(dataset_root: str, norm_mode: str = "minmax") -> dict:
        """Load action normalization bounds from lerobot stats.json."""
        stats_path = Path(dataset_root) / "meta" / "stats.json"
        if not stats_path.exists():
            raise FileNotFoundError(f"stats.json not found: {stats_path}")
        with open(stats_path) as f:
            stats = json.load(f)
        action_stats = stats["action"]
        if norm_mode == "minmax":
            lo = np.array(action_stats["min"], dtype=np.float32)
            hi = np.array(action_stats["max"], dtype=np.float32)
        elif norm_mode == "quantile":
            lo = np.array(action_stats["q01"], dtype=np.float32)
            hi = np.array(action_stats["q99"], dtype=np.float32)
        else:
            raise ValueError(f"Unknown norm_mode: {norm_mode}")
        return {"lo": lo, "hi": hi}

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
        """Map normalized action [-1, 1] → bin indices [0, n_bins-1]."""
        bins = np.floor((action_norm + 1.0) / 2.0 * self.n_bins).astype(int)
        return np.clip(bins, 0, self.n_bins - 1)

    def undiscretize(self, bins: np.ndarray) -> np.ndarray:
        """Map bin indices → normalized action (bin center)."""
        return (bins.astype(float) + 0.5) / self.n_bins * 2.0 - 1.0

    # ------------------------------------------------------------------
    # Token conversion
    # ------------------------------------------------------------------

    def encode(self, action_norm: np.ndarray) -> List[int]:
        """Normalized action → list of token IDs (one per dimension)."""
        bins = self.discretize(action_norm)
        return [self._token_ids[int(b)] for b in bins]

    def decode(self, token_ids: List[int]) -> np.ndarray:
        """List of token IDs → normalized action array."""
        bins = np.array([self._token_ids.index(tid) for tid in token_ids])
        return self.undiscretize(bins)

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
