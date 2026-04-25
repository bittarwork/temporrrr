"""Orchestrates data load, DT + GA pipeline, and catalogue scoring for the Flask app."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from recsys_hw import config
from recsys_hw.baselines import mse_constant_mean, mse_dt_column_only, mse_uniform_blend
from recsys_hw.data_pipeline import add_derived_columns, build_interaction_frame, load_raw_tables
from recsys_hw.features import (
    FEATURE_LABELS,
    FEATURE_LABELS_AR,
    build_train_statistics,
    composite_target,
    design_matrix_with_dt,
    inference_matrix_for_catalog,
)
from recsys_hw.ga_optimizer import ga_optimize, mse_loss


@dataclass
class OptimizationSnapshot:
    """Last successful run metrics (JSON-friendly for templates / API)."""

    seed: int
    population_size: int
    generations: int
    best_mse_ga: float
    mse_baseline_mean: float
    mse_baseline_dt_only: float
    mse_baseline_uniform: float
    weights: list[float] = field(default_factory=list)
    feature_labels: list[str] = field(default_factory=list)
    feature_labels_ar: list[str] = field(default_factory=list)
    ga_history_tail: list[dict] = field(default_factory=list)


class PreferenceEngine:
    """
    Paper-aligned pipeline: behaviour/content features → decision tree signal →
    genetic algorithm blends dimensions into a single «preference surface» for ranking.
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        self._data_dir = data_dir
        self._lock = threading.Lock()
        self._df: pd.DataFrame | None = None
        self._users: pd.DataFrame | None = None
        self._products: pd.DataFrame | None = None
        self._train: pd.DataFrame | None = None
        self._val: pd.DataFrame | None = None
        self._stats: dict | None = None
        self._dt = None
        self._enc = None
        self._weights: np.ndarray | None = None
        self._mean_rating: float = 3.0
        self._snapshot: OptimizationSnapshot | None = None
        self._last_error: str | None = None

    def last_error(self) -> str | None:
        return self._last_error

    def snapshot(self) -> OptimizationSnapshot | None:
        return self._snapshot

    def load_tables(self) -> None:
        """Load Excel bundle and build the merged interaction frame (no GA yet)."""
        users, products, ratings, behavior = load_raw_tables(self._data_dir)
        self._users = users
        self._products = products
        raw = build_interaction_frame(users, products, ratings, behavior)
        self._df = add_derived_columns(raw)
        self._last_error = None

    def has_data(self) -> bool:
        return self._df is not None and len(self._df) > 0

    def is_ready_for_reco(self) -> bool:
        return (
            self._weights is not None
            and self._stats is not None
            and self._dt is not None
            and self._enc is not None
            and self._products is not None
        )

    def user_list(self) -> list[dict[str, Any]]:
        """Basic directory for the UI."""
        if self._users is None or self._df is None:
            return []
        u = self._users.copy()
        u["user_id"] = pd.to_numeric(u["user_id"], errors="coerce")
        u = u.dropna(subset=["user_id"])
        # Count interactions per user in merged frame
        counts = self._df.groupby("user_id").size().rename("n_interactions")
        u = u.merge(counts, on="user_id", how="left")
        u["n_interactions"] = u["n_interactions"].fillna(0).astype(int)
        return u.sort_values("user_id").to_dict(orient="records")

    def run_optimization(self, seed: int | None = None) -> OptimizationSnapshot:
        """Train DT on train split, tune blend weights with GA on validation rows."""
        seed = int(seed if seed is not None else config.SEED_DEFAULT)
        with self._lock:
            self.load_tables()
            df = self._df
            assert df is not None
            df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
            n_val = max(1, int(len(df) * config.VAL_FRACTION))
            val = df.iloc[:n_val].copy()
            train = df.iloc[n_val:].copy()
            self._train = train
            self._val = val
            self._mean_rating = float(train["rating"].mean())

            stats = build_train_statistics(train)
            self._stats = stats
            X_train, X_val, y_val, self._dt, self._enc = design_matrix_with_dt(train, val, stats)
            y_train = composite_target(train)

            w, history = ga_optimize(
                X_val,
                y_val,
                population_size=config.POPULATION_SIZE,
                generations=config.GENERATIONS,
                seed=seed,
                elite_count=config.ELITE_COUNT,
                mutation_std=config.MUTATION_STD,
                crossover_rate=config.CROSSOVER_RATE,
            )
            self._weights = w

            best_mse = mse_loss(w, X_val, y_val)
            snap = OptimizationSnapshot(
                seed=seed,
                population_size=config.POPULATION_SIZE,
                generations=config.GENERATIONS,
                best_mse_ga=float(best_mse),
                mse_baseline_mean=mse_constant_mean(composite_target(train), y_val),
                mse_baseline_dt_only=mse_dt_column_only(X_val, y_val),
                mse_baseline_uniform=mse_uniform_blend(X_val, y_val),
                weights=[float(x) for x in w.tolist()],
                feature_labels=list(FEATURE_LABELS),
                feature_labels_ar=list(FEATURE_LABELS_AR),
                ga_history_tail=history[-5:] if history else [],
            )
            self._snapshot = snap
            self._last_error = None
            return snap

    def preference_profile(self) -> list[dict[str, Any]]:
        """Human-readable GA solution: each axis is one interpretable signal."""
        if not self.is_ready_for_reco() or self._snapshot is None:
            return []
        out = []
        ar_labels = self._snapshot.feature_labels_ar
        if len(ar_labels) != len(self._snapshot.feature_labels):
            ar_labels = list(FEATURE_LABELS_AR)
        for label, label_ar, wt in zip(self._snapshot.feature_labels, ar_labels, self._snapshot.weights):
            out.append(
                {
                    "label": label,
                    "label_ar": label_ar,
                    "weight": wt,
                    "weight_pct": round(100.0 * wt, 2),
                }
            )
        return sorted(out, key=lambda r: -r["weight"])

    def decision_tree_rules_text(self, *, max_depth: int = 10) -> str | None:
        """Plain-text rules from the trained decision tree (inputs: age, price, category code)."""
        if self._dt is None:
            return None
        from sklearn.tree import export_text

        names = ["age", "price", "category_code"]
        kwargs = {"feature_names": names, "decimals": 3, "spacing": 2}
        try:
            txt = export_text(self._dt, max_depth=max_depth, **kwargs)
        except TypeError:
            txt = export_text(self._dt, **kwargs)
        return txt

    def recommend_for_user(self, user_id: int, top_k: int = 10) -> list[dict[str, Any]]:
        """Rank full product catalogue for the user using the optimized blend."""
        if not self.is_ready_for_reco():
            raise RuntimeError("Run optimization first.")
        assert self._users is not None and self._products is not None
        assert self._stats is not None and self._weights is not None
        uid = int(user_id)
        urow = self._users.loc[pd.to_numeric(self._users["user_id"], errors="coerce") == uid]
        if urow.empty:
            return []
        age = float(pd.to_numeric(urow["age"], errors="coerce").iloc[0])
        if np.isnan(age):
            age = float(self._df["age"].mean())  # type: ignore[union-attr]

        p = self._products.copy()
        p["product_id"] = pd.to_numeric(p["product_id"], errors="coerce")
        p["price"] = pd.to_numeric(p["price"], errors="coerce")
        p["category"] = p.get("category", pd.Series(index=p.index, dtype=object)).astype(str)
        p = p.dropna(subset=["product_id", "price"])

        X = inference_matrix_for_catalog(
            uid, age, p, self._stats, self._dt, self._enc, self._mean_rating
        )
        scores = (X @ self._weights).astype(np.float64)
        p = p.assign(score=scores).sort_values("score", ascending=False).head(int(top_k))
        return p[["product_id", "category", "price", "score"]].to_dict(orient="records")


_engine: PreferenceEngine | None = None


def get_engine() -> PreferenceEngine:
    """Process-wide singleton (fine for local coursework demo)."""
    global _engine
    if _engine is None:
        _engine = PreferenceEngine()
    return _engine
