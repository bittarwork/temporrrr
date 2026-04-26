"""
Decision tree (rating prediction) + genetic algorithm list optimizer.
Inspired by: content-based candidates, GA optimizes the final recommendation list.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor


@dataclass
class GAConfig:
    list_size: int = 8
    population: int = 64
    generations: int = 30
    crossover_rate: float = 0.85
    mutation_rate: float = 0.25
    diversity_weight: float = 1.2  # bonus for category variety (reduces overspecialization)


class RecommenderEngine:
    """
    Trains a decision tree on (user, product, behavior) rows to predict rating.
    Builds candidate items, then runs a simple GA to pick a strong diverse list.
    """

    def __init__(self, frames: dict[str, pd.DataFrame], random_seed: int = 42):
        self.rng = np.random.default_rng(random_seed)
        self.users = frames["users"].copy()
        self.products = frames["products"].copy()
        self.ratings = frames["ratings"].copy()
        self.behavior = frames["behavior"].copy()

        self._enc_loc = LabelEncoder()
        self._enc_cat = LabelEncoder()
        self._train_matrix = None
        self._dt: DecisionTreeRegressor | None = None
        self._fit()

    def _fit(self) -> None:
        # Single table: ratings + user + product + behavior
        m = self.ratings.merge(self.users, on="user_id", how="left")
        m = m.merge(self.products, on="product_id", how="left")
        m = m.merge(
            self.behavior,
            on=["user_id", "product_id"],
            how="left",
        )
        m["viewed"] = m["viewed"].fillna(0).astype(int)
        m["clicked"] = m["clicked"].fillna(0).astype(int)
        m["purchased"] = m["purchased"].fillna(0).astype(int)

        m["location"] = m["location"].fillna("Unknown")
        m["category"] = m["category"].fillna("General")
        m["age"] = pd.to_numeric(m["age"], errors="coerce").fillna(
            m["age"].median()
        )

        # Fit encoders on all labels seen in users/products so inference never breaks
        loc_labels = pd.concat(
            [self.users["location"].astype(str), m["location"].astype(str)],
            ignore_index=True,
        ).fillna("Unknown")
        cat_labels = pd.concat(
            [self.products["category"].astype(str), m["category"].astype(str)],
            ignore_index=True,
        ).fillna("General")
        self._enc_loc.fit(loc_labels)
        self._enc_cat.fit(cat_labels)

        X = np.column_stack(
            [
                m["age"].values,
                self._enc_loc.transform(m["location"].astype(str)),
                self._enc_cat.transform(m["category"].astype(str)),
                m["price"].values,
                m["viewed"].values,
                m["clicked"].values,
                m["purchased"].values,
            ]
        )
        y = m["rating"].values.astype(float)
        self._train_matrix = m
        self._dt = DecisionTreeRegressor(
            max_depth=12,
            min_samples_leaf=4,
            random_state=42,
        )
        self._dt.fit(X, y)

    def _row_features(self, user_id: int, product_row: pd.Series) -> np.ndarray:
        u = self.users.loc[self.users["user_id"] == user_id].iloc[0]
        loc = str(u["location"]) if pd.notna(u["location"]) else "Unknown"
        cat = str(product_row["category"]) if pd.notna(product_row["category"]) else "General"
        age = float(u["age"]) if pd.notna(u["age"]) else float(self.users["age"].median())

        b = self.behavior[
            (self.behavior["user_id"] == user_id)
            & (self.behavior["product_id"] == int(product_row["product_id"]))
        ]
        if len(b):
            viewed = int(b.iloc[0]["viewed"])
            clicked = int(b.iloc[0]["clicked"])
            purchased = int(b.iloc[0]["purchased"])
        else:
            viewed, clicked, purchased = 0, 0, 0

        loc_i = self._enc_loc.transform([loc])[0]
        cat_i = self._enc_cat.transform([cat])[0]
        return np.array(
            [[age, loc_i, cat_i, float(product_row["price"]), viewed, clicked, purchased]]
        )

    def _predict_rating(self, user_id: int, product_row: pd.Series) -> float:
        assert self._dt is not None
        x = self._row_features(user_id, product_row)
        return float(self._dt.predict(x)[0])

    def _candidate_pool(self, user_id: int, pool_size: int = 80) -> list[int]:
        """Top predicted items plus random exploration (GA search space)."""
        rated = set(self.ratings.loc[self.ratings["user_id"] == user_id, "product_id"])
        preds = []
        for _, prow in self.products.iterrows():
            pid = int(prow["product_id"])
            pr = self._predict_rating(user_id, prow)
            preds.append((pid, pr))
        preds.sort(key=lambda x: x[1], reverse=True)

        top = [p for p, _ in preds[: pool_size // 2]]
        self.rng.shuffle(preds)
        explore = [p for p, _ in preds[: pool_size]]
        out = list(dict.fromkeys(top + explore))  # preserve order, unique
        return out[:pool_size]

    def _fitness(
        self,
        user_id: int,
        chromosome: list[int],
        product_by_id: dict[int, pd.Series],
        cfg: GAConfig,
    ) -> float:
        preds = [self._predict_rating(user_id, product_by_id[pid]) for pid in chromosome]
        cats = [str(product_by_id[pid]["category"]) for pid in chromosome]
        diversity = len(set(cats)) / max(len(chromosome), 1)
        return float(np.mean(preds) + cfg.diversity_weight * diversity)

    def _repair_unique(self, chromo: list[int], pool: list[int]) -> list[int]:
        seen = set()
        out = []
        for pid in chromo:
            if pid not in seen:
                seen.add(pid)
                out.append(pid)
        spare = [p for p in pool if p not in seen]
        self.rng.shuffle(spare)
        while len(out) < len(chromo) and spare:
            out.append(spare.pop())
        return out[: len(chromo)]

    def recommend_for_user(
        self,
        user_id: int,
        cfg: GAConfig | None = None,
    ) -> tuple[list[dict], dict]:
        cfg = cfg or GAConfig()
        pool = self._candidate_pool(user_id, pool_size=max(80, cfg.list_size * 10))
        if len(pool) < cfg.list_size:
            pool = [int(x) for x in self.products["product_id"].tolist()]
        pool = list(dict.fromkeys(pool))
        if len(pool) < cfg.list_size:
            raise ValueError("Not enough products in catalog for GA list size.")

        product_by_id = {
            int(r["product_id"]): r for _, r in self.products.iterrows()
        }

        def random_chromo() -> list[int]:
            pick = self.rng.choice(pool, size=cfg.list_size, replace=False).tolist()
            return [int(x) for x in pick]

        population = [random_chromo() for _ in range(cfg.population)]

        def score(ch: list[int]) -> float:
            return self._fitness(user_id, ch, product_by_id, cfg)

        best = max(population, key=score)
        best_score = score(best)

        for _ in range(cfg.generations):
            ranked = sorted(population, key=score, reverse=True)
            elite_n = max(2, cfg.population // 8)
            next_gen = [list(x) for x in ranked[:elite_n]]
            top_k = max(2, len(ranked) // 2)

            while len(next_gen) < cfg.population:
                if self.rng.random() < cfg.crossover_rate and len(ranked) >= 2:
                    i1, i2 = self.rng.integers(0, top_k, size=2)
                    a, b = ranked[int(i1)], ranked[int(i2)]
                    if a is b:
                        b = ranked[(int(i2) + 1) % len(ranked)]
                    cut = self.rng.integers(1, cfg.list_size)
                    child = list(a[:cut]) + list(b[cut:])
                    child = self._repair_unique(child, pool)
                else:
                    pidx = int(self.rng.integers(0, top_k))
                    child = list(ranked[pidx])

                if self.rng.random() < cfg.mutation_rate:
                    i = self.rng.integers(0, cfg.list_size)
                    child[i] = int(self.rng.choice(pool))
                    child = self._repair_unique(child, pool)

                next_gen.append(child)

            population = next_gen[: cfg.population]
            top = max(population, key=score)
            sc = score(top)
            if sc > best_score:
                best, best_score = top, sc

        rows = []
        for rank, pid in enumerate(best, start=1):
            r = product_by_id[pid]
            rows.append(
                {
                    "rank": rank,
                    "product_id": int(pid),
                    "category": str(r["category"]),
                    "price": float(r["price"]),
                    "predicted_rating": round(self._predict_rating(user_id, r), 3),
                }
            )
        meta = {
            "ga_best_fitness": round(best_score, 4),
            "pool_size": len(pool),
            "list_size": cfg.list_size,
            "generations": cfg.generations,
            "population": cfg.population,
        }
        return rows, meta
