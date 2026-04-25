"""Simple real-valued GA on a simplex: blend engineered + DT features (paper-style pipeline)."""

from __future__ import annotations

import numpy as np

from recsys_hw import config


def _project_simplex(v: np.ndarray) -> np.ndarray:
    """Map unconstrained vector to probability simplex (sum = 1, all >= 0)."""
    u = v - np.max(v)
    e = np.exp(np.clip(u, -40.0, 40.0))
    s = e.sum(axis=-1, keepdims=True)
    s = np.maximum(s, 1e-12)
    return e / s


def mse_loss(weights: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """Mean squared error between y and linear blend w·x (weights on simplex)."""
    w = _project_simplex(weights)
    pred = X @ w
    err = pred - y
    return float(np.mean(err * err))


def ga_optimize(
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    population_size: int = config.POPULATION_SIZE,
    generations: int = config.GENERATIONS,
    seed: int = config.SEED_DEFAULT,
    elite_count: int = config.ELITE_COUNT,
    mutation_std: float = config.MUTATION_STD,
    crossover_rate: float = config.CROSSOVER_RATE,
) -> tuple[np.ndarray, list[dict]]:
    """
    Maximize negative MSE (i.e., minimize MSE) over weight vectors.

    Encoding: length = X.shape[1]; internal representation is R^d before softmax.
    Selection: keep top elite_count, refill with uniform crossover + Gaussian mutation.
    """
    rng = np.random.default_rng(seed)
    d = X_val.shape[1]
    pop = rng.normal(size=(population_size, d))
    history: list[dict] = []

    def fitness(w: np.ndarray) -> float:
        return -mse_loss(w, X_val, y_val)

    scores = np.array([fitness(ind) for ind in pop])
    best_idx = int(np.argmax(scores))
    best_w = pop[best_idx].copy()
    best_fit = float(scores[best_idx])

    for gen in range(generations):
        order = np.argsort(-scores)
        elite = pop[order[:elite_count]]

        new_pop = list(elite)
        while len(new_pop) < population_size:
            p1, p2 = elite[rng.integers(0, elite_count, size=2)]
            child = p1.copy()
            if rng.random() < crossover_rate:
                mask = rng.random(d) < 0.5
                child[mask] = p2[mask]
            child = child + rng.normal(0.0, mutation_std, size=d)
            new_pop.append(child)

        pop = np.stack(new_pop[:population_size], axis=0)
        scores = np.array([fitness(ind) for ind in pop])
        gi = int(np.argmax(scores))
        if float(scores[gi]) > best_fit:
            best_fit = float(scores[gi])
            best_w = pop[gi].copy()

        history.append(
            {
                "generation": gen,
                "best_fitness": best_fit,
                "best_mse": mse_loss(best_w, X_val, y_val),
            }
        )

    return _project_simplex(best_w), history
