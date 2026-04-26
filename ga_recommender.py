"""
Genetic Algorithm Recommender
==============================
Based on: "Optimization of E-Commerce Product Recommendation Algorithm
Based on User Behavior" — Ji, Chen & Xiong, 2024 (DOI 10.12694/scpe.v25i5.3072)

Core idea
---------
Each chromosome is a weight vector  W = [w_rating, w_viewed, w_clicked, w_purchased].
For a given user the score of a product is the weighted sum of the user's own signals:

    score(p) = w_rating  * (rating / 5)
             + w_viewed   * viewed
             + w_clicked  * clicked
             + w_purchased* purchased

The GA evolves W to maximise the total score of products the user has engaged with
positively (purchased or rated ≥ 4).  The best W is then used to rank all candidate
products and the top-N are returned as recommendations.

GA parameters (chosen to balance speed and quality for a live demo)
--------------------------------------------------------------------
  pop_size   = 30    (individuals per generation)
  generations= 50    (iterations)
  crossover  = 0.80  (Pc — within recommended range 0.6–0.9 from lecture notes)
  mutation   = 0.10  (Pm)
  elitism    = 1     (keep the single best individual each generation)
"""

import numpy as np
import pandas as pd


class GeneticRecommender:

    def __init__(self, ratings_df, behavior_df,
                 pop_size=30, generations=50,
                 crossover_rate=0.80, mutation_rate=0.10,
                 seed=42):
        self.ratings    = ratings_df
        self.behavior   = behavior_df
        self.pop_size   = pop_size
        self.generations = generations
        self.cr         = crossover_rate
        self.mr         = mutation_rate
        self.n_genes    = 4          # number of weight genes per chromosome
        np.random.seed(seed)

    # ------------------------------------------------------------------
    # Helper: keep weights positive and normalised (sum = 1)
    # ------------------------------------------------------------------
    def _norm(self, w):
        w = np.abs(w)
        s = w.sum()
        return w / s if s > 0 else np.ones(self.n_genes) / self.n_genes

    # ------------------------------------------------------------------
    # Build a single DataFrame of per-product signals for one user
    # ------------------------------------------------------------------
    def _user_signals(self, user_id):
        r = self.ratings[self.ratings['user_id'] == user_id][['product_id', 'rating']].copy()
        b = self.behavior[self.behavior['user_id'] == user_id][
            ['product_id', 'viewed', 'clicked', 'purchased']
        ].copy()

        if r.empty and b.empty:
            return pd.DataFrame()

        # Union of product IDs the user has any interaction with
        all_pids = pd.concat(
            [r[['product_id']], b[['product_id']]]
        ).drop_duplicates()

        df = all_pids.merge(r, on='product_id', how='left')
        df = df.merge(b, on='product_id', how='left')
        df = df.fillna(0)

        # Normalise rating to [0, 1]
        df['rating_norm'] = df['rating'] / 5.0
        return df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Score every row (product) in user_signals using chromosome W
    # ------------------------------------------------------------------
    def _score(self, W, sig):
        return (W[0] * sig['rating_norm'] +
                W[1] * sig['viewed']      +
                W[2] * sig['clicked']     +
                W[3] * sig['purchased'])

    # ------------------------------------------------------------------
    # Fitness: sum of scores on "positively engaged" products
    # ------------------------------------------------------------------
    def _fitness(self, W, sig):
        if sig.empty:
            return 0.0

        scores = self._score(W, sig)

        # Primary positive signal: purchased OR rated ≥ 4
        positive = (sig['purchased'] == 1) | (sig['rating'] >= 4)

        # Fallback: any viewed product
        if positive.sum() == 0:
            positive = sig['viewed'] == 1

        # Final fallback: use all products
        if positive.sum() == 0:
            return float(scores.sum())

        return float(scores[positive].sum())

    # ------------------------------------------------------------------
    # GA operators
    # ------------------------------------------------------------------
    def _init_pop(self):
        pop = np.random.rand(self.pop_size, self.n_genes)
        return np.array([self._norm(w) for w in pop])

    def _tournament(self, pop, fits, k=3):
        """Tournament selection — pick winner among k random candidates."""
        idx = np.random.choice(len(pop), k, replace=False)
        best = idx[np.argmax([fits[i] for i in idx])]
        return pop[best].copy()

    def _crossover(self, p1, p2):
        """Single-point crossover."""
        if np.random.rand() < self.cr:
            pt = np.random.randint(1, self.n_genes)
            child = np.concatenate([p1[:pt], p2[pt:]])
            return self._norm(child)
        return p1.copy()

    def _mutate(self, ind):
        """Gaussian mutation on a random gene."""
        if np.random.rand() < self.mr:
            i = np.random.randint(self.n_genes)
            ind[i] = max(0, ind[i] + np.random.randn() * 0.15)
            return self._norm(ind)
        return ind

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def recommend(self, user_id, top_n=8):
        """
        Run the GA for *user_id* and return top-N recommendations.

        Returns
        -------
        product_ids    : list[int]   — recommended product IDs (best first)
        best_weights   : np.ndarray  — evolved weight vector [r, v, c, p]
        fitness_history: list[float] — best fitness value per generation
        """
        sig = self._user_signals(user_id)

        # Cold-start: no data → return empty (caller adds fallback)
        if sig.empty:
            return [], self._norm(np.ones(self.n_genes)), []

        # ---- Initialise population ----
        pop = self._init_pop()
        best_W   = pop[0].copy()
        best_fit = -np.inf
        history  = []

        for _ in range(self.generations):
            fits = [self._fitness(ind, sig) for ind in pop]

            # Track global best (elitism)
            top_idx = int(np.argmax(fits))
            if fits[top_idx] > best_fit:
                best_fit = fits[top_idx]
                best_W   = pop[top_idx].copy()

            history.append(round(best_fit, 4))

            # Build next generation
            next_pop = [best_W.copy()]          # elitism: keep best
            while len(next_pop) < self.pop_size:
                p1    = self._tournament(pop, fits)
                p2    = self._tournament(pop, fits)
                child = self._crossover(p1, p2)
                child = self._mutate(child)
                next_pop.append(child)

            pop = np.array(next_pop)

        # ---- Score all candidate products with best_W ----
        scores = self._score(best_W, sig).values
        sig_copy = sig.copy()
        sig_copy['score'] = scores

        # Exclude already-purchased products (already bought → not a recommendation)
        purchased_ids = sig[sig['purchased'] == 1]['product_id'].values
        candidates = sig_copy[~sig_copy['product_id'].isin(purchased_ids)]

        # Sort and pick top-N
        top = candidates.nlargest(top_n, 'score')
        recommended_ids = top['product_id'].astype(int).tolist()

        return recommended_ids, best_W, history
