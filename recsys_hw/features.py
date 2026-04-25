"""Feature vectors for (user, product) rows using train-only statistics (no leakage)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor


class SimpleOrdinalEncoder:
    """
    Lightweight drop-in for sklearn OrdinalEncoder.
    Maps string categories to integer codes; unknown values map to -1.
    Avoids pulling in scipy (~100 MB) as a transitive dependency.
    """

    def __init__(self, handle_unknown: str = "use_encoded_value", unknown_value: int = -1) -> None:
        self.unknown_value = unknown_value
        self._mapping: dict[str, int] = {}

    def fit(self, X) -> "SimpleOrdinalEncoder":
        # Accept both pandas DataFrames and numpy arrays
        arr = np.asarray(X)
        categories = sorted({str(v) for v in arr[:, 0]})
        self._mapping = {cat: i for i, cat in enumerate(categories)}
        return self

    def transform(self, X) -> np.ndarray:
        # Accept both pandas DataFrames and numpy arrays
        arr = np.asarray(X)
        return np.array(
            [[self._mapping.get(str(v), self.unknown_value)] for v in arr[:, 0]],
            dtype=np.float64,
        )

from recsys_hw import config


def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros_like(a, dtype=np.float64)
    m = b > 1e-9
    out[m] = a[m] / b[m]
    return out


def build_train_statistics(train: pd.DataFrame) -> dict:
    """Aggregate tables used to score validation rows without peeking at val labels."""
    pr = train.groupby("product_id", as_index=False).agg(
        pr_mean_rating=("rating", "mean"),
        pr_sum_click=("clicked", "sum"),
        pr_sum_purchase=("purchased", "sum"),
        pr_count=("rating", "size"),
    )
    pr["pr_click_rate"] = _safe_div(pr["pr_sum_click"].to_numpy(), pr["pr_count"].to_numpy())
    pr["pr_purchase_rate"] = _safe_div(pr["pr_sum_purchase"].to_numpy(), pr["pr_count"].to_numpy())

    ur = train.groupby("user_id", as_index=False).agg(ur_mean_rating=("rating", "mean"))

    cr = train.groupby("category", as_index=False).agg(cr_mean_rating=("rating", "mean"))

    price_min = float(train["price"].min())
    price_max = float(train["price"].max())
    denom = max(price_max - price_min, 1e-9)

    return {
        "product": pr,
        "user": ur,
        "category": cr,
        "price_min": price_min,
        "price_max": price_max,
        "price_denom": denom,
    }


def attach_stats(rows: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Left-join train aggregates onto rows (user_id / product_id / category)."""
    x = rows.merge(stats["product"], on="product_id", how="left").merge(
        stats["user"], on="user_id", how="left"
    )
    x = x.merge(stats["category"], on="category", how="left")

    x["pr_mean_rating"] = x["pr_mean_rating"].fillna(x["rating"].mean())
    x["pr_click_rate"] = x["pr_click_rate"].fillna(0.0)
    x["pr_purchase_rate"] = x["pr_purchase_rate"].fillna(0.0)
    x["ur_mean_rating"] = x["ur_mean_rating"].fillna(x["rating"].mean())
    x["cr_mean_rating"] = x["cr_mean_rating"].fillna(x["rating"].mean())

    inv_price = (stats["price_max"] - x["price"].to_numpy(dtype=np.float64)) / stats["price_denom"]
    x["inv_price_norm"] = np.clip(inv_price, 0.0, 1.0)
    return x


def composite_target(df: pd.DataFrame) -> np.ndarray:
    """Scalar engagement label from official columns (for fitness / baselines)."""
    r = df["rating_n"].to_numpy(dtype=np.float64)
    v = df["viewed"].to_numpy(dtype=np.float64)
    c = df["clicked"].to_numpy(dtype=np.float64)
    p = df["purchased"].to_numpy(dtype=np.float64)
    y = (
        config.WEIGHT_RATING * r
        + config.WEIGHT_VIEWED * np.clip(v, 0.0, 1.0)
        + config.WEIGHT_CLICKED * np.clip(c, 0.0, 1.0)
        + config.WEIGHT_PURCHASED * np.clip(p, 0.0, 1.0)
    )
    return np.clip(y, 0.0, 1.0)


FEATURE_LABELS = (
    "Product mean rating (norm)",
    "Product click rate",
    "Product purchase rate",
    "User mean rating (norm)",
    "Category mean rating (norm)",
    "Price preference (cheaper = higher)",
    "Decision tree behaviour score",
)

# Short Arabic labels for UI / charts (aligned with FEATURE_LABELS order).
FEATURE_LABELS_AR = (
    "متوسط تقييم المنتج",
    "نقرات المنتج",
    "مشتريات المنتج",
    "تقييم المستخدم",
    "التصنيف",
    "تفضيل السعر",
    "خرج شجرة القرار",
)


def design_matrix_with_dt(
    train: pd.DataFrame, val: pd.DataFrame, stats: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray, DecisionTreeRegressor, SimpleOrdinalEncoder]:
    """
    Build X_train, X_val (includes DT prediction column — paper alignment).

    Columns:
      product mean rating, product click rate, product purchase rate,
      user mean rating, category mean rating, inv price norm,
      DT predicted target from (age, price, category code).
    """
    enc = SimpleOrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    enc.fit(train[["category"]].astype(str))

    def base_block(part: pd.DataFrame) -> pd.DataFrame:
        z = attach_stats(part, stats)
        mat = pd.DataFrame(
            {
                "f0": z["pr_mean_rating"] / 5.0,
                "f1": z["pr_click_rate"],
                "f2": z["pr_purchase_rate"],
                "f3": z["ur_mean_rating"] / 5.0,
                "f4": z["cr_mean_rating"] / 5.0,
                "f5": z["inv_price_norm"],
            }
        )
        return mat

    Xtr_raw = np.column_stack(
        [
            train["age"].to_numpy(dtype=np.float64),
            train["price"].to_numpy(dtype=np.float64),
            enc.transform(train[["category"]].astype(str)).ravel(),
        ]
    )
    ytr = composite_target(train)
    dt = DecisionTreeRegressor(max_depth=8, min_samples_leaf=10, random_state=config.SEED_DEFAULT)
    dt.fit(Xtr_raw, ytr)

    Xva_raw = np.column_stack(
        [
            val["age"].to_numpy(dtype=np.float64),
            val["price"].to_numpy(dtype=np.float64),
            enc.transform(val[["category"]].astype(str)).ravel(),
        ]
    )

    tr = base_block(train).to_numpy(dtype=np.float64)
    va = base_block(val).to_numpy(dtype=np.float64)
    tr_dt = dt.predict(Xtr_raw).reshape(-1, 1)
    va_dt = dt.predict(Xva_raw).reshape(-1, 1)
    tr_dt = np.clip(tr_dt, 0.0, 1.0)
    va_dt = np.clip(va_dt, 0.0, 1.0)

    X_train = np.hstack([tr, tr_dt])
    X_val = np.hstack([va, va_dt])
    y_val = composite_target(val)
    return X_train, X_val, y_val, dt, enc


def inference_matrix_for_catalog(
    user_id: int,
    user_age: float,
    products: pd.DataFrame,
    stats: dict,
    dt: DecisionTreeRegressor,
    enc: OrdinalEncoder,
    mean_rating_fill: float,
) -> np.ndarray:
    """
    Score every catalogue row for one user (cold interaction row: behaviour zeros).

    Used by the web app to rank recommendations after GA preference weights are known.
    """
    n = len(products)
    rows = pd.DataFrame(
        {
            "user_id": np.full(n, user_id, dtype=np.int64),
            "product_id": products["product_id"].to_numpy(),
            "age": np.full(n, user_age, dtype=np.float64),
            "category": products["category"].astype(str).to_numpy(),
            "price": products["price"].to_numpy(dtype=np.float64),
            "rating": np.full(n, mean_rating_fill, dtype=np.float64),
            "viewed": np.zeros(n, dtype=np.float64),
            "clicked": np.zeros(n, dtype=np.float64),
            "purchased": np.zeros(n, dtype=np.float64),
        }
    )
    rows["rating_n"] = (rows["rating"] - 1.0) / 4.0
    rows["rating_n"] = rows["rating_n"].clip(0.0, 1.0)

    z = attach_stats(rows, stats)
    base = pd.DataFrame(
        {
            "f0": z["pr_mean_rating"] / 5.0,
            "f1": z["pr_click_rate"],
            "f2": z["pr_purchase_rate"],
            "f3": z["ur_mean_rating"] / 5.0,
            "f4": z["cr_mean_rating"] / 5.0,
            "f5": z["inv_price_norm"],
        }
    ).to_numpy(dtype=np.float64)

    X_raw = np.column_stack(
        [
            rows["age"].to_numpy(dtype=np.float64),
            rows["price"].to_numpy(dtype=np.float64),
            enc.transform(rows[["category"]].astype(str)).ravel(),
        ]
    )
    dt_col = np.clip(dt.predict(X_raw), 0.0, 1.0).reshape(-1, 1)
    return np.hstack([base, dt_col])
