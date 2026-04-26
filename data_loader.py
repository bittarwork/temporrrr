"""
Load HW-style Excel files from /data.
Creates small sample files if missing so the app runs out of the box.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"


def _canon(name: str) -> str:
    """Map spreadsheet headers to snake_case field names we expect."""
    s = str(name).strip().lower().replace(" ", "_")
    fixes = {
        "userid": "user_id",
        "user_id": "user_id",
        "productid": "product_id",
        "prod_id": "product_id",
        "item_id": "product_id",
        "cat": "category",
        "region": "location",
        "city": "location",
        "country": "location",
    }
    return fixes.get(s, s)


def _normalize_users(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: _canon(c) for c in df.columns})
    for need in ["user_id", "age"]:
        if need not in df.columns:
            raise ValueError(f"users.xlsx must include a '{need}' column (found: {list(df.columns)})")
    if "location" not in df.columns:
        df["location"] = "Unknown"
    return df


def _normalize_products(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: _canon(c) for c in df.columns})
    for need in ["product_id", "category", "price"]:
        if need not in df.columns:
            raise ValueError(
                f"products.xlsx must include '{need}' (found: {list(df.columns)})"
            )
    return df


def _normalize_ratings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: _canon(c) for c in df.columns})
    for need in ["user_id", "product_id", "rating"]:
        if need not in df.columns:
            raise ValueError(
                f"ratings.xlsx must include '{need}' (found: {list(df.columns)})"
            )
    return df


def _normalize_behavior(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: _canon(c) for c in df.columns})
    for need in ["user_id", "product_id"]:
        if need not in df.columns:
            raise ValueError(
                f"behavior.xlsx must include '{need}' (found: {list(df.columns)})"
            )
    for c in ["viewed", "clicked", "purchased"]:
        if c not in df.columns:
            df[c] = 0
    return df


def _write_sample_excel() -> None:
    """Build tiny synthetic datasets matching the assignment schema."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    n_users, n_products = 48, 60

    users = pd.DataFrame(
        {
            "user_id": range(1, n_users + 1),
            "age": rng.integers(18, 70, size=n_users),
            "location": rng.choice(
                ["North", "South", "East", "West", "Central"], size=n_users
            ),
        }
    )
    products = pd.DataFrame(
        {
            "product_id": range(1, n_products + 1),
            "category": rng.choice(
                ["Electronics", "Books", "Home", "Fashion", "Sports"], size=n_products
            ),
            "price": np.round(rng.uniform(5, 500, size=n_products), 2),
        }
    )

    pairs = []
    for u in users["user_id"]:
        for _ in range(rng.integers(3, 12)):
            p = int(rng.integers(1, n_products + 1))
            pairs.append((u, p))
    pairs = list({tuple(x) for x in pairs})

    ratings = []
    for u, p in pairs[: min(400, len(pairs))]:
        cat = products.loc[products["product_id"] == p, "category"].values[0]
        base = {"Electronics": 4, "Books": 3, "Home": 3, "Fashion": 3, "Sports": 3}[cat]
        r = int(np.clip(base + rng.integers(-2, 2), 1, 5))
        ratings.append({"user_id": u, "product_id": p, "rating": r})

    ratings_df = pd.DataFrame(ratings)

    behavior = []
    for _, row in ratings_df.iterrows():
        u, p = int(row["user_id"]), int(row["product_id"])
        behavior.append(
            {
                "user_id": u,
                "product_id": p,
                "viewed": int(rng.integers(0, 6)),
                "clicked": int(rng.integers(0, 4)),
                "purchased": int(rng.choice([0, 0, 0, 1])),
            }
        )
    extra = 120
    for _ in range(extra):
        u = int(rng.integers(1, n_users + 1))
        p = int(rng.integers(1, n_products + 1))
        behavior.append(
            {
                "user_id": u,
                "product_id": p,
                "viewed": int(rng.integers(0, 8)),
                "clicked": int(rng.integers(0, 5)),
                "purchased": int(rng.choice([0, 1])),
            }
        )
    behavior_df = pd.DataFrame(behavior).drop_duplicates(
        subset=["user_id", "product_id"]
    )

    users.to_excel(DATA_DIR / "users.xlsx", index=False)
    products.to_excel(DATA_DIR / "products.xlsx", index=False)
    ratings_df.to_excel(DATA_DIR / "ratings.xlsx", index=False)
    behavior_df.to_excel(DATA_DIR / "behavior.xlsx", index=False)


def ensure_data_files() -> None:
    needed = ["users.xlsx", "products.xlsx", "ratings.xlsx", "behavior.xlsx"]
    if not all((DATA_DIR / name).exists() for name in needed):
        _write_sample_excel()


def load_all() -> dict[str, pd.DataFrame]:
    ensure_data_files()
    users = _normalize_users(pd.read_excel(DATA_DIR / "users.xlsx"))
    products = _normalize_products(pd.read_excel(DATA_DIR / "products.xlsx"))
    ratings = _normalize_ratings(pd.read_excel(DATA_DIR / "ratings.xlsx"))
    behavior = _normalize_behavior(pd.read_excel(DATA_DIR / "behavior.xlsx"))

    # One consistent policy for missing values (assignment: document one policy)
    users["user_id"] = pd.to_numeric(users["user_id"], errors="coerce").astype("Int64")
    users = users.dropna(subset=["user_id"])
    users["user_id"] = users["user_id"].astype(int)
    users["age"] = pd.to_numeric(users["age"], errors="coerce")
    users["age"] = users["age"].fillna(int(users["age"].median()))
    users["location"] = users["location"].fillna("Unknown")

    products["product_id"] = pd.to_numeric(products["product_id"], errors="coerce").astype("Int64")
    products = products.dropna(subset=["product_id"])
    products["product_id"] = products["product_id"].astype(int)
    products["price"] = pd.to_numeric(products["price"], errors="coerce")
    products["price"] = products["price"].fillna(products["price"].median())
    products["category"] = products["category"].fillna("General")

    ratings["user_id"] = pd.to_numeric(ratings["user_id"], errors="coerce").astype("Int64")
    ratings["product_id"] = pd.to_numeric(ratings["product_id"], errors="coerce").astype("Int64")
    ratings = ratings.dropna(subset=["user_id", "product_id"])
    ratings["user_id"] = ratings["user_id"].astype(int)
    ratings["product_id"] = ratings["product_id"].astype(int)
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
    ratings = ratings.dropna(subset=["rating"])
    ratings["rating"] = ratings["rating"].clip(1, 5).astype(int)

    behavior["user_id"] = pd.to_numeric(behavior["user_id"], errors="coerce").astype("Int64")
    behavior["product_id"] = pd.to_numeric(behavior["product_id"], errors="coerce").astype("Int64")
    behavior = behavior.dropna(subset=["user_id", "product_id"])
    behavior["user_id"] = behavior["user_id"].astype(int)
    behavior["product_id"] = behavior["product_id"].astype(int)
    behavior = behavior.drop_duplicates(subset=["user_id", "product_id"], keep="last")

    for c in ["viewed", "clicked", "purchased"]:
        behavior[c] = pd.to_numeric(behavior[c], errors="coerce").fillna(0).astype(int)

    return {
        "users": users,
        "products": products,
        "ratings": ratings,
        "behavior": behavior,
    }
