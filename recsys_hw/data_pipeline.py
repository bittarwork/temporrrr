"""Load HW__Data_S25 Excel files, merge on (user_id, product_id), apply a fixed cleaning policy."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from recsys_hw.config import DATA_DIR


def _require_file(folder: Path, name: str) -> Path:
    p = folder / name
    if not p.is_file():
        raise FileNotFoundError(
            f"Missing '{name}' in {folder}. See HW_DATA_README.txt for expected layout."
        )
    return p


def _behavior_workbook_path(base: Path) -> Path:
    """
    Official name is behavior.xlsx; also accept behavior_*.xlsx (e.g. behavior_15500.xlsx)
    when the canonical file is absent.
    """
    canonical = base / "behavior.xlsx"
    if canonical.is_file():
        return canonical
    matches = sorted(base.glob("behavior*.xlsx"))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"Missing 'behavior.xlsx' (or behavior*.xlsx) in {base}. See HW_DATA_README.txt."
    )


def load_raw_tables(data_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read the four official spreadsheets."""
    base = Path(data_dir or DATA_DIR)
    users = pd.read_excel(_require_file(base, "users.xlsx"))
    products = pd.read_excel(_require_file(base, "products.xlsx"))
    ratings = pd.read_excel(_require_file(base, "ratings.xlsx"))
    behavior = pd.read_excel(_behavior_workbook_path(base))
    return users, products, ratings, behavior


def build_interaction_frame(
    users: pd.DataFrame,
    products: pd.DataFrame,
    ratings: pd.DataFrame,
    behavior: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge all sources on user_id and product_id.

    Cleaning policy (single, consistent — document same wording in report):
    - Drop exact duplicate keys (user_id, product_id), keep last row (stable within one import).
    - Coerce numeric columns; invalid numbers become NaN then filled with 0 for behaviour flags.
    - Missing rating after merge: drop those rows (cannot supervise content / DT target).
    - Missing age: fill with median age; missing location: fill with literal 'unknown'.
    - Missing product fields: drop rows (broken catalogue link).
    """
    r = ratings.copy()
    b = behavior.copy()
    for col in ("user_id", "product_id"):
        r[col] = pd.to_numeric(r[col], errors="coerce")
        b[col] = pd.to_numeric(b[col], errors="coerce")

    r = r.dropna(subset=["user_id", "product_id"])
    b = b.dropna(subset=["user_id", "product_id"])

    r = r.drop_duplicates(subset=["user_id", "product_id"], keep="last")
    b = b.drop_duplicates(subset=["user_id", "product_id"], keep="last")

    df = r.merge(b, on=["user_id", "product_id"], how="inner", validate="one_to_one")

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"])

    for c in ("viewed", "clicked", "purchased"):
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).clip(lower=0)

    u = users.copy()
    u["user_id"] = pd.to_numeric(u["user_id"], errors="coerce")
    u = u.dropna(subset=["user_id"]).drop_duplicates(subset=["user_id"], keep="last")
    u["age"] = pd.to_numeric(u["age"], errors="coerce")
    median_age = float(np.nanmedian(u["age"].to_numpy())) if u["age"].notna().any() else 30.0
    u["age"] = u["age"].fillna(median_age)
    if "location" not in u.columns:
        u["location"] = "unknown"
    u["location"] = u["location"].fillna("unknown").astype(str)

    p = products.copy()
    p["product_id"] = pd.to_numeric(p["product_id"], errors="coerce")
    p = p.dropna(subset=["product_id"]).drop_duplicates(subset=["product_id"], keep="last")
    p["price"] = pd.to_numeric(p["price"], errors="coerce")
    p["category"] = p.get("category", pd.Series(index=p.index, dtype=object)).astype(str)
    p = p.dropna(subset=["price"])

    out = (
        df.merge(u, on="user_id", how="left", validate="many_to_one")
        .merge(p, on="product_id", how="inner", validate="many_to_one")
    )

    out["location"] = out["location"].fillna("unknown").astype(str)
    out["age"] = pd.to_numeric(out["age"], errors="coerce").fillna(median_age)

    return out.reset_index(drop=True)


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add normalized targets and simple content labels for modelling."""
    x = df.copy()
    x["rating_n"] = (x["rating"] - 1.0) / 4.0
    x["rating_n"] = x["rating_n"].clip(0.0, 1.0)
    return x
