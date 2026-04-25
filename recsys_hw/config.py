"""Central defaults for reproducible runs (align with report / paper discussion)."""

import os
from pathlib import Path

# Project root = parent of this package
ROOT = Path(__file__).resolve().parent.parent


def _default_data_dir() -> Path:
    """
    Prefer explicit course layout data/HW__Data_S25; if empty, use data/ when the
    four workbooks already live at project root data/ (common local layout).
    """
    nested = ROOT / "data" / "HW__Data_S25"
    flat = ROOT / "data"
    marker = "users.xlsx"
    if (flat / marker).is_file() and not (nested / marker).is_file():
        return flat
    return nested


DEFAULT_DATA_DIR = _default_data_dir()
DATA_DIR = Path(os.environ.get("HW_DATA_DIR", DEFAULT_DATA_DIR)).resolve()

# GA hyperparameters (document in report; tune if needed)
POPULATION_SIZE = 80
GENERATIONS = 60
SEED_DEFAULT = 42
ELITE_COUNT = 4
MUTATION_STD = 0.12
CROSSOVER_RATE = 0.85

# Train/validation split for fitness (row-level, stratified by user not required for HW)
VAL_FRACTION = 0.25

# Target blend for supervised score (ratings + behaviour), used only on val rows
WEIGHT_RATING = 0.35
WEIGHT_VIEWED = 0.08
WEIGHT_CLICKED = 0.27
WEIGHT_PURCHASED = 0.30
