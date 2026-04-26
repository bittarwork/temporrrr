import pandas as pd
import pickle
import os

DATA_DIR  = os.path.join(os.path.dirname(__file__), 'data')
PKL_DIR   = os.path.join(os.path.dirname(__file__), '.cache')
os.makedirs(PKL_DIR, exist_ok=True)

# In-memory cache so we only read files once per process
_cache = {}

# Category → visual config (emoji + gradient color)
_CATEGORY_STYLES = {
    'electronics': {'emoji': '💻', 'color': '#667eea'},
    'clothing':    {'emoji': '👕', 'color': '#f5576c'},
    'books':       {'emoji': '📚', 'color': '#f6a623'},
    'sports':      {'emoji': '⚽', 'color': '#4CAF50'},
    'home':        {'emoji': '🏠', 'color': '#30cfd0'},
    'beauty':      {'emoji': '💄', 'color': '#ff9a9e'},
    'food':        {'emoji': '🍕', 'color': '#fda085'},
    'toys':        {'emoji': '🧸', 'color': '#a1c4fd'},
    'garden':      {'emoji': '🌿', 'color': '#56ab2f'},
    'automotive':  {'emoji': '🚗', 'color': '#373B44'},
}

def _category_style(category):
    """Return emoji and color for a given category string."""
    key = str(category).lower().strip()
    for k, v in _CATEGORY_STYLES.items():
        if k in key:
            return v
    return {'emoji': '🛍️', 'color': '#5C6BC0'}


def _read(filename):
    """
    Read an Excel file, using a pickle cache for speed.
    On first load: reads xlsx → saves .pkl next to it.
    On subsequent loads: reads the much faster .pkl file.
    Cache is invalidated automatically if the xlsx file changes.
    """
    xlsx_path = os.path.join(DATA_DIR, filename)
    pkl_path  = os.path.join(PKL_DIR, filename.replace('.xlsx', '.pkl'))

    # Use pickle cache if it exists and is newer than the source xlsx
    if os.path.exists(pkl_path):
        if os.path.getmtime(pkl_path) >= os.path.getmtime(xlsx_path):
            with open(pkl_path, 'rb') as f:
                return pickle.load(f)

    # First-time read: parse Excel and save cache
    df = pd.read_excel(xlsx_path)
    with open(pkl_path, 'wb') as f:
        pickle.dump(df, f)
    return df


def load_users():
    if 'users' not in _cache:
        _cache['users'] = _read('users.xlsx')
    return _cache['users']


def load_products():
    if 'products' not in _cache:
        df = _read('products.xlsx')

        # Generate a product name if the column is missing
        if 'name' not in df.columns:
            df['name'] = df.apply(
                lambda r: f"{r['category']} – Item #{int(r['product_id'])}", axis=1
            )

        # Attach visual helpers
        df['emoji']    = df['category'].apply(lambda c: _category_style(c)['emoji'])
        df['cat_color'] = df['category'].apply(lambda c: _category_style(c)['color'])

        _cache['products'] = df
    return _cache['products']


def load_ratings():
    if 'ratings' not in _cache:
        _cache['ratings'] = _read('ratings.xlsx')
    return _cache['ratings']


def load_behavior():
    if 'behavior' not in _cache:
        for fname in ('behavior.xlsx', 'behavior_15500.xlsx'):
            path = os.path.join(DATA_DIR, fname)
            if os.path.isfile(path):
                # Use _read() so pickle cache + same logic as other sheets
                _cache['behavior'] = _read(fname)
                break
        else:
            raise FileNotFoundError(
                f"No behavior file found in {DATA_DIR}. "
                "Expected behavior.xlsx (or behavior_15500.xlsx)."
            )
    return _cache['behavior']


def product_stats(product_id, ratings_df, behavior_df):
    """Return aggregated stats for one product (avg rating, views, purchases)."""
    r = ratings_df[ratings_df['product_id'] == product_id]
    b = behavior_df[behavior_df['product_id'] == product_id]

    avg_rating    = round(float(r['rating'].mean()), 1) if not r.empty else 0.0
    rating_count  = len(r)
    view_count    = int(b['viewed'].sum())    if not b.empty else 0
    purchase_count = int(b['purchased'].sum()) if not b.empty else 0

    return {
        'avg_rating':    avg_rating,
        'rating_count':  rating_count,
        'view_count':    view_count,
        'purchase_count': purchase_count,
    }
