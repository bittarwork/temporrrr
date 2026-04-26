"""
AI-Powered E-Commerce Store — Flask Application
================================================
Uses a Genetic Algorithm to generate personalised product recommendations
based on user behaviour (views, clicks, purchases) and explicit ratings.
"""

from flask import (Flask, render_template, request,
                   session, redirect, url_for, jsonify)
from data_loader import (load_users, load_products,
                         load_ratings, load_behavior, product_stats)
from ga_recommender import GeneticRecommender

app = Flask(__name__)
app.secret_key = 'ia-smartshop-2024'

PRODUCTS_PER_PAGE = 12


# -----------------------------------------------------------------------
# Utility: enrich a list of product dicts with rating/behaviour stats
# -----------------------------------------------------------------------
def _enrich_products(product_rows, ratings_df, behavior_df):
    enriched = []
    for p in product_rows:
        stats = product_stats(p['product_id'], ratings_df, behavior_df)
        enriched.append({**p, **stats})
    return enriched


# -----------------------------------------------------------------------
# Home — product listing with category filter and pagination
# -----------------------------------------------------------------------
@app.route('/')
def index():
    products_df  = load_products()
    users_df     = load_users()
    ratings_df   = load_ratings()
    behavior_df  = load_behavior()

    # Filters
    category = request.args.get('category', '')
    search   = request.args.get('search', '').strip()
    page     = request.args.get('page', 1, type=int)

    filtered = products_df.copy()
    if category:
        filtered = filtered[filtered['category'] == category]
    if search:
        filtered = filtered[
            filtered['name'].str.contains(search, case=False, na=False)
        ]

    # Pagination
    total       = len(filtered)
    total_pages = max(1, (total + PRODUCTS_PER_PAGE - 1) // PRODUCTS_PER_PAGE)
    page        = max(1, min(page, total_pages))
    slice_start = (page - 1) * PRODUCTS_PER_PAGE
    page_rows   = filtered.iloc[slice_start: slice_start + PRODUCTS_PER_PAGE]

    products_list = _enrich_products(page_rows.to_dict('records'),
                                     ratings_df, behavior_df)
    categories    = sorted(products_df['category'].dropna().unique().tolist())

    # Sample users for the demo user-picker (first 30)
    sample_users = users_df.head(30).to_dict('records')

    return render_template('index.html',
                           products=products_list,
                           categories=categories,
                           selected_category=category,
                           search=search,
                           current_page=page,
                           total_pages=total_pages,
                           total_products=total,
                           sample_users=sample_users,
                           current_user_id=session.get('user_id'))


# -----------------------------------------------------------------------
# Select active user (session)
# -----------------------------------------------------------------------
@app.route('/select-user', methods=['POST'])
def select_user():
    uid = request.form.get('user_id', '').strip()
    if uid.isdigit():
        session['user_id'] = int(uid)
        return redirect(url_for('recommendations', user_id=int(uid)))
    return redirect(url_for('index'))


# -----------------------------------------------------------------------
# Product detail page
# -----------------------------------------------------------------------
@app.route('/product/<int:product_id>')
def product_detail(product_id):
    products_df = load_products()
    ratings_df  = load_ratings()
    behavior_df = load_behavior()

    row = products_df[products_df['product_id'] == product_id]
    if row.empty:
        return redirect(url_for('index'))

    product = row.iloc[0].to_dict()
    stats   = product_stats(product_id, ratings_df, behavior_df)
    product.update(stats)

    # Rating distribution (1-5 stars)
    prod_ratings = ratings_df[ratings_df['product_id'] == product_id]
    star_dist = {i: int((prod_ratings['rating'] == i).sum()) for i in range(1, 6)}

    # Similar products — same category, different id
    similar_df = products_df[
        (products_df['category'] == product['category']) &
        (products_df['product_id'] != product_id)
    ].head(4)
    similar = _enrich_products(similar_df.to_dict('records'), ratings_df, behavior_df)

    return render_template('product.html',
                           product=product,
                           star_dist=star_dist,
                           similar=similar,
                           current_user_id=session.get('user_id'))


# -----------------------------------------------------------------------
# Recommendations page (shell — data loaded via AJAX)
# -----------------------------------------------------------------------
@app.route('/recommendations/<int:user_id>')
def recommendations(user_id):
    users_df = load_users()
    row = users_df[users_df['user_id'] == user_id]
    if row.empty:
        return redirect(url_for('index'))

    session['user_id'] = user_id
    user = row.iloc[0].to_dict()
    return render_template(
        'recommendations.html',
        user=user,
        user_id=user_id,
        current_user_id=user_id,
    )


# -----------------------------------------------------------------------
# API — run GA and return recommendations as JSON
# -----------------------------------------------------------------------
@app.route('/api/recommendations/<int:user_id>')
def api_recommendations(user_id):
    products_df = load_products()
    ratings_df  = load_ratings()
    behavior_df = load_behavior()

    # Run the Genetic Algorithm
    # pop_size=20, generations=30 → 600 fitness evaluations (fast for demo)
    ga = GeneticRecommender(
        ratings_df   = ratings_df,
        behavior_df  = behavior_df,
        pop_size     = 20,
        generations  = 30,
        crossover_rate = 0.80,
        mutation_rate  = 0.10,
        seed         = 42
    )

    rec_ids, best_W, fitness_hist = ga.recommend(user_id, top_n=8)

    # ---- Fallback: cold-start → most popular products ----
    if not rec_ids:
        popular = (
            behavior_df.groupby('product_id')['purchased']
            .sum()
            .nlargest(8)
            .index.tolist()
        )
        rec_ids = popular

    # Build product cards
    rec_products = []
    for pid in rec_ids:
        p_row = products_df[products_df['product_id'] == pid]
        if p_row.empty:
            continue
        p = p_row.iloc[0].to_dict()
        stats = product_stats(pid, ratings_df, behavior_df)
        rec_products.append({**p, **stats})

    # Round weights for display
    labels  = ['Rating Weight', 'Viewed Weight', 'Clicked Weight', 'Purchased Weight']
    weights = {labels[i]: round(float(best_W[i]), 3) for i in range(4)}

    return jsonify({
        'products':       rec_products,
        'weights':        weights,
        'fitness_history': fitness_hist,
        'generations':    len(fitness_hist),
        'user_id':        user_id,
    })


# -----------------------------------------------------------------------
# API — quick product search (for search-as-you-type)
# -----------------------------------------------------------------------
@app.route('/api/search')
def api_search():
    q = request.args.get('q', '').strip()
    if len(q) < 2:
        return jsonify([])

    products_df = load_products()
    matched = products_df[
        products_df['name'].str.contains(q, case=False, na=False)
    ].head(8)

    return jsonify(matched[['product_id', 'name', 'category', 'price', 'emoji']].to_dict('records'))


def _preload():
    """Warm up the data cache before the first request arrives."""
    import sys
    print('[SmartAIShop] Preloading data files into cache...')
    load_users()
    load_products()
    load_ratings()
    load_behavior()
    print('[SmartAIShop] Cache ready — server accepting requests.')

# Preload once at startup (works in both debug-reloader processes)
_preload()

if __name__ == '__main__':
    app.run(debug=True)
