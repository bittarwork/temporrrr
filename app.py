"""
Flask storefront: products from Excel + DT + GA recommendations per user.
"""
from __future__ import annotations

from functools import lru_cache

from flask import Flask, abort, jsonify, render_template, request

from data_loader import load_all
from recommender import GAConfig, RecommenderEngine

app = Flask(__name__)
app.config["SECRET_KEY"] = "dev-change-in-production"


@lru_cache(maxsize=1)
def get_engine() -> RecommenderEngine:
    frames = load_all()
    return RecommenderEngine(frames, random_seed=42)


@app.context_processor
def inject_globals():
    return {"currency": "€"}


@app.route("/")
def home():
    frames = load_all()
    return render_template(
        "index.html",
        user_count=len(frames["users"]),
        product_count=len(frames["products"]),
    )


@app.route("/shop")
def shop():
    frames = load_all()
    products = frames["products"].sort_values("category").to_dict("records")
    return render_template("shop.html", products=products)


@app.route("/users")
def users_page():
    frames = load_all()
    users = frames["users"].sort_values("user_id").to_dict("records")
    return render_template("users.html", users=users)


def _ga_config_from_request() -> GAConfig:
    return GAConfig(
        list_size=int(request.args.get("k", 8)),
        generations=int(request.args.get("gen", 30)),
        population=int(request.args.get("pop", 64)),
    )


@app.route("/recommendations/<int:user_id>")
def recommendations(user_id):
    frames = load_all()
    if user_id not in frames["users"]["user_id"].values:
        abort(404)
    urow = frames["users"].loc[frames["users"]["user_id"] == user_id].iloc[0]
    return render_template(
        "recommendations.html",
        user_id=user_id,
        user=urow.to_dict(),
    )


@app.route("/api/recommendations/<int:user_id>")
def api_recommendations(user_id):
    frames = load_all()
    if user_id not in frames["users"]["user_id"].values:
        abort(404)
    cfg = _ga_config_from_request()
    engine = get_engine()
    items, meta = engine.recommend_for_user(user_id, cfg=cfg)
    return jsonify({"items": items, "meta": meta})


# PythonAnywhere and other WSGI hosts expect this name
application = app

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
