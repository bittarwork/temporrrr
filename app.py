"""Flask entry point: preference learning (DT + GA) and product ranking UI."""

from __future__ import annotations

import os

from flask import Flask, flash, redirect, render_template, request, url_for

from recsys_hw.preference_service import get_engine


def create_app() -> Flask:
    app = Flask(__name__)
    app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-course-hw-change-me")

    @app.route("/")
    def home():
        eng = get_engine()
        err = None
        snap = eng.snapshot()
        try:
            eng.load_tables()
        except FileNotFoundError as e:
            err = str(e)
        users = eng.user_list() if eng.has_data() else []
        chart_spec = None
        dt_tree = None
        if snap is not None and eng.is_ready_for_reco():
            chart_spec = {"labels": snap.feature_labels, "values": snap.weights}
            dt_tree = eng.decision_tree_rules_text()
        return render_template(
            "index.html",
            data_ready=eng.has_data(),
            model_ready=eng.is_ready_for_reco(),
            users=users,
            snapshot=snap,
            error=err,
            chart_spec=chart_spec,
            dt_tree=dt_tree,
        )

    @app.route("/optimize", methods=["POST"])
    def optimize():
        eng = get_engine()
        try:
            seed_raw = request.form.get("seed", "").strip()
            seed = int(seed_raw) if seed_raw else None
            snap = eng.run_optimization(seed=seed)
            flash(
                f"Training finished. Validation MSE after GA = {snap.best_mse_ga:.6f} "
                f"(random seed = {snap.seed}).",
                "success",
            )
        except FileNotFoundError as e:
            flash(str(e), "error")
        except Exception as e:  # noqa: BLE001 — surface any training bug in coursework UI
            flash(f"Run failed: {e!s}", "error")
        return redirect(url_for("home"))

    @app.route("/user/<int:user_id>")
    def user_page(user_id: int):
        eng = get_engine()
        if not eng.has_data():
            try:
                eng.load_tables()
            except FileNotFoundError as e:
                flash(str(e), "error")
                return redirect(url_for("home"))
        ulist = {int(r["user_id"]) for r in eng.user_list()}
        if user_id not in ulist:
            flash("Unknown user_id.", "error")
            return redirect(url_for("home"))
        snap = eng.snapshot()
        prefs = eng.preference_profile() if eng.is_ready_for_reco() else []
        recs: list = []
        chart_spec = None
        if snap is not None and eng.is_ready_for_reco():
            chart_spec = {"labels": snap.feature_labels, "values": snap.weights}
        if eng.is_ready_for_reco():
            try:
                recs = eng.recommend_for_user(user_id, top_k=12)
            except Exception as e:  # noqa: BLE001
                flash(str(e), "error")
        urow = next((r for r in eng.user_list() if int(r["user_id"]) == user_id), None)
        return render_template(
            "user.html",
            user_id=user_id,
            urow=urow,
            prefs=prefs,
            recs=recs,
            model_ready=eng.is_ready_for_reco(),
            snapshot=snap,
            chart_spec=chart_spec,
        )

    @app.route("/api/preferences")
    def api_preferences():
        """JSON view of the latest GA preference weights (for reports / external tools)."""
        from flask import jsonify

        eng = get_engine()
        snap = eng.snapshot()
        if snap is None:
            return jsonify({"ok": False, "error": "No optimization run yet. Use Train + GA on the home page."}), 404
        from recsys_hw.features import FEATURE_LABELS_AR

        labs_ar = snap.feature_labels_ar
        if len(labs_ar) != len(snap.feature_labels):
            labs_ar = list(FEATURE_LABELS_AR)
        return jsonify(
            {
                "ok": True,
                "seed": snap.seed,
                "population_size": snap.population_size,
                "generations": snap.generations,
                "mse_ga": snap.best_mse_ga,
                "mse_baselines": {
                    "train_mean": snap.mse_baseline_mean,
                    "dt_only": snap.mse_baseline_dt_only,
                    "uniform_blend": snap.mse_baseline_uniform,
                },
                "preference_axes": [
                    {"label": lab, "label_ar": lab_ar, "weight": wt}
                    for lab, lab_ar, wt in zip(snap.feature_labels, labs_ar, snap.weights)
                ],
            }
        )

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=True)
