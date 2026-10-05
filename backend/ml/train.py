"""Train and compare crisis-risk classifiers, run the feature-ablation study and write the evaluation report.

Usage: python ml/train.py [--backend transformer|lexicon]
Requires data/risk_dataset.csv (prepare_data.py) and data/features_<backend>.parquet (featurize.py).
"""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.utils.class_weight import compute_sample_weight  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.nlp.features import BEHAVIORAL_FEATURES  # noqa: E402
from app.nlp.risk import RISK_LABELS, RiskModel, apply_thresholds  # noqa: E402
from ml.behavioral import simulate  # noqa: E402
from ml.featurize import EMBEDDING_MODEL  # noqa: E402

DATA_DIR = ROOT / "data"
ART_DIR = ROOT / "artifacts"
SEED = 42
TEXT_GROUPS = ("sentiment", "emotion", "linguistic")
HIGH_RECALL_TARGET = 0.90
AT_RISK_RECALL_TARGET = 0.90

ABLATIONS = {
    "E1 Sentiment": ("sentiment",),
    "E2 +Emotion": ("sentiment", "emotion"),
    "E3 +Linguistic": ("sentiment", "emotion", "linguistic"),
    "E4 +Behavioral (synthetic)": ("sentiment", "emotion", "linguistic", "behavioral"),
}


def candidate_models() -> dict[str, dict]:
    """name -> {estimator grid, RiskModel kwargs, needs sample weights}."""
    return {
        "Logistic Regression": {
            "grid": [LogisticRegression(C=c, max_iter=3000, class_weight="balanced") for c in (0.3, 1.0, 3.0)],
            "kwargs": {},
            "weights": False,
        },
        "Random Forest": {
            "grid": [
                RandomForestClassifier(n_estimators=400, max_depth=d, min_samples_split=4,
                                       class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)
                for d in (None, 20)
            ],
            "kwargs": {"tfidf_params": {"ngram_range": (1, 2), "max_features": 5000, "min_df": 3, "max_df": 0.9,
                                        "sublinear_tf": True}},
            "weights": False,
        },
        "XGBoost": {
            "grid": [
                XGBClassifier(n_estimators=n, max_depth=d, learning_rate=0.08, subsample=0.8, colsample_bytree=0.5,
                              tree_method="hist", eval_metric="mlogloss", n_jobs=2, random_state=SEED)
                for n, d in ((300, 4), (500, 3))
            ],
            "kwargs": {"tfidf_params": {"ngram_range": (1, 2), "max_features": 5000, "min_df": 3, "max_df": 0.9,
                                        "sublinear_tf": True}},
            "weights": True,
        },
        "Transformer (MiniLM embeddings + LR)": {
            "grid": [LogisticRegression(C=c, max_iter=3000, class_weight="balanced") for c in (0.3, 1.0, 3.0)],
            "kwargs": {"use_tfidf": False, "use_embeddings": True},
            "weights": False,
        },
    }


def groups_for(name: str, base: tuple[str, ...]) -> tuple[str, ...]:
    # The transformer variant replaces TF-IDF with embeddings but keeps the dense linguistic features.
    return base


def fit_model(spec: dict, est, groups, data, idx, y) -> RiskModel:
    m = RiskModel(clone(est), groups, **spec["kwargs"])
    sw = compute_sample_weight("balanced", y[idx]) if spec["weights"] else None
    emb = data["emb"][idx] if m.use_embeddings else None
    return m.fit(data["clean"][idx], data["dense"].iloc[idx], y[idx], emb=emb, sample_weight=sw)


def proba(m: RiskModel, data, idx) -> np.ndarray:
    emb = data["emb"][idx] if m.use_embeddings else None
    return m.predict_proba(data["clean"][idx], data["dense"].iloc[idx], emb)


def tune_thresholds(p: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Max macro-F1 subject to High recall and at-risk recall targets; falls back to max High recall."""
    best, best_key = None, None
    grid = np.round(np.arange(0.10, 0.81, 0.025), 3)
    for th, tm in itertools.product(grid, grid):
        pred = apply_thresholds(p, {"high": th, "moderate": tm})
        hi_rec = np.mean(pred[y == 2] == 2)
        risk_rec = np.mean(pred[y >= 1] >= 1)
        ok = hi_rec >= HIGH_RECALL_TARGET and risk_rec >= AT_RISK_RECALL_TARGET
        key = (ok, f1_score(y, pred, average="macro") if ok else hi_rec + risk_rec, th)
        if best_key is None or key > best_key:
            best_key, best = key, {"high": float(th), "moderate": float(tm)}
    return best


def evaluate(y: np.ndarray, p: np.ndarray, pred: np.ndarray) -> dict:
    prec, rec, f1, sup = precision_recall_fscore_support(y, pred, labels=[0, 1, 2], zero_division=0)
    cm = confusion_matrix(y, pred, labels=[0, 1, 2])
    hi_true, hi_pred = y == 2, pred == 2
    risk_true, risk_pred = y >= 1, pred >= 1
    return {
        "accuracy": accuracy_score(y, pred),
        "macro_precision": float(prec.mean()),
        "macro_recall": float(rec.mean()),
        "macro_f1": float(f1.mean()),
        "roc_auc_ovr_macro": float(roc_auc_score(y, p, multi_class="ovr", average="macro")),
        "high_roc_auc": float(roc_auc_score(hi_true, p[:, 2])),
        "per_class": {
            lab: {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i]), "support": int(sup[i])}
            for i, lab in enumerate(RISK_LABELS)
        },
        "high_false_negative_rate": float(np.mean(~hi_pred[hi_true])),
        "high_false_positive_rate": float(np.mean(hi_pred[~hi_true])),
        "at_risk_recall": float(np.mean(risk_pred[risk_true])),
        "at_risk_false_positive_rate": float(np.mean(risk_pred[~risk_true])),
        "confusion_matrix": cm.tolist(),
    }


def cross_validate(spec, est, groups, data, idx, y, folds=5) -> dict:
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    f1s, aucs, hi_recs = [], [], []
    for tr, te in skf.split(idx, y[idx]):
        m = fit_model(spec, est, groups, data, idx[tr], y)
        p = proba(m, data, idx[te])
        pred = p.argmax(1)
        f1s.append(f1_score(y[idx[te]], pred, average="macro"))
        aucs.append(roc_auc_score(y[idx[te]], p, multi_class="ovr", average="macro"))
        hi_recs.append(np.mean(pred[y[idx[te]] == 2] == 2))
    return {
        "macro_f1_mean": float(np.mean(f1s)), "macro_f1_std": float(np.std(f1s)),
        "roc_auc_mean": float(np.mean(aucs)), "roc_auc_std": float(np.std(aucs)),
        "high_recall_argmax_mean": float(np.mean(hi_recs)),
    }


def plot_confusion(cm, title, path):
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3), RISK_LABELS)
    ax.set_yticks(range(3), RISK_LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i][j], ha="center", va="center", color="white" if cm[i][j] > np.max(cm) / 2 else "black")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_ablation(abl: dict, path):
    models = list(abl.keys())
    exps = list(ABLATIONS.keys())
    metrics = ["macro_f1", "high_recall", "roc_auc_ovr_macro"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    w = 0.8 / len(models)
    for ax, met in zip(axes, metrics):
        for k, mname in enumerate(models):
            vals = [abl[mname][e][met] for e in exps]
            ax.bar(np.arange(len(exps)) + k * w, vals, w, label=mname)
        ax.set_xticks(np.arange(len(exps)) + w * (len(models) - 1) / 2, [e.split(" ")[0] for e in exps])
        ax.set_title(met)
        ax.set_ylim(0, 1)
    axes[0].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="transformer", choices=["transformer", "lexicon"])
    ap.add_argument("--skip-cv", action="store_true")
    args = ap.parse_args()

    ds = pd.read_csv(DATA_DIR / "risk_dataset.csv")
    feats = pd.read_parquet(DATA_DIR / f"features_{args.backend}.parquet")
    emb_path = DATA_DIR / "embeddings.npy"
    emb = np.load(emb_path) if emb_path.exists() else None
    y = ds["label"].map({k: i for i, k in enumerate(RISK_LABELS)}).to_numpy()

    real_dense = feats.drop(columns=["clean_text"])
    synth = simulate(ds["label"])
    sim_dense = real_dense.copy()
    sim_dense[BEHAVIORAL_FEATURES] = synth[BEHAVIORAL_FEATURES].to_numpy()
    data = {"clean": feats["clean_text"].fillna("").to_numpy(), "dense": real_dense, "emb": emb}
    data_sim = {**data, "dense": sim_dense}

    tr = np.where(ds.split == "train")[0]
    va = np.where(ds.split == "val")[0]
    te = np.where(ds.split == "test")[0]
    trva = np.concatenate([tr, va])
    cssrs_te = te[ds.source.to_numpy()[te] == "cssrs"]

    specs = candidate_models()
    if emb is None:
        specs.pop("Transformer (MiniLM embeddings + LR)")

    results, fitted = {}, {}
    for name, spec in specs.items():
        best = None
        for est in spec["grid"]:
            m = fit_model(spec, est, TEXT_GROUPS, data, tr, y)
            pv = proba(m, data, va)
            score = f1_score(y[va], pv.argmax(1), average="macro")
            if best is None or score > best[0]:
                best = (score, est, m, pv)
        _, est, m, pv = best
        m.thresholds = tune_thresholds(pv, y[va])
        pt = proba(m, data, te)
        val_pred = m.decide(pv)
        res = {
            "params": {k: v for k, v in est.get_params().items() if k in
                       ("C", "max_iter", "class_weight", "n_estimators", "max_depth", "min_samples_split",
                        "learning_rate", "subsample", "colsample_bytree")},
            "thresholds": m.thresholds,
            "val": {"macro_f1": float(f1_score(y[va], val_pred, average="macro")),
                    "high_recall": float(np.mean(val_pred[y[va] == 2] == 2))},
            "test_thresholded": evaluate(y[te], pt, m.decide(pt)),
            "test_argmax": evaluate(y[te], pt, pt.argmax(1)),
            "test_cssrs_only": evaluate(y[cssrs_te], proba(m, data, cssrs_te), m.decide(proba(m, data, cssrs_te))),
        }
        if not args.skip_cv:
            res["cv5_train_val"] = cross_validate(spec, est, TEXT_GROUPS, data, trva, y)
        results[name] = res
        fitted[name] = (m, spec, est)
        t = res["test_thresholded"]
        print(f"{name:40s} macroF1={t['macro_f1']:.3f} highRecall={t['per_class']['high']['recall']:.3f} "
              f"AUC={t['roc_auc_ovr_macro']:.3f} acc={t['accuracy']:.3f}", flush=True)

    # Feature-group ablation (SRS 13.2) for each classical model, best hyperparameters, thresholds re-tuned on val.
    ablation = {}
    for name in ("Logistic Regression", "Random Forest", "XGBoost"):
        m0, spec, est = fitted[name]
        ablation[name] = {}
        for exp, groups in ABLATIONS.items():
            d = data_sim if "behavioral" in groups else data
            m = fit_model(spec, est, groups, d, tr, y)
            m.thresholds = tune_thresholds(proba(m, d, va), y[va])
            pt = proba(m, d, te)
            ev = evaluate(y[te], pt, m.decide(pt))
            ablation[name][exp] = {
                "macro_precision": ev["macro_precision"], "macro_recall": ev["macro_recall"],
                "macro_f1": ev["macro_f1"], "roc_auc_ovr_macro": ev["roc_auc_ovr_macro"],
                "high_recall": ev["per_class"]["high"]["recall"], "high_precision": ev["per_class"]["high"]["precision"],
                "high_fnr": ev["high_false_negative_rate"],
            }
        print(f"ablation done: {name}", flush=True)

    # Model selection on validation only: High recall target first, then macro-F1 (accuracy is not used).
    selected = max(results, key=lambda n: (results[n]["val"]["high_recall"] >= HIGH_RECALL_TARGET,
                                           results[n]["val"]["macro_f1"]))
    model = fitted[selected][0]
    version = f"{args.backend}-{dt.date.today().isoformat()}-{selected.split(' ')[0].lower()}"
    meta = {
        "version": version, "model_name": selected, "nlp_backend": args.backend, "groups": list(model.groups),
        "embedding_model": EMBEDDING_MODEL, "thresholds": model.thresholds,
        "labels": RISK_LABELS, "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    ART_DIR.mkdir(exist_ok=True)
    fig_dir = ART_DIR / "figures"
    fig_dir.mkdir(exist_ok=True)
    joblib.dump({"model": model, "meta": meta}, ART_DIR / "risk_model.joblib" if args.backend == "transformer"
                else ART_DIR / f"risk_model_{args.backend}.joblib")

    split_counts = ds.groupby(["split", "label"]).size().unstack(fill_value=0).loc[["train", "val", "test"], RISK_LABELS]
    out = {"meta": meta, "selected_model": selected, "models": results, "ablation": ablation,
           "dataset": {"n": len(ds), "splits": split_counts.to_dict(orient="index"),
                       "sources": ds.groupby(["source", "label"]).size().unstack(fill_value=0).to_dict(orient="index")}}
    suffix = "" if args.backend == "transformer" else f"_{args.backend}"
    (ART_DIR / f"metrics{suffix}.json").write_text(json.dumps(out, indent=2, default=float))

    for name, res in results.items():
        slug = name.split(" ")[0].lower()
        plot_confusion(res["test_thresholded"]["confusion_matrix"], f"{name}\n(test, tuned thresholds)",
                       fig_dir / f"cm_{slug}{suffix}.png")
    plot_ablation(ablation, fig_dir / f"ablation{suffix}.png")
    write_report(out, ART_DIR / f"EVALUATION{suffix}.md", suffix)
    print(f"selected: {selected} -> {version}")


def _f(x: float) -> str:
    return f"{x:.3f}"


def write_report(out: dict, path: Path, suffix: str) -> None:
    m = out["models"]
    sel = out["selected_model"]
    L = []
    L.append(f"# Evaluation Report — Crisis-Risk Classification ({out['meta']['nlp_backend']} features)\n")
    L.append(f"Generated by `ml/train.py` on {out['meta']['trained_at'][:10]}. Model version: `{out['meta']['version']}`.\n")
    L.append("> Risk levels are model-generated indications, **not** clinical diagnoses. Metrics come from a small, "
             "public research dataset and do not establish real-world clinical validity.\n")
    L.append("## Dataset\n")
    L.append(f"{out['dataset']['n']} texts. CSSR-S severity 0 → Low, 1–2 → Moderate, 3–6 → High; everyday "
             "emotional statements from `dair-ai/emotion` added as Low. Stratified 70/15/15 split (by label × source).\n")
    L.append("| Split | Low | Moderate | High |\n|---|---|---|---|")
    for s, row in out["dataset"]["splits"].items():
        L.append(f"| {s} | {row['low']} | {row['moderate']} | {row['high']} |")
    L.append("\n## Model comparison (held-out test set)\n")
    L.append(f"Decision thresholds tuned on the validation set to reach High recall ≥ {HIGH_RECALL_TARGET} and "
             f"at-risk (Moderate+High) recall ≥ {AT_RISK_RECALL_TARGET} while maximising macro-F1. "
             "Selection uses validation metrics only — High recall first, then macro-F1; accuracy is not a criterion.\n")
    L.append("| Model | Accuracy | Macro P | Macro R | Macro F1 | ROC-AUC (OvR) | High recall | High precision | High FNR | High FPR | At-risk recall |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for name, r in m.items():
        t = r["test_thresholded"]
        star = " **(selected)**" if name == sel else ""
        L.append(f"| {name}{star} | {_f(t['accuracy'])} | {_f(t['macro_precision'])} | {_f(t['macro_recall'])} | "
                 f"{_f(t['macro_f1'])} | {_f(t['roc_auc_ovr_macro'])} | {_f(t['per_class']['high']['recall'])} | "
                 f"{_f(t['per_class']['high']['precision'])} | {_f(t['high_false_negative_rate'])} | "
                 f"{_f(t['high_false_positive_rate'])} | {_f(t['at_risk_recall'])} |")
    L.append("\n### Same models with plain argmax decisions (no threshold tuning)\n")
    L.append("| Model | Accuracy | Macro F1 | High recall | High FNR |\n|---|---|---|---|---|")
    for name, r in m.items():
        t = r["test_argmax"]
        L.append(f"| {name} | {_f(t['accuracy'])} | {_f(t['macro_f1'])} | {_f(t['per_class']['high']['recall'])} | "
                 f"{_f(t['high_false_negative_rate'])} |")
    L.append("\n### CSSR-S posts only (harder subset — no everyday-emotion Low examples)\n")
    L.append("| Model | Macro F1 | ROC-AUC | High recall | High precision |\n|---|---|---|---|---|")
    for name, r in m.items():
        t = r["test_cssrs_only"]
        L.append(f"| {name} | {_f(t['macro_f1'])} | {_f(t['roc_auc_ovr_macro'])} | "
                 f"{_f(t['per_class']['high']['recall'])} | {_f(t['per_class']['high']['precision'])} |")
    if any("cv5_train_val" in r for r in m.values()):
        L.append("\n### 5-fold cross-validation on train+val (argmax)\n")
        L.append("| Model | Macro F1 | ROC-AUC | High recall |\n|---|---|---|---|")
        for name, r in m.items():
            c = r.get("cv5_train_val")
            if c:
                L.append(f"| {name} | {_f(c['macro_f1_mean'])} ± {_f(c['macro_f1_std'])} | "
                         f"{_f(c['roc_auc_mean'])} ± {_f(c['roc_auc_std'])} | {_f(c['high_recall_argmax_mean'])} |")
    L.append("\n### Hyperparameters and thresholds\n")
    L.append("| Model | Parameters | Thresholds (high / moderate) |\n|---|---|---|")
    for name, r in m.items():
        p = ", ".join(f"{k}={v}" for k, v in r["params"].items())
        L.append(f"| {name} | {p} | {r['thresholds']['high']} / {r['thresholds']['moderate']} |")
    L.append("\n### Confusion matrices (test, tuned thresholds)\n")
    for name in m:
        L.append(f"![{name}](figures/cm_{name.split(' ')[0].lower()}{suffix}.png)")
    L.append("\n## Feature-ablation study (SRS §13.2)\n")
    L.append("Each experiment retrains the model with the listed feature groups and re-tunes thresholds on validation. "
             "**E4 behavioural features are simulated** (`ml/behavioral.py`) because no public dataset pairs crisis "
             "text with mood logs; E4 shows that the pipeline can use such signals, not that they predict real risk, "
             "and the production model does not include them (behavioural history is applied as a transparent "
             "escalation rule at runtime instead).\n")
    for name, exps in out["ablation"].items():
        L.append(f"**{name}**\n")
        L.append("| Experiment | Macro P | Macro R | Macro F1 | ROC-AUC | High recall | High precision | High FNR |")
        L.append("|---|---|---|---|---|---|---|---|")
        for e, r in exps.items():
            L.append(f"| {e} | {_f(r['macro_precision'])} | {_f(r['macro_recall'])} | {_f(r['macro_f1'])} | "
                     f"{_f(r['roc_auc_ovr_macro'])} | {_f(r['high_recall'])} | {_f(r['high_precision'])} | {_f(r['high_fnr'])} |")
        L.append("")
    L.append(f"![Ablation](figures/ablation{suffix}.png)\n")
    L.append("## Runtime safety layer (applies on top of every model)\n")
    L.append("- Explicit crisis language (`app/nlp/crisis.py`) always forces **High**, whatever the model says.")
    L.append("- A Low prediction is raised to Moderate when the user has a recent High-risk interaction or a sustained, "
             "declining low mood.")
    L.append("- High-risk replies use fixed, reviewed safety templates rather than generated text.\n")
    L.append("## Limitations\n")
    L.append("- The test set is small (tens of High examples), so expect wide confidence intervals; the CV table gives a sense of variance.")
    L.append("- CSSR-S labels come from one subreddit; the Low class mixes in a different source (short tweets), "
             "which makes Low easier to separate than it would be in real use — see the CSSR-S-only table.")
    L.append("- English only; sarcasm, indirect language and cultural variation are not covered.")
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
