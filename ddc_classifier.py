"""
ddc_classifier.py
=================

CNU classifier that follows the meeting-board idea:

    abstract/resume -> predict Dewey/DDC code(s) -> map DDC -> CNU

Two scikit-learn variants are implemented:

    1. logreg: TF-IDF + One-vs-Rest Logistic Regression
       This trains one binary classifier per DDC code.

    2. mlp: TF-IDF + one Multi-Layer Perceptron
       This trains one neural network that predicts the full DDC indicator
       vector at once.

Usage:
    python ddc_classifier.py --train --model-type logreg
    python ddc_classifier.py --eval  --model-type logreg
    python ddc_classifier.py --baselines
    python ddc_classifier.py --train --model-type mlp
    python ddc_classifier.py --eval  --model-type mlp
    python ddc_classifier.py --compare
    python ddc_classifier.py --model-type logreg "course description..."
    python ddc_classifier.py --model-type mlp
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.exceptions import ConvergenceWarning

from dewey_to_cnu import lookup as ddc_to_cnu, section_display_name


warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message=r"unknown class\(es\).*")


# ─────────── Configuration / 配置 ───────────
HERE = Path(__file__).parent
TRAIN_CSV = HERE / "theses_labeled.csv"
KB_PATH = HERE / "cnu_knowledge_base_official.json"

MODEL_PATHS = {
    "logreg": HERE / "ddc_model_logreg.pkl",
    "mlp": HERE / "ddc_model_mlp.pkl",
}

TOP_K = 3
PROBA_THRESHOLD = 0.40
VALIDATION_FRACTION = 0.20
RANDOM_SEED = 42
THRESHOLD_GRID = np.linspace(0.05, 0.95, 37)

LOGREG_TFIDF_PARAMS = dict(
    lowercase=True,
    strip_accents="unicode",
    analyzer="word",
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.65,
    max_features=100_000,
    sublinear_tf=True,
)

# MLP is much more expensive than linear logistic regression on sparse text.
# This smaller vocabulary keeps the neural-network baseline practical.
MLP_TFIDF_PARAMS = dict(
    lowercase=True,
    strip_accents="unicode",
    analyzer="word",
    ngram_range=(1, 1),
    min_df=3,
    max_df=0.80,
    max_features=20_000,
    sublinear_tf=True,
)


# ─────────── Data / 数据 ───────────
def load_kb() -> dict[str, str]:
    """Load official CNU names and attach project-maintained English labels.

    The official JSON contains authoritative CNU codes and French names only.
    English names are kept in dewey_to_cnu.py so that official data remains
    clean while CLI output stays readable.
    """
    kb = json.loads(KB_PATH.read_text(encoding="utf-8"))
    sections = kb.get("sections", [])
    return {
        s["code_section"]: section_display_name(
            s["code_section"],
            s.get("section_fr"),
        )
        for s in sections
    }


def load_data(max_samples: int | None = None) -> tuple[list[str], list[list[str]], list[list[str]]]:
    """Read the labeled CSV used by every DDC-based experiment.

    Returns:
        A tuple of (texts, ddc_labels, cnu_labels), where labels are already
        split into lists. Keeping DDC and CNU labels side by side lets us score
        both the intermediate DDC task and the final CNU task.
    """
    if not TRAIN_CSV.exists():
        sys.exit(
            f"Error: {TRAIN_CSV.name} not found.\n"
            f"   Run label_theses.py first to generate it."
        )

    texts: list[str] = []
    ddc_labels: list[list[str]] = []
    cnu_labels: list[list[str]] = []
    with open(TRAIN_CSV, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row["resume"])
            ddc_labels.append([c for c in row["ddc"].split("|") if c])
            cnu_labels.append([c for c in row["cnu_codes"].split("|") if c])
            if max_samples and len(texts) >= max_samples:
                break
    return texts, ddc_labels, cnu_labels


def split_indices(n: int, train_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic shuffled split of integer row indices."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    split = int(n * train_fraction)
    return indices[:split], indices[split:]


def make_train_test_split(
    n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create the shared fit/validation/test indices for evaluation.

    All reported rows, including Random and Majority baselines, use the same
    80/20 outer split. Trainable models further split the 80% training portion
    into fit and validation subsets for threshold calibration.
    """
    train_idx, test_idx = split_indices(n, 0.80, RANDOM_SEED)
    fit_pos, val_pos = split_indices(
        len(train_idx), 1.0 - VALIDATION_FRACTION, RANDOM_SEED + 1
    )
    return train_idx, train_idx[fit_pos], train_idx[val_pos], test_idx


# ─────────── Model factories / 模型工厂 ───────────
def make_vectorizer(model_type: str) -> TfidfVectorizer:
    """Create the TF-IDF vectorizer for one model family."""
    if model_type == "mlp":
        return TfidfVectorizer(**MLP_TFIDF_PARAMS)
    return TfidfVectorizer(**LOGREG_TFIDF_PARAMS)


def make_classifier(model_type: str):
    """Create the sklearn classifier behind each DDC model.

    LogReg uses one binary classifier per DDC label. The MLP baseline uses one
    neural network that predicts the full DDC multi-label vector at once.
    """
    if model_type == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(64,),
            activation="relu",
            solver="adam",
            alpha=1e-5,
            batch_size=256,
            learning_rate_init=3e-3,
            max_iter=50,
            early_stopping=False,
            random_state=RANDOM_SEED,
        )

    # One-vs-Rest means one binary logistic-regression model per DDC code.
    return OneVsRestClassifier(
        LogisticRegression(
            max_iter=700,
            C=2.0,
            solver="liblinear",
            class_weight="balanced",
        ),
        n_jobs=1,
    )


def model_path(model_type: str) -> Path:
    """Return the local pickle path for a trained DDC model."""
    return MODEL_PATHS[model_type]


# ─────────── Prediction helpers / 预测辅助 ───────────
def predict_proba_matrix(clf, x) -> np.ndarray:
    """Normalize sklearn probability outputs to shape [n_samples, n_classes]."""
    proba = clf.predict_proba(x)
    if isinstance(proba, list):
        return np.vstack([p[:, 1] for p in proba]).T
    return np.asarray(proba)


def predict_binary_matrix(
    proba: np.ndarray,
    threshold: float,
    top_k: int = TOP_K,
) -> np.ndarray:
    """Convert probability scores into a multi-label prediction matrix.

    A label is kept when it is both among the top-k labels for that row and
    above the calibrated threshold. If no label passes the threshold, the
    highest-scoring label is kept so every input receives at least one DDC.
    """
    y_pred = np.zeros(proba.shape, dtype=int)
    for i in range(proba.shape[0]):
        ranked = np.argsort(-proba[i])
        kept = [j for j in ranked[:top_k] if proba[i, j] >= threshold]
        if not kept:
            kept = [ranked[0]]
        y_pred[i, kept] = 1
    return y_pred


def tune_threshold(proba: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
    """Select the global DDC threshold that maximizes validation micro-F1."""
    best_threshold = PROBA_THRESHOLD
    best_score = -1.0
    for threshold in THRESHOLD_GRID:
        y_pred = predict_binary_matrix(proba, float(threshold))
        score = f1_score(y_true, y_pred, average="micro", zero_division=0)
        if score > best_score:
            best_threshold = float(threshold)
            best_score = float(score)
    return best_threshold, best_score


def ddc_to_cnu_codes(ddc_codes: list[str]) -> list[str]:
    """Map a list of DDC codes to a de-duplicated list of CNU section codes."""
    out: list[str] = []
    seen: set[str] = set()
    for ddc in ddc_codes:
        for cnu in ddc_to_cnu(ddc):
            if cnu not in seen:
                out.append(cnu)
                seen.add(cnu)
    return out


def ddc_binary_to_cnu_binary(
    y_ddc: np.ndarray,
    ddc_classes: np.ndarray,
    cnu_mlb: MultiLabelBinarizer,
) -> np.ndarray:
    """Map binary DDC predictions into the CNU label space.

    This keeps the evaluation faithful to the project design: models predict
    DDC first, and all final CNU scores are computed after the rule-based
    DDC-to-CNU mapping.
    """
    cnu_rows: list[list[str]] = []
    for row in y_ddc:
        ddc_codes = [ddc_classes[j] for j in np.where(row)[0]]
        cnu_rows.append(ddc_to_cnu_codes(ddc_codes))
    return cnu_mlb.transform(cnu_rows)


def ddc_top_k_accuracy_from_proba(
    proba: np.ndarray,
    y_ddc_true: np.ndarray,
    k: int,
) -> float:
    """Return whether each sample's true DDC appears in the top-k scores.

    The source dataset is effectively single-DDC in the current preprocessing
    setup, but this implementation also handles multi-DDC rows by counting a
    sample as correct when any true DDC label appears in the top-k candidates.
    """
    hits = 0
    for i in range(proba.shape[0]):
        ranked = np.argsort(-proba[i])[:k]
        if y_ddc_true[i, ranked].any():
            hits += 1
    return hits / max(proba.shape[0], 1)


def ddc_candidate_hit_accuracy(
    y_ddc_pred: np.ndarray,
    y_ddc_true: np.ndarray,
) -> float:
    """Return candidate-hit accuracy for predictors without ranked scores."""
    hits = ((y_ddc_pred & y_ddc_true).sum(axis=1) > 0).sum()
    return float(hits / max(y_ddc_pred.shape[0], 1))


def predict_one(text: str, vectorizer, clf, ddc_mlb, threshold: float) -> tuple[list[str], list[str]]:
    """Predict DDC and mapped CNU codes for one free-text description."""
    x = vectorizer.transform([text])
    proba = predict_proba_matrix(clf, x)
    y_pred = predict_binary_matrix(proba, threshold)
    ranked = np.argsort(-proba[0])
    selected = set(np.where(y_pred[0])[0])
    ddc_codes = [ddc_mlb.classes_[j] for j in ranked if j in selected]
    return ddc_codes, ddc_to_cnu_codes(ddc_codes)


# ─────────── Metrics / 指标 ───────────
def compute_metrics_from_ddc_binary(
    model_name: str,
    y_ddc_pred: np.ndarray,
    y_ddc_true: np.ndarray,
    y_cnu_true: np.ndarray,
    ddc_mlb: MultiLabelBinarizer,
    cnu_mlb: MultiLabelBinarizer,
    threshold: float | None = None,
    ddc_top1_accuracy: float | None = None,
    ddc_top3_accuracy: float | None = None,
) -> dict[str, float | str]:
    """Compute the compact report row from DDC predictions.

    The source dataset contains DDC metadata, not expert CNU annotations.
    Therefore DDC top-k accuracies are the primary supervised metrics. Mapped
    CNU Micro F1 is reported only after applying the rule-based DDC-to-CNU
    mapping, so it should be interpreted as a proxy metric.
    """
    y_cnu_pred = ddc_binary_to_cnu_binary(y_ddc_pred, ddc_mlb.classes_, cnu_mlb)
    if ddc_top1_accuracy is None:
        ddc_top1_accuracy = ddc_candidate_hit_accuracy(y_ddc_pred, y_ddc_true)
    if ddc_top3_accuracy is None:
        ddc_top3_accuracy = ddc_candidate_hit_accuracy(y_ddc_pred, y_ddc_true)

    return {
        "model": model_name,
        "ddc_micro_f1": f1_score(
            y_ddc_true, y_ddc_pred, average="micro", zero_division=0
        ),
        "cnu_micro_f1": f1_score(
            y_cnu_true, y_cnu_pred, average="micro", zero_division=0
        ),
        "ddc_top1_accuracy": ddc_top1_accuracy,
        "ddc_top3_accuracy": ddc_top3_accuracy,
        "threshold": threshold if threshold is not None else "",
    }


def compute_report_metrics(
    model_type: str,
    proba: np.ndarray,
    y_ddc_true: np.ndarray,
    y_cnu_true: np.ndarray,
    ddc_mlb: MultiLabelBinarizer,
    cnu_mlb: MultiLabelBinarizer,
    threshold: float,
) -> dict[str, float | str]:
    """Compute metrics for a trained probability-based DDC model."""
    y_ddc_pred = predict_binary_matrix(proba, threshold)
    model_name = "LogReg" if model_type == "logreg" else "MLP"
    return compute_metrics_from_ddc_binary(
        model_name, y_ddc_pred, y_ddc_true, y_cnu_true,
        ddc_mlb, cnu_mlb, threshold,
        ddc_top1_accuracy=ddc_top_k_accuracy_from_proba(proba, y_ddc_true, 1),
        ddc_top3_accuracy=ddc_top_k_accuracy_from_proba(proba, y_ddc_true, 3),
    )


def print_report_table(rows: list[dict[str, float | str]]) -> None:
    """Print the compact evaluation table used in README and CLI output."""
    print("\nEVALUATION SUMMARY")
    print("Model      DDC Top-1 Acc   DDC Top-3 Acc   Mapped CNU Micro F1")
    print("--------   -------------   -------------   -------------------")
    for row in rows:
        print(
            f"{row['model']:<8}   "
            f"{row['ddc_top1_accuracy'] * 100:6.2f}%        "
            f"{row['ddc_top3_accuracy'] * 100:6.2f}%        "
            f"{row['cnu_micro_f1']:.4f}"
        )


# ─────────── Simple baselines / 简单基线 ───────────
def random_ddc_predictions(n_samples: int, n_classes: int, seed: int) -> np.ndarray:
    """Predict one uniformly random DDC label for every test sample.

    The random predictor is intentionally simple. It gives an external reader
    a floor for the metric values: if a trained model cannot beat this row, it
    is not learning meaningful signal from the abstracts.
    """
    rng = np.random.default_rng(seed)
    chosen = rng.integers(0, n_classes, size=n_samples)
    y_pred = np.zeros((n_samples, n_classes), dtype=int)
    y_pred[np.arange(n_samples), chosen] = 1
    return y_pred


def majority_ddc_predictions(y_train: np.ndarray, n_samples: int) -> np.ndarray:
    """Always predict the most frequent DDC label observed in training data.

    This is the standard majority-class baseline adapted to the DDC-first
    multi-label setting. It is deliberately naive, but it is often much stronger
    than random when the dataset is imbalanced.
    """
    majority_label = int(np.argmax(y_train.sum(axis=0)))
    y_pred = np.zeros((n_samples, y_train.shape[1]), dtype=int)
    y_pred[:, majority_label] = 1
    return y_pred


def evaluate_baselines(max_samples: int | None = None, verbose: bool = True) -> list[dict[str, float | str]]:
    """Evaluate Random and Majority baselines on the shared 80/20 split."""
    print(f"Loading data: {TRAIN_CSV.name}")
    _, ddc_labels, cnu_labels = load_data(max_samples)
    if max_samples:
        print(f"   Max samples mode: {max_samples:,}")

    train_idx, _, _, test_idx = make_train_test_split(len(ddc_labels))
    train_ddc = [ddc_labels[i] for i in train_idx]
    test_ddc = [ddc_labels[i] for i in test_idx]
    test_cnu = [cnu_labels[i] for i in test_idx]

    print(f"   Train: {len(train_ddc):,}   Test: {len(test_ddc):,}")
    print("   Baselines: random, majority")

    ddc_mlb = MultiLabelBinarizer()
    y_train = ddc_mlb.fit_transform(train_ddc)
    y_test = ddc_mlb.transform(test_ddc)

    cnu_mlb = MultiLabelBinarizer()
    cnu_mlb.fit(cnu_labels)
    y_cnu_test = cnu_mlb.transform(test_cnu)

    random_pred = random_ddc_predictions(
        n_samples=len(test_ddc),
        n_classes=len(ddc_mlb.classes_),
        seed=RANDOM_SEED,
    )
    majority_pred = majority_ddc_predictions(y_train, len(test_ddc))

    rows = [
        compute_metrics_from_ddc_binary(
            "Random", random_pred, y_test, y_cnu_test, ddc_mlb, cnu_mlb
        ),
        compute_metrics_from_ddc_binary(
            "Majority", majority_pred, y_test, y_cnu_test, ddc_mlb, cnu_mlb
        ),
    ]
    if verbose:
        print_report_table(rows)
    return rows


# ─────────── Train / Eval / Predict ───────────
def step_train(model_type: str, max_samples: int | None = None) -> None:
    """Train one selected DDC model and save it as a pickle artifact."""
    print(f"Loading data: {TRAIN_CSV.name}")
    texts, ddc_labels, cnu_labels = load_data(max_samples)
    print(f"   Samples: {len(texts):,}")
    if max_samples:
        print(f"   Max samples mode: {max_samples:,}")
    print(f"   Model type: {model_type}")

    fit_idx, val_idx = split_indices(
        len(texts), 1.0 - VALIDATION_FRACTION, RANDOM_SEED
    )
    fit_texts = [texts[i] for i in fit_idx]
    fit_ddc = [ddc_labels[i] for i in fit_idx]
    val_texts = [texts[i] for i in val_idx]
    val_ddc = [ddc_labels[i] for i in val_idx]
    print(f"   Fit: {len(fit_texts):,}   Validation: {len(val_texts):,}")

    ddc_mlb = MultiLabelBinarizer()
    y_fit = ddc_mlb.fit_transform(fit_ddc)
    y_val = ddc_mlb.transform(val_ddc)
    print(f"   DDC classes: {len(ddc_mlb.classes_)}")

    print("Fitting TF-IDF for threshold tuning...")
    t0 = time.time()
    tuning_vectorizer = make_vectorizer(model_type)
    x_fit = tuning_vectorizer.fit_transform(fit_texts)
    x_val = tuning_vectorizer.transform(val_texts)
    print(f"   Vocab size: {len(tuning_vectorizer.vocabulary_):,}")
    print(f"   Matrix shape: {x_fit.shape}")
    print(f"   Time: {time.time() - t0:.1f}s")

    print("Training tuning DDC model...")
    t0 = time.time()
    tuning_clf = make_classifier(model_type)
    tuning_clf.fit(x_fit, y_fit)
    print(f"   Time: {time.time() - t0:.1f}s")

    print("Tuning DDC threshold...")
    val_proba = predict_proba_matrix(tuning_clf, x_val)
    threshold, val_micro_f1 = tune_threshold(val_proba, y_val)
    print(f"   Best threshold: {threshold:.3f}")
    print(f"   Validation DDC micro-F1: {val_micro_f1:.4f}")

    print("Fitting final TF-IDF on all data...")
    t0 = time.time()
    vectorizer = make_vectorizer(model_type)
    x_all = vectorizer.fit_transform(texts)
    print(f"   Vocab size: {len(vectorizer.vocabulary_):,}")
    print(f"   Matrix shape: {x_all.shape}")
    print(f"   Time: {time.time() - t0:.1f}s")

    final_ddc_mlb = MultiLabelBinarizer()
    y_all = final_ddc_mlb.fit_transform(ddc_labels)

    print("Training final DDC model...")
    t0 = time.time()
    clf = make_classifier(model_type)
    clf.fit(x_all, y_all)
    print(f"   Time: {time.time() - t0:.1f}s")

    path = model_path(model_type)
    print(f"Saving model: {path.name}")
    with open(path, "wb") as f:
        pickle.dump({
            "model_type": model_type,
            "vectorizer": vectorizer,
            "classifier": clf,
            "ddc_mlb": final_ddc_mlb,
            "threshold": threshold,
            "top_k": TOP_K,
            "validation_ddc_micro_f1": val_micro_f1,
            "tfidf_params": MLP_TFIDF_PARAMS if model_type == "mlp" else LOGREG_TFIDF_PARAMS,
        }, f)
    print(f"   Size: {path.stat().st_size / 1024 / 1024:.1f} MB")
    print("Training complete.")


def evaluate_model(
    model_type: str,
    max_samples: int | None = None,
    verbose: bool = True,
) -> dict[str, float | str]:
    """Train and evaluate one DDC model on the shared 80/20 split."""
    print(f"Loading data: {TRAIN_CSV.name}")
    texts, ddc_labels, cnu_labels = load_data(max_samples)
    if max_samples:
        print(f"   Max samples mode: {max_samples:,}")

    _, fit_idx, val_idx, test_idx = make_train_test_split(len(texts))

    fit_texts = [texts[i] for i in fit_idx]
    fit_ddc = [ddc_labels[i] for i in fit_idx]
    val_texts = [texts[i] for i in val_idx]
    val_ddc = [ddc_labels[i] for i in val_idx]
    test_texts = [texts[i] for i in test_idx]
    test_ddc = [ddc_labels[i] for i in test_idx]
    test_cnu = [cnu_labels[i] for i in test_idx]

    print(f"   Fit: {len(fit_texts):,}   "
          f"Validation: {len(val_texts):,}   "
          f"Test: {len(test_texts):,}")
    print(f"   Model type: {model_type}")

    ddc_mlb = MultiLabelBinarizer()
    y_fit = ddc_mlb.fit_transform(fit_ddc)
    y_val = ddc_mlb.transform(val_ddc)
    y_test = ddc_mlb.transform(test_ddc)

    cnu_mlb = MultiLabelBinarizer()
    cnu_mlb.fit(cnu_labels)
    y_cnu_test = cnu_mlb.transform(test_cnu)

    print("Fitting TF-IDF...")
    t0 = time.time()
    vectorizer = make_vectorizer(model_type)
    x_fit = vectorizer.fit_transform(fit_texts)
    x_val = vectorizer.transform(val_texts)
    x_test = vectorizer.transform(test_texts)
    print(f"   Vocab size: {len(vectorizer.vocabulary_):,}")
    print(f"   Matrix shape: {x_fit.shape}")
    print(f"   Time: {time.time() - t0:.1f}s")

    print("Training DDC model...")
    t0 = time.time()
    clf = make_classifier(model_type)
    clf.fit(x_fit, y_fit)
    print(f"   Time: {time.time() - t0:.1f}s")

    print("Tuning DDC threshold...")
    val_proba = predict_proba_matrix(clf, x_val)
    threshold, val_micro_f1 = tune_threshold(val_proba, y_val)
    print(f"   Best threshold: {threshold:.3f}")
    print(f"   Validation DDC micro-F1: {val_micro_f1:.4f}")

    print("\nEvaluating...")
    test_proba = predict_proba_matrix(clf, x_test)
    row = compute_report_metrics(
        model_type, test_proba, y_test, y_cnu_test, ddc_mlb, cnu_mlb, threshold
    )
    if verbose:
        print_report_table([row])
    return row


def step_eval(model_type: str, max_samples: int | None = None) -> None:
    """CLI wrapper for evaluating one trained-model family from scratch."""
    evaluate_model(model_type, max_samples, verbose=True)


def step_baselines(max_samples: int | None = None) -> None:
    """CLI wrapper for evaluating only the two simple baseline predictors."""
    evaluate_baselines(max_samples, verbose=True)


def step_compare(max_samples: int | None = None) -> None:
    """Evaluate baselines plus both trainable DDC models."""
    rows = evaluate_baselines(max_samples, verbose=False)
    for model_type in ("logreg", "mlp"):
        print(f"\n{'=' * 70}")
        print(f"Running {model_type}")
        print(f"{'=' * 70}")
        rows.append(evaluate_model(model_type, max_samples, verbose=False))
    print_report_table(rows)


def load_model(model_type: str):
    """Load a saved model bundle produced by step_train."""
    path = model_path(model_type)
    if not path.exists():
        sys.exit(
            f"Error: {path.name} not found.\n"
            f"   Run: python ddc_classifier.py --train --model-type {model_type}"
        )
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    return (
        bundle["vectorizer"],
        bundle["classifier"],
        bundle["ddc_mlb"],
        float(bundle.get("threshold", PROBA_THRESHOLD)),
    )


def interactive_mode(model_type: str, vectorizer, clf, ddc_mlb, threshold: float) -> None:
    """Run a small REPL for manual one-by-one predictions."""
    by_code = load_kb()
    print(f"Model: DDC {model_type}")
    print(f"Decision threshold: {threshold:.3f}")
    print("Interactive mode. Enter a course description + Enter to classify.")
    print("Type 'exit' / 'quit' or press Ctrl-D to leave.\n")
    while True:
        try:
            text = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text.lower() in ("exit", "quit", ":q"):
            break

        ddc_codes, cnu_codes = predict_one(text, vectorizer, clf, ddc_mlb, threshold)
        print(f"  DDC: {', '.join(ddc_codes)}")
        print("  CNU:")
        for cnu in cnu_codes:
            print(f"    {cnu}  {by_code.get(cnu, '?')}")
        print()
    print("Bye!")


def main() -> None:
    """Parse CLI arguments and dispatch to training, evaluation, or prediction."""
    parser = argparse.ArgumentParser(
        description="Predict DDC first, then map DDC to CNU."
    )
    parser.add_argument("text", nargs="?",
                        help="Course description. Omit for interactive mode.")
    parser.add_argument("--model-type", choices=("logreg", "mlp"), default="logreg",
                        help="DDC model variant: logreg or mlp.")
    parser.add_argument("--train", action="store_true",
                        help="Train and save the selected DDC model.")
    parser.add_argument("--eval", action="store_true",
                        help="80/20 evaluation of the selected DDC model.")
    parser.add_argument("--baselines", action="store_true",
                        help="Evaluate random and majority baselines only.")
    parser.add_argument("--compare", action="store_true",
                        help="Evaluate baselines, logreg, and mlp, then print one compact table.")
    parser.add_argument("--max-samples", type=int,
                        help="Optional small-sample run for quick tests. "
                             "Omit this for full training/evaluation.")
    args = parser.parse_args()

    if args.train:
        step_train(args.model_type, args.max_samples)
        return
    if args.eval:
        step_eval(args.model_type, args.max_samples)
        return
    if args.baselines:
        step_baselines(args.max_samples)
        return
    if args.compare:
        step_compare(args.max_samples)
        return

    vectorizer, clf, ddc_mlb, threshold = load_model(args.model_type)
    by_code = load_kb()
    if args.text:
        ddc_codes, cnu_codes = predict_one(args.text, vectorizer, clf, ddc_mlb, threshold)
        print(f"Model: DDC {args.model_type}")
        print(f"Decision threshold: {threshold:.3f}")
        print(f"Description: {args.text}")
        print(f"Predicted DDC: {', '.join(ddc_codes)}")
        print("Mapped CNU:")
        for cnu in cnu_codes:
            print(f"  {cnu}  {by_code.get(cnu, '?')}")
    else:
        interactive_mode(args.model_type, vectorizer, clf, ddc_mlb, threshold)


if __name__ == "__main__":
    main()
