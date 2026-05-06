"""
tfidf_classifier.py
====================

CNU course-description classifier based on classical TF-IDF + One-vs-Rest
Logistic Regression. Trained on real French PhD thesis abstracts labeled
via Dewey → CNU mapping.

基于经典 TF-IDF + One-vs-Rest 逻辑回归的 CNU 课程描述分类器.
训练数据是真实法语博士论文摘要, 标签通过 Dewey → CNU 映射得到.

This is the "traditional ML" baseline counterpart to llm_classifier.py
(the LLM-based approach). The two scripts share the same CLI surface so
results can be compared side-by-side.

这是 llm_classifier.py (LLM 方案) 的"传统 ML"对照基线.
两个脚本的命令行接口完全对齐, 方便并排对比结果.

Usage / 用法:
    python tfidf_classifier.py --train       # 训练 + 保存模型 (~3-5 min)
    python tfidf_classifier.py --eval        # 80/20 评估
    python tfidf_classifier.py "..."         # 单条分类
    python tfidf_classifier.py               # 交互模式
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    jaccard_score,
)

# ─────────── Configuration / 配置 ───────────
HERE = Path(__file__).parent
TRAIN_CSV = HERE / "theses_labeled.csv"
KB_PATH = HERE / "cnu_knowledge_base_v2.json"
MODEL_PATH = HERE / "model.pkl"

# Output ranking parameters / 输出排序参数
TOP_K = 3                  # max number of CNU labels per prediction
PROBA_THRESHOLD = 0.40     # fallback threshold for old model.pkl files
VALIDATION_FRACTION = 0.20 # held-out train split used to tune prediction threshold
RANDOM_SEED = 42
THRESHOLD_GRID = np.linspace(0.05, 0.95, 37)


# TF-IDF parameters / TF-IDF 参数
# Slightly wider than the original baseline. Validation experiments on the
# current theses_labeled.csv showed a small Top-1/Top-3 gain without the large
# cost of character n-grams.
# 比原始 baseline 稍微放宽. 在当前 theses_labeled.csv 上验证后, 这个版本能小幅提升
# Top-1/Top-3, 同时避免字符 n-gram 带来的巨大训练成本.
TFIDF_PARAMS = dict(
    lowercase=True,
    strip_accents="unicode",     # remove French accents, makes "Ã©" → "e"
    ngram_range=(1, 2),          # unigrams + bigrams
    min_df=3,                    # word must appear in ≥3 docs
    max_df=0.65,                 # word in >65% docs is too common (filters stopwords-ish)
    max_features=100_000,        # vocabulary cap
    sublinear_tf=True,           # log scaling
)


# ─────────── Helpers / 辅助函数 ───────────
def load_kb() -> dict[str, str]:
    """Load CNU knowledge base, return {code: english_name}.
    加载 CNU 知识库, 返回 {代码: 英文名}.
    """
    kb = json.loads(KB_PATH.read_text(encoding="utf-8"))
    return {s["code_section"]: s["section_en"] for s in kb}


def load_data() -> tuple[list[str], list[list[str]]]:
    """Read theses_labeled.csv, return (texts, label_lists).
    读取已标注 CSV, 返回 (文本列表, 标签列表的列表).
    """
    if not TRAIN_CSV.exists():
        sys.exit(
            f"Error: {TRAIN_CSV.name} not found.\n"
            f"   Run label_theses.py first to generate it."
        )
    texts: list[str] = []
    labels: list[list[str]] = []
    with open(TRAIN_CSV, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row["resume"])
            labels.append(row["cnu_codes"].split("|"))
    return texts, labels


def split_indices(n: int, train_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic shuffled split, returning (train_idx, test_idx).
    确定性随机划分, 返回 (训练索引, 测试索引).
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    split = int(n * train_fraction)
    return indices[:split], indices[split:]


def make_vectorizer() -> TfidfVectorizer:
    """Create the configured TF-IDF vectorizer.
    创建配置好的 TF-IDF 向量化器.
    """
    return TfidfVectorizer(**TFIDF_PARAMS)


def make_classifier() -> OneVsRestClassifier:
    """Create the configured multi-label classifier.
    创建配置好的多标签分类器.
    """
    # NOTE: We deliberately use n_jobs=1 (no parallelism).
    # Parallel training (n_jobs=-1) crashes on macOS with the error
    # "WRITEBACKIFCOPY base is read-only" due to a joblib + scipy sparse
    # interaction. Sequential training is slower but reliable across platforms.
    # 注意: 故意用 n_jobs=1 (不并行).
    # 并行训练 (n_jobs=-1) 在 macOS 上会因 joblib+scipy 稀疏矩阵兼容性问题崩溃
    # 错误信息: "WRITEBACKIFCOPY base is read-only".
    # 顺序训练慢一点, 但跨平台稳定.
    return OneVsRestClassifier(
        LogisticRegression(max_iter=700, C=2.0, solver="liblinear",
                           class_weight="balanced"),
        n_jobs=1,
    )


def predict_binary_matrix(
    proba: np.ndarray,
    threshold: float,
    top_k: int = TOP_K,
) -> np.ndarray:
    """Convert probabilities to a multi-label binary matrix.
    将概率矩阵转换成多标签 0/1 矩阵.
    """
    y_pred = np.zeros(proba.shape, dtype=int)
    for i in range(proba.shape[0]):
        ranked = np.argsort(-proba[i])
        kept = [j for j in ranked[:top_k] if proba[i, j] >= threshold]

        # If nothing passed threshold, fall back to top-1 so every input gets
        # at least one CNU section.
        # 如果没有任何标签过阈值, 回退到 top-1, 保证每条输入至少有一个 CNU section.
        if not kept:
            kept = [ranked[0]]
        y_pred[i, kept] = 1
    return y_pred


def tune_threshold(proba: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
    """Tune one global probability threshold on validation micro-F1.
    在验证集上用 micro-F1 调一个全局概率阈值.
    """
    best_threshold = PROBA_THRESHOLD
    best_score = -1.0
    for threshold in THRESHOLD_GRID:
        y_pred = predict_binary_matrix(proba, float(threshold))
        score = f1_score(y_true, y_pred, average="micro", zero_division=0)
        if score > best_score:
            best_threshold = float(threshold)
            best_score = float(score)
    return best_threshold, best_score


def print_metrics(proba: np.ndarray, y_true: np.ndarray, threshold: float) -> None:
    """Print ranking and multi-label metrics.
    打印排序指标和多标签指标.
    """
    y_pred = predict_binary_matrix(proba, threshold)

    # Top-1 accuracy: does the top-prediction match ANY of the true labels?
    # Top-1 准确率: 第一个预测是否在真实标签集合里?
    top1_correct = 0
    top3_correct = 0
    for i in range(proba.shape[0]):
        ranked = np.argsort(-proba[i])
        true_idx = set(np.where(y_true[i])[0])
        if ranked[0] in true_idx:
            top1_correct += 1
        if any(r in true_idx for r in ranked[:3]):
            top3_correct += 1

    # Multi-label metrics / 多标签指标
    print("\n" + "=" * 54)
    print("EVALUATION RESULTS")
    print("=" * 54 + "\n")

    print(f"Decision threshold: {threshold:.3f}")

    print(f"\nTop-1 accuracy (top prediction in true set):")
    print(f"  Top-1 accuracy: "
          f"{top1_correct / y_true.shape[0] * 100:.2f}%")

    print(f"\nTop-3 accuracy (any of top-3 in true set):")
    print(f"  Top-3 accuracy: "
          f"{top3_correct / y_true.shape[0] * 100:.2f}%")

    print(f"\nMulti-label metrics (predicted labels vs true labels):")
    print(f"  Subset accuracy: "
          f"{accuracy_score(y_true, y_pred) * 100:.2f}%")
    print(f"  Hamming loss: "
          f"{hamming_loss(y_true, y_pred):.4f}  (lower is better)")
    print(f"  Micro F1: "
          f"{f1_score(y_true, y_pred, average='micro', zero_division=0):.4f}")
    print(f"  Macro F1: "
          f"{f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"  Jaccard (samples): "
          f"{jaccard_score(y_true, y_pred, average='samples', zero_division=0):.4f}")

    print(f"\nInterpretation:")
    print(f"   Top-1 is approximately 'the model's best guess matches the truth' "
          f"(single-label accuracy)")
    print(f"   Top-3 is approximately 'truth is in the model's top 3 guesses'")
    print(f"   Hamming loss is the average fraction of wrong bits in the label vector")


# ─────────── Training / 训练 ───────────
def step_train() -> None:
    """Train TF-IDF + OvR LogReg, save to disk.
    训练 TF-IDF + OvR LogReg, 保存到磁盘.
    """
    print("Loading training data...")
    texts, labels = load_data()
    print(f"   {len(texts):,} samples")

    fit_idx, val_idx = split_indices(
        len(texts), 1.0 - VALIDATION_FRACTION, RANDOM_SEED
    )
    fit_texts = [texts[i] for i in fit_idx]
    fit_labels = [labels[i] for i in fit_idx]
    val_texts = [texts[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]
    print(f"   Fit: {len(fit_texts):,}   Validation: {len(val_texts):,}")

    print("Fitting TF-IDF for threshold tuning...")
    t0 = time.time()
    tuning_vectorizer = make_vectorizer()
    X_fit = tuning_vectorizer.fit_transform(fit_texts)
    X_val = tuning_vectorizer.transform(val_texts)
    print(f"   Vocab size: {len(tuning_vectorizer.vocabulary_):,}")
    print(f"   Matrix shape: {X_fit.shape}")
    print(f"   Time: {time.time() - t0:.1f}s")

    print("Binarizing multi-labels...")
    tuning_mlb = MultiLabelBinarizer()
    Y_fit = tuning_mlb.fit_transform(fit_labels)
    Y_val = tuning_mlb.transform(val_labels)
    print(f"   Number of CNU classes: {len(tuning_mlb.classes_)}")
    print(f"   Y matrix shape: {Y_fit.shape}")

    print("Training tuning model...")
    t0 = time.time()
    tuning_clf = make_classifier()
    tuning_clf.fit(X_fit, Y_fit)
    print(f"   Time: {time.time() - t0:.1f}s")

    print("Tuning decision threshold...")
    val_proba = tuning_clf.predict_proba(X_val)
    threshold, val_micro_f1 = tune_threshold(val_proba, Y_val)
    print(f"   Best threshold: {threshold:.3f}")
    print(f"   Validation micro-F1: {val_micro_f1:.4f}")

    print("Fitting final TF-IDF on all data...")
    t0 = time.time()
    vectorizer = make_vectorizer()
    X = vectorizer.fit_transform(texts)
    print(f"   Vocab size: {len(vectorizer.vocabulary_):,}")
    print(f"   Matrix shape: {X.shape}")
    print(f"   Time: {time.time() - t0:.1f}s")

    print("Binarizing final multi-labels...")
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(labels)
    print(f"   Number of CNU classes: {len(mlb.classes_)}")
    print(f"   Y matrix shape: {Y.shape}")

    print("Training final One-vs-Rest LogReg...")
    t0 = time.time()
    clf = make_classifier()
    clf.fit(X, Y)
    print(f"   Time: {time.time() - t0:.1f}s")

    print(f"Saving model: {MODEL_PATH.name}")
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({
            "vectorizer": vectorizer,
            "classifier": clf,
            "mlb": mlb,
            "threshold": threshold,
            "top_k": TOP_K,
            "tfidf_params": TFIDF_PARAMS,
            "validation_micro_f1": val_micro_f1,
        }, f)
    size_mb = MODEL_PATH.stat().st_size / 1024 / 1024
    print(f"   Size: {size_mb:.1f} MB")
    print("\nTraining complete.")


# ─────────── Evaluation / 评估 ───────────
def step_eval() -> None:
    """80/20 holdout evaluation with threshold tuning on train only.
    80/20 评估: 只在训练部分内部校准阈值, 再测试.
    """
    print("Loading data...")
    texts, labels = load_data()
    n = len(texts)

    # Deterministic 80/20 split / 确定性的 80/20 划分
    train_idx, test_idx = split_indices(n, 0.80, RANDOM_SEED)
    fit_pos, val_pos = split_indices(
        len(train_idx), 1.0 - VALIDATION_FRACTION, RANDOM_SEED + 1
    )
    fit_idx = train_idx[fit_pos]
    val_idx = train_idx[val_pos]

    fit_texts = [texts[i] for i in fit_idx]
    fit_labels = [labels[i] for i in fit_idx]
    val_texts = [texts[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]
    test_texts = [texts[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]
    print(f"   Fit: {len(fit_texts):,}   "
          f"Validation: {len(val_texts):,}   "
          f"Test: {len(test_texts):,}")

    print("Fitting TF-IDF on fit set...")
    vectorizer = make_vectorizer()
    X_fit = vectorizer.fit_transform(fit_texts)
    X_val = vectorizer.transform(val_texts)
    X_test = vectorizer.transform(test_texts)
    print(f"   Vocab size: {len(vectorizer.vocabulary_):,}")
    print(f"   Matrix shape: {X_fit.shape}")

    mlb = MultiLabelBinarizer()
    Y_fit = mlb.fit_transform(fit_labels)
    Y_val = mlb.transform(val_labels)
    Y_test = mlb.transform(test_labels)

    print("Training OvR LogReg on fit set...")
    t0 = time.time()
    clf = make_classifier()
    clf.fit(X_fit, Y_fit)
    print(f"   Time: {time.time() - t0:.1f}s")

    print("Tuning threshold on validation set...")
    val_proba = clf.predict_proba(X_val)
    threshold, val_micro_f1 = tune_threshold(val_proba, Y_val)
    print(f"   Best threshold: {threshold:.3f}")
    print(f"   Validation micro-F1: {val_micro_f1:.4f}")

    print("\nEvaluating...")
    proba = clf.predict_proba(X_test)
    print_metrics(proba, Y_test, threshold)


# ─────────── Prediction (single + interactive) / 预测 ───────────
def load_model() -> tuple:
    if not MODEL_PATH.exists():
        sys.exit(
            f"Error: {MODEL_PATH.name} not found.\n"
            f"   Run --train first to create the model.\n"
        )
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    return (
        bundle["vectorizer"],
        bundle["classifier"],
        bundle["mlb"],
        float(bundle.get("threshold", PROBA_THRESHOLD)),
    )


def predict(text: str, vectorizer, clf, mlb, threshold: float) -> list[str]:
    """Predict CNU codes for a single text. Returns list of 1-3 codes.
    对单条文本预测 CNU 代码, 返回 1-3 个代码的列表.
    """
    X = vectorizer.transform([text])
    proba = clf.predict_proba(X)
    y_pred = predict_binary_matrix(proba, threshold)
    ranked = np.argsort(-proba[0])
    selected = set(np.where(y_pred[0])[0])
    return [mlb.classes_[j] for j in ranked if j in selected]


def interactive_mode(vectorizer, clf, mlb, threshold: float,
                     by_code: dict[str, str]) -> None:
    print("Model loaded: TF-IDF + OvR LogReg")
    print(f"Decision threshold: {threshold:.3f}")
    print("Interactive mode. Enter a course description + Enter to classify.")
    print("Type 'exit' / 'quit' or press Ctrl-D to leave.")
    print()
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
        codes = predict(text, vectorizer, clf, mlb, threshold)
        for c in codes:
            print(f"  {c}  {by_code.get(c, '?')}")
        print()
    print("Bye!")


# ─────────── Entry point / 命令行入口 ───────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="TF-IDF + OvR baseline classifier for CNU sections."
    )
    parser.add_argument("text", nargs="?",
                        help="Course description (in quotes). "
                             "Omit for interactive mode.")
    parser.add_argument("--train", action="store_true",
                        help="Train and save the model.")
    parser.add_argument("--eval", action="store_true",
                        help="80/20 holdout evaluation.")
    args = parser.parse_args()

    if args.train:
        step_train()
        return
    if args.eval:
        step_eval()
        return

    # Predict path needs the saved model / 预测路径需要训练好的模型
    vectorizer, clf, mlb, threshold = load_model()
    by_code = load_kb()

    if args.text:
        codes = predict(args.text, vectorizer, clf, mlb, threshold)
        print(f"Model: TF-IDF + OvR LogReg")
        print(f"Decision threshold: {threshold:.3f}")
        print(f"Description: {args.text}")
        print("Prediction:")
        for c in codes:
            print(f"  {c}  {by_code.get(c, '?')}")
    else:
        interactive_mode(vectorizer, clf, mlb, threshold, by_code)


if __name__ == "__main__":
    main()
