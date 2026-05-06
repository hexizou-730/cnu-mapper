"""
check_coverage.py
==================

诊断当前 CNU 分类器的覆盖度: 名义 → 映射 → 数据 → 模型 四层
Diagnose CNU coverage across 4 layers: nominal → mapping → data → model.

Usage / 用法:
    python check_coverage.py
"""

import csv
import json
import pickle
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent

# ─── Layer 1: Official CNU sections ───
OFFICIAL_KB = HERE / "cnu_knowledge_base_official.json"
# ─── Layer 2: Dewey → CNU mapping table ───
from dewey_to_cnu import DEWEY_TO_CNU
# ─── Layer 3: Labeled training data ───
LABELED_CSV = HERE / "theses_labeled.csv"
# ─── Layer 4: Trained model ───
MODEL_PKL = HERE / "model.pkl"


def main() -> None:
    print("=" * 70)
    print("CNU COVERAGE DIAGNOSTIC")
    print("=" * 70)

    # ────────────────────────────────────────────────
    # Layer 1: Official CNU sections / 官方 CNU
    # ────────────────────────────────────────────────
    if not OFFICIAL_KB.exists():
        print(f"\nError: {OFFICIAL_KB.name} not found")
        return
    with open(OFFICIAL_KB, encoding="utf-8") as f:
        kb = json.load(f)
    official = set(s["code_section"] for s in kb["sections"])
    print(f"\n[Layer 1] Official CNU sections:")
    print(f"   {len(official)} sections")

    # ────────────────────────────────────────────────
    # Layer 2: Mapping table coverage / 映射表覆盖
    # ────────────────────────────────────────────────
    mapped = set()
    for cnus in DEWEY_TO_CNU.values():
        mapped.update(cnus)
    print(f"\n[Layer 2] CNU sections reachable through Dewey mapping:")
    print(f"   {len(mapped)} sections")
    missing_at_layer2 = sorted(official - mapped)
    print(f"   Missing from official (10 expected): {len(missing_at_layer2)} sections")
    print(f"   Missing sections: {missing_at_layer2}")

    # Show mapping table stats / 映射表统计
    label_counts = Counter(len(v) for v in DEWEY_TO_CNU.values())
    print(f"\n   Mapping table label-count distribution:")
    for n_labels, count in sorted(label_counts.items()):
        print(f"     {n_labels} label(s): {count} DDC entries")
    max_labels = max(len(v) for v in DEWEY_TO_CNU.values())
    print(f"   Max labels per DDC: {max_labels}  "
          f"({'v2' if max_labels <= 2 else 'v1 (uncapped)'})")

    # ────────────────────────────────────────────────
    # Layer 3: Training data coverage / 训练数据覆盖
    # ────────────────────────────────────────────────
    if not LABELED_CSV.exists():
        print(f"\nError: {LABELED_CSV.name} not found")
        print(f"   Run: python label_theses.py")
        return

    in_data = Counter()
    n_label_dist = Counter()
    total_rows = 0
    with open(LABELED_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            labels = row["cnu_codes"].split("|")
            n_label_dist[len(labels)] += 1
            for c in labels:
                in_data[c] += 1

    print(f"\n[Layer 3] CNU sections appearing in training data:")
    print(f"   {len(in_data)} sections (out of {total_rows:,} training rows)")
    print(f"   Missing sections compared with official list:")
    missing_at_layer3 = sorted(official - set(in_data.keys()))
    print(f"     Count: {len(missing_at_layer3)}; sections: {missing_at_layer3}")

    print(f"\n   Sections appearing but with very few samples (< 100):")
    rare = [(c, n) for c, n in in_data.items() if n < 100]
    if rare:
        for c, n in sorted(rare, key=lambda kv: kv[1]):
            print(f"     CNU {c}: {n} samples")
        reliable_count = len(in_data) - len(rare)
        print(f"\n   Reliable count (>=100 samples): {reliable_count}")
    else:
        print(f"   (none)")
        reliable_count = len(in_data)

    print(f"\n   Label-count distribution in training data:")
    for n, c in sorted(n_label_dist.items()):
        pct = c / total_rows * 100
        print(f"     {n} label(s): {c:>7,} ({pct:5.1f}%)")

    # ────────────────────────────────────────────────
    # Layer 4: Model output capability / 模型输出能力
    # ────────────────────────────────────────────────
    if not MODEL_PKL.exists():
        print(f"\nError: {MODEL_PKL.name} not found")
        print(f"   Run: python tfidf_classifier.py --train")
        return

    with open(MODEL_PKL, "rb") as f:
        bundle = pickle.load(f)
    model_classes = list(bundle["mlb"].classes_)
    print(f"\n[Layer 4] CNU sections the model can output:")
    print(f"   {len(model_classes)} sections")

    # ────────────────────────────────────────────────
    # Funnel summary / 漏斗总结
    # ────────────────────────────────────────────────
    print(f"\n" + "=" * 70)
    print(f"COVERAGE FUNNEL")
    print(f"=" * 70)
    print(f"")
    print(f"   Layer 1 - Official CNU:           {len(official):>3} sections")
    print(f"     lose {len(official) - len(mapped)} due to DDC granularity in health disciplines")
    print(f"")
    print(f"   Layer 2 - Reachable via mapping:  {len(mapped):>3} sections")
    print(f"     lose {len(mapped) - len(in_data)} due to data scarcity in theses.fr")
    print(f"")
    print(f"   Layer 3 - Appearing in training:  {len(in_data):>3} sections")
    print(f"     lose {len(in_data) - reliable_count} with < 100 samples")
    print(f"")
    print(f"   Layer 4 - Model can output:       {len(model_classes):>3} sections")
    print(f"   Reliable sections (>=100 samples): {reliable_count:>3}")
    print(f"")


if __name__ == "__main__":
    main()
