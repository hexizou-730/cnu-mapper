"""
inspect_labels.py
==================

诊断 raw.csv 里 discipline 和 oai_set_specs 字段的覆盖率与分布,
帮我们决定怎么把原始数据映射到 80 个 CNU section.

Diagnose discipline + oai_set_specs coverage in raw.csv to decide
how to map ~86k rows to the 80 CNU sections.

Usage / 用法:
    python inspect_labels.py
"""

from __future__ import annotations

import csv
import unicodedata
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
RAW_CSV = HERE / "raw.csv"

# Filtering parameters (must match fetch_theses.py)
# 过滤参数 (跟 fetch_theses.py 保持一致)
RESUME_MIN_LEN = 100
RESUME_MAX_LEN = 1500


def normalize(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.lower().strip()


def main() -> None:
    if not RAW_CSV.exists():
        raise SystemExit(f"Error: {RAW_CSV.name} not found")

    # Detect delimiter / 检测分隔符
    with open(RAW_CSV, "r", encoding="utf-8", errors="replace") as f:
        sample = f.read(8192)
        try:
            delim = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
        except csv.Error:
            delim = ","

    # Counters / 统计器
    total = 0
    qualifying = 0                           # passes resume length filter
    has_discipline = 0
    has_dewey = 0
    has_both = 0
    has_neither = 0

    discipline_norm_counter: Counter = Counter()
    dewey_main_counter: Counter = Counter()  # main Dewey class (first digit)
    dewey_two_counter: Counter = Counter()   # first 2 digits of Dewey
    dewey_full_counter: Counter = Counter()  # full Dewey value

    print(f"Diagnosing: {RAW_CSV.name}\n")

    with open(RAW_CSV, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter=delim)
        for row in reader:
            total += 1
            if total % 50000 == 0:
                print(f"   Processed {total:>7} rows ...", flush=True)

            resume = (row.get("resumes.fr") or "").strip()
            if not (RESUME_MIN_LEN <= len(resume) <= RESUME_MAX_LEN):
                continue
            qualifying += 1

            disc = (row.get("discipline") or "").strip()
            dewey_raw = (row.get("oai_set_specs") or "").strip()

            has_d = bool(disc)
            has_w = "ddc:" in dewey_raw

            if has_d:
                has_discipline += 1
                discipline_norm_counter[normalize(disc)] += 1

            if has_w:
                has_dewey += 1
                # oai_set_specs may contain multiple values like "ddc:540 ddc:570"
                # 字段可能含多个 Dewey 码, 比如 "ddc:540 ddc:570"
                # We take the first ddc value here for stats / 统计时取第一个
                for tok in dewey_raw.split():
                    if tok.startswith("ddc:"):
                        ddc = tok.replace("ddc:", "").strip()
                        if ddc.isdigit() and len(ddc) <= 3:
                            dewey_full_counter[ddc] += 1
                            dewey_two_counter[ddc[:2].zfill(2)] += 1
                            dewey_main_counter[ddc[:1]] += 1
                        break

            if has_d and has_w:
                has_both += 1
            elif not has_d and not has_w:
                has_neither += 1

    # ─── Report / 报告 ───
    print("\n" + "=" * 70)
    print("DIAGNOSTIC REPORT")
    print("=" * 70)

    print(f"\nTotal rows: {total:,}")
    print(f"Qualifying rows (resume {RESUME_MIN_LEN}-{RESUME_MAX_LEN} chars): {qualifying:,}")
    print("  This is the pool we will work with.\n")

    if qualifying == 0:
        return
    pct = lambda n: f"{n:,} ({n / qualifying * 100:.1f}%)"

    print(f"Field coverage in qualifying pool:")
    print(f"  Has 'discipline'     : {pct(has_discipline)}")
    print(f"  Has Dewey code (ddc) : {pct(has_dewey)}")
    print(f"  Has BOTH             : {pct(has_both)}")
    print(f"  Has NEITHER          : {pct(has_neither)}  unusable")

    # Dewey distribution
    print(f"\nDewey main classes (first digit):")
    DEWEY_NAMES = {
        "0": "Computer/info/general",
        "1": "Philosophy/psychology",
        "2": "Religion/theology",
        "3": "Social sciences",
        "4": "Languages",
        "5": "Pure sciences (math/physics/chem/bio/earth)",
        "6": "Technology/applied sciences (engineering/medicine)",
        "7": "Arts/recreation",
        "8": "Literature",
        "9": "History/geography",
    }
    for d in sorted(dewey_main_counter):
        n = dewey_main_counter[d]
        print(f"  {d}xx {DEWEY_NAMES.get(d, '?'):<48} "
              f"{n:>6,}")

    print(f"\nTop 25 Dewey 2-digit classes:")
    for ddc, n in dewey_two_counter.most_common(25):
        print(f"  {ddc}x  {n:>6,}")

    print(f"\nTop 30 normalized disciplines:")
    for disc, n in discipline_norm_counter.most_common(30):
        print(f"  {n:>5,}  {disc[:60]}")

    print(f"\nUnique normalized disciplines (after lowercase + remove accents)")
    print(f"Unique normalized disciplines: {len(discipline_norm_counter):,}")
    print(f"Unique 2-digit Dewey classes: {len(dewey_two_counter)}")
    print(f"Unique 3-digit Dewey codes: {len(dewey_full_counter)}")

    # ─── Recommendation / 建议 ───
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    dewey_pct = has_dewey / qualifying * 100
    disc_pct = has_discipline / qualifying * 100
    if dewey_pct >= 80:
        print(f"\nDewey coverage is GOOD ({dewey_pct:.0f}%).")
        print("   Recommendation: use Dewey codes as the primary mapping source.")
    elif dewey_pct >= 50:
        print(f"\nDewey coverage is PARTIAL ({dewey_pct:.0f}%).")
        print("   Recommendation: use Dewey first, then fall back to discipline.")
    else:
        print(f"\nDewey coverage is LOW ({dewey_pct:.0f}%).")
        print("   Recommendation: use discipline string rules as the primary source.")
    print(f"\n   Discipline coverage: {disc_pct:.0f}%")
    print(f"   Top 30 disciplines cover: "
          f"{sum(n for _, n in discipline_norm_counter.most_common(30)) / qualifying * 100:.0f}% of qualifying rows")


if __name__ == "__main__":
    main()
