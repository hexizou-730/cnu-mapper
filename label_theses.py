"""
label_theses.py
================

Read raw.csv, filter by resume length, attach CNU codes via Dewey mapping,
write theses_labeled.csv with one row per thèse.

读取 raw.csv, 按摘要长度过滤, 通过 Dewey 映射打上 CNU 代码标签,
输出 theses_labeled.csv.

Output schema / 输出字段:
    nnt           : National thesis number (unique ID)
    discipline    : Original discipline text (kept for reference)
    resume        : French abstract (the input text)
    ddc           : Pipe-separated Dewey codes (e.g. "540" or "540|530")
    cnu_codes     : Pipe-separated CNU codes (e.g. "31|32|33")
    n_labels      : Number of CNU labels assigned

Usage / 用法:
    python label_theses.py
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

from dewey_to_cnu import lookup as ddc_to_cnu

HERE = Path(__file__).parent
RAW_CSV = HERE / "raw.csv"
OUT_CSV = HERE / "theses_labeled.csv"

RESUME_MIN_LEN = 100
RESUME_MAX_LEN = 4000


def main() -> None:
    """Build theses_labeled.csv from the raw theses CSV.

    The output file is the weakly labeled training set used by the DDC-based
    classifiers. Labels are not manually annotated; they are derived by reading
    Dewey codes from the source CSV and mapping those codes to CNU sections.
    """
    if not RAW_CSV.exists():
        sys.exit(f"Error: {RAW_CSV.name} not found. Run fetch_theses.py --download first.")

    # Detect delimiter / 检测分隔符
    with open(RAW_CSV, "r", encoding="utf-8", errors="replace") as f:
        sample = f.read(8192)
        try:
            delim = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
        except csv.Error:
            delim = ","

    print(f"Reading: {RAW_CSV.name}")
    print(f"Filter: resume length {RESUME_MIN_LEN}-{RESUME_MAX_LEN} chars")
    print("Labeling: via Dewey to CNU mapping\n")

    # Counters / 统计器
    total = 0
    skipped_short = skipped_long = 0
    skipped_no_dewey = skipped_unknown_dewey = 0
    kept_rows = 0
    cnu_counter: Counter = Counter()
    n_label_counter: Counter = Counter()
    seen_resumes: set[str] = set()
    skipped_dup = 0

    out_fields = ["nnt", "discipline", "resume", "ddc", "cnu_codes", "n_labels"]

    with open(RAW_CSV, "r", encoding="utf-8", errors="replace", newline="") as f_in, \
         open(OUT_CSV, "w", encoding="utf-8", newline="") as f_out:

        reader = csv.DictReader(f_in, delimiter=delim)
        writer = csv.DictWriter(f_out, fieldnames=out_fields)
        writer.writeheader()

        for row in reader:
            total += 1
            if total % 50000 == 0:
                print(f"   Processed {total:>7} rows ... kept {kept_rows}",
                      flush=True)

            resume = (row.get("resumes.fr") or "").strip()
            n = len(resume)
            if n < RESUME_MIN_LEN:
                skipped_short += 1
                continue
            if n > RESUME_MAX_LEN:
                skipped_long += 1
                continue

            # Dedup on first 200 chars / 用前 200 字符去重
            sig = resume[:200]
            if sig in seen_resumes:
                skipped_dup += 1
                continue
            seen_resumes.add(sig)

            # Extract Dewey codes / 抽取 Dewey 码
            oai = (row.get("oai_set_specs") or "").strip()
            if "ddc:" not in oai:
                skipped_no_dewey += 1
                continue

            ddc_codes = []
            for tok in oai.split():
                if tok.startswith("ddc:"):
                    code = tok.replace("ddc:", "").strip()
                    if code.isdigit() and len(code) <= 3:
                        ddc_codes.append(code.zfill(3))
            if not ddc_codes:
                skipped_no_dewey += 1
                continue

            # Map each DDC to CNU, take union / 把每个 DDC 映射到 CNU, 取并集
            cnu_set: set[str] = set()
            for ddc in ddc_codes:
                for cnu in ddc_to_cnu(ddc):
                    cnu_set.add(cnu)

            if not cnu_set:
                skipped_unknown_dewey += 1
                continue

            cnu_list = sorted(cnu_set)

            writer.writerow({
                "nnt": (row.get("nnt") or "").strip(),
                "discipline": (row.get("discipline") or "").strip(),
                "resume": resume,
                "ddc": "|".join(ddc_codes),
                "cnu_codes": "|".join(cnu_list),
                "n_labels": len(cnu_list),
            })
            kept_rows += 1
            for c in cnu_list:
                cnu_counter[c] += 1
            n_label_counter[len(cnu_list)] += 1

    # Report / 报告
    print("\n" + "=" * 70)
    print("LABELING COMPLETE")
    print("=" * 70)
    print(f"\nTotal rows scanned:        {total:,}")
    print(f"Skipped (too short):       {skipped_short:,}")
    print(f"Skipped (too long):        {skipped_long:,}")
    print(f"Skipped (duplicate):       {skipped_dup:,}")
    print(f"Skipped (no Dewey):        {skipped_no_dewey:,}")
    print(f"Skipped (unknown DDC):     {skipped_unknown_dewey:,}")
    print(f"Kept:                      {kept_rows:,}")

    if kept_rows == 0:
        return

    print(f"\nOutput: {OUT_CSV.name} ({OUT_CSV.stat().st_size / 1024 / 1024:.1f} MB)")

    print("\nNumber-of-labels distribution:")
    for n, c in sorted(n_label_counter.items()):
        pct = c / kept_rows * 100
        print(f"  {n} label(s): {c:>7,} ({pct:5.1f}%)")

    print("\nCNU section coverage:")
    print(f"   Sections appearing: {len(cnu_counter)} / 80")
    print("\n   Top 20 most-represented CNU sections:")
    for cnu, c in cnu_counter.most_common(20):
        print(f"     CNU {cnu}: {c:>6,}")

    print("\n   Bottom 10 least represented sections that appear:")
    for cnu, c in sorted(cnu_counter.items(), key=lambda kv: kv[1])[:10]:
        print(f"     CNU {cnu}: {c:>6,}")

    # Which sections never appeared? / 哪些 section 从未出现?
    all_cnu = set(cnu_counter.keys())
    print("\n   Sections with zero samples:")
    # We list any 2-digit code 00-99 that isn't covered
    # Note: not all 2-digit codes are valid CNU; this is just a sanity sketch
    # 这里只是粗略提示, 不是所有 2 位数都是合法 CNU
    print("     Check this against cnu_knowledge_base_official.json")
    print(f"     Code count appearing in output: {len(all_cnu)}")


if __name__ == "__main__":
    main()
