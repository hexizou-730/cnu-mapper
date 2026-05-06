"""
fetch_theses.py
================

从 data.gouv.fr 下载法国博士论文官方 CSV (~1.4 GB, 456k 条),
本地流式过滤出我们需要的样本.

Download official French PhD theses CSV from data.gouv.fr (~1.4 GB, 456k rows),
filter locally to extract usable samples for CNU classification training.

Usage / 用法:
    python fetch_theses.py --download    # 下载 / download (~5-15 min)
    python fetch_theses.py --inspect     # 探测字段名 / inspect columns
    python fetch_theses.py --filter      # 过滤抽样 / filter + sample

Output / 输出:
    raw.csv        ~1.4 GB  原始数据
    theses.csv     ~80 MB   过滤后的 30k 条样本
"""

from __future__ import annotations

import argparse
import csv
import sys
import unicodedata
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

# ─────────── Configuration / 配置 ───────────
HERE = Path(__file__).parent

# Source: https://www.data.gouv.fr/datasets/theses-soutenues-en-france-depuis-1985
# Format: CSV, 1.4 GB, ~456,000 rows, License: Open License 2.0
DATA_URL = "https://www.data.gouv.fr/api/1/datasets/r/eb06a4f5-a9f1-4775-8226-33425c933272"
RAW_CSV = HERE / "raw.csv"
OUT_CSV = HERE / "theses.csv"

# Filtering parameters / 过滤参数
TARGET_TOTAL = 30_000           # total samples to keep / 总样本数
PER_DISCIPLINE_MAX = 400        # cap per discipline to avoid imbalance / 每学科上限
RESUME_MIN_LEN = 100            # min length of French abstract / 摘要最小长度
RESUME_MAX_LEN = 4000           # max length of French abstract / 摘要最大长度

# Column names — confirmed from actual data.gouv.fr CSV schema
# 字段名 — 已根据实际 CSV schema 确认
COLUMNS_GUESS = {
    "discipline": ["discipline"],
    "resume_fr":  ["resumes.fr"],
    "title_fr":   ["titres.fr"],
    "language":   ["langues.0"],
}


# ─────────── Step 1: download / 下载 ───────────
def step_download() -> None:
    """Download the full CSV from data.gouv.fr.
    从 data.gouv.fr 下载完整 CSV.
    """
    if RAW_CSV.exists():
        size_mb = RAW_CSV.stat().st_size / 1024 / 1024
        print(f"Already downloaded: {RAW_CSV.name} ({size_mb:.1f} MB)")
        print("   To re-download, delete the file first.")
        return

    print("Downloading from:")
    print(f"   {DATA_URL}")
    print(f"   Output: {RAW_CSV.name} (expected ~1.4 GB, ~5-15 min)")
    print("   Tip: you can also use curl in another terminal:")
    print(f'   curl -L -o "{RAW_CSV.name}" "{DATA_URL}"')
    print()

    def _hook(blocks, block_size, total):
        if total > 0:
            done = blocks * block_size / total * 100
            sys.stdout.write(f"\r   Progress: {done:5.1f}%")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(DATA_URL, RAW_CSV, _hook)
        print()
        size_mb = RAW_CSV.stat().st_size / 1024 / 1024
        print(f"Downloaded {size_mb:.1f} MB to {RAW_CSV.name}")
    except Exception as e:
        print(f"\nDownload failed: {e}")
        print(f"   Try downloading manually with curl (see command above).")
        sys.exit(1)


# ─────────── Step 2: inspect / 探测字段 ───────────
def step_inspect() -> None:
    """Print CSV header + first sample row to confirm column names.
    打印 CSV 表头 + 第一条样本, 用于确认字段名.
    """
    if not RAW_CSV.exists():
        sys.exit(f"Error: {RAW_CSV.name} not found. Run --download first.")

    print(f"Inspecting: {RAW_CSV.name}\n")

    with open(RAW_CSV, "r", encoding="utf-8", errors="replace") as f:
        sample = f.read(8192)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            delim = dialect.delimiter
        except csv.Error:
            delim = ","
    print(f"Detected delimiter: {repr(delim)}\n")

    with open(RAW_CSV, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f, delimiter=delim)
        header = next(reader)
        print(f"Total columns: {len(header)}")
        print("\nColumn names:")
        for i, col in enumerate(header):
            print(f"  [{i:2d}] {col}")

        print("\nFirst sample row:")
        for row in reader:
            if any(c.strip() for c in row):
                for i, (col, val) in enumerate(zip(header, row)):
                    preview = val[:120].replace("\n", " ")
                    if len(val) > 120:
                        preview += "..."
                    print(f"  [{i:2d}] {col}: {preview}")
                break

    print("\nUpdate COLUMNS_GUESS at the top if needed, then run --filter.")


# ─────────── Step 3: filter / 过滤 ───────────
def normalize(s: str) -> str:
    """Lowercase + strip accents / 小写 + 去重音."""
    if not s:
        return ""
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.lower().strip()


def find_column(header: list[str], candidates: list[str]) -> str | None:
    """Find first matching column name from candidates list.
    从候选列表里找第一个匹配的列名.
    """
    header_norm = {normalize(h): h for h in header}
    for c in candidates:
        if normalize(c) in header_norm:
            return header_norm[normalize(c)]
    for c in candidates:
        cn = normalize(c)
        for h_norm, h_orig in header_norm.items():
            if cn in h_norm:
                return h_orig
    return None


def step_filter() -> None:
    """Stream through the raw CSV, filter, and write filtered CSV.
    流式遍历原始 CSV, 过滤后写出.
    """
    if not RAW_CSV.exists():
        sys.exit(f"Error: {RAW_CSV.name} not found. Run --download first.")

    with open(RAW_CSV, "r", encoding="utf-8", errors="replace") as f:
        sample = f.read(8192)
        try:
            delim = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
        except csv.Error:
            delim = ","

    with open(RAW_CSV, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f, delimiter=delim)
        header = next(reader)

    col_disc = find_column(header, COLUMNS_GUESS["discipline"])
    col_resume = find_column(header, COLUMNS_GUESS["resume_fr"])
    col_title = find_column(header, COLUMNS_GUESS["title_fr"])
    col_lang = find_column(header, COLUMNS_GUESS["language"])

    print("Column mapping:")
    print(f"  discipline -> {col_disc}")
    print(f"  resume     -> {col_resume}")
    print(f"  title      -> {col_title}")
    print(f"  language   -> {col_lang}")

    if not (col_disc and col_resume):
        sys.exit(
            "Error: could not find required columns 'discipline' and 'resume'.\n"
            "Run --inspect to check the actual schema, then update COLUMNS_GUESS."
        )

    print(f"\nTarget: {TARGET_TOTAL} samples,"
          f" max {PER_DISCIPLINE_MAX}/discipline\n")
    print("Streaming through CSV...")

    buckets: dict[str, list[dict]] = defaultdict(list)
    seen_resumes: set[str] = set()
    total_rows = kept_rows = skipped_short = skipped_long = skipped_dup = 0

    with open(RAW_CSV, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter=delim)
        for row in reader:
            total_rows += 1
            if total_rows % 50000 == 0:
                print(f"   Processed {total_rows:>7} rows ... "
                      f"kept {kept_rows}", flush=True)

            disc = (row.get(col_disc) or "").strip()
            resume = (row.get(col_resume) or "").strip()

            if not disc or not resume:
                continue

            n = len(resume)
            if n < RESUME_MIN_LEN:
                skipped_short += 1
                continue
            if n > RESUME_MAX_LEN:
                skipped_long += 1
                continue

            sig = resume[:200]
            if sig in seen_resumes:
                skipped_dup += 1
                continue
            seen_resumes.add(sig)

            disc_norm = normalize(disc)
            if len(buckets[disc_norm]) >= PER_DISCIPLINE_MAX:
                continue

            buckets[disc_norm].append({
                "discipline": disc,
                "title": (row.get(col_title) or "").strip() if col_title else "",
                "resume": resume,
                "language": (row.get(col_lang) or "").strip() if col_lang else "",
            })
            kept_rows += 1

    print(f"\n   Total rows scanned: {total_rows}")
    print(f"   Kept after per-discipline cap: {kept_rows}")
    print(f"   Skipped (too short): {skipped_short}")
    print(f"   Skipped (too long): {skipped_long}")
    print(f"   Skipped (duplicate): {skipped_dup}")
    print(f"   Unique disciplines: {len(buckets)}")

    # Round-robin balancing / 轮询均衡
    print(f"\nBalancing to {TARGET_TOTAL} samples...")
    rows_out: list[dict] = []
    cursors = {d: 0 for d in buckets}
    while len(rows_out) < TARGET_TOTAL:
        progressed = False
        for disc_norm in sorted(buckets, key=lambda d: -len(buckets[d])):
            if len(rows_out) >= TARGET_TOTAL:
                break
            if cursors[disc_norm] < len(buckets[disc_norm]):
                rows_out.append(buckets[disc_norm][cursors[disc_norm]])
                cursors[disc_norm] += 1
                progressed = True
        if not progressed:
            break

    print(f"\nWriting: {OUT_CSV.name} ({len(rows_out)} rows)")
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["discipline", "title", "resume", "language"]
        )
        writer.writeheader()
        writer.writerows(rows_out)

    size_mb = OUT_CSV.stat().st_size / 1024 / 1024
    print(f"   File size: {size_mb:.1f} MB")

    counter = Counter(r["discipline"] for r in rows_out)
    print("\nTop 15 disciplines:")
    for disc, n in counter.most_common(15):
        print(f"  {n:4d}  {disc}")
    print(f"\n   Total disciplines in output: {len(counter)}")


# ─────────── Entry point / 命令行入口 ───────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch + filter French theses CSV for CNU classifier training."
    )
    parser.add_argument("--download", action="store_true",
                        help="Download the raw CSV (~1.4 GB).")
    parser.add_argument("--inspect", action="store_true",
                        help="Print CSV column names + first row to confirm schema.")
    parser.add_argument("--filter", action="store_true",
                        help="Filter the raw CSV into a smaller balanced sample.")
    args = parser.parse_args()

    if args.download:
        step_download()
    elif args.inspect:
        step_inspect()
    elif args.filter:
        step_filter()
    else:
        parser.print_help()
        print("\nTypical workflow:")
        print("  1. python fetch_theses.py --download")
        print("  2. python fetch_theses.py --inspect")
        print("  3. python fetch_theses.py --filter")


if __name__ == "__main__":
    main()
