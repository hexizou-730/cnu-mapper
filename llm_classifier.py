"""
CNU Mapper · LLM-based Classifier
==================================

Classify university course descriptions (EN / FR / mixed) into one or more
CNU (Conseil National des Universités) sections, using Claude Opus 4.6
accessed via the OpenRouter API.

将大学课程描述 (英 / 法 / 混合) 映射到一个或多个 CNU 学科代码.
通过 OpenRouter API 调用 Claude Opus 4.6 完成分类.

Usage / 用法:
    conda activate cnu_mapper
    python llm_classifier.py                    # interactive mode / 交互模式
    python llm_classifier.py "description..."   # one-shot / 单条分类
    python llm_classifier.py --dry-run          # preview prompt only / 仅预览 prompt
    python llm_classifier.py --model <slug>     # switch model / 换模型

API key resolution order / API key 查找顺序:
    1. OPENROUTER_API_KEY environment variable / 环境变量
    2. .env file in the project directory / 项目目录下的 .env 文件
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path


# ───────── .env auto-loader / 自动加载 .env ─────────
def _load_env_file() -> None:
    """Load .env file before any imports that might use the key.
    在需要 key 的模块导入之前加载 .env.

    Accepts two formats / 支持两种格式:
      OPENROUTER_API_KEY=sk-or-v1-xxx     (standard / 标准)
      sk-or-v1-xxx                         (tolerated / 容错, 只写 key 也行)
    """
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, _, v = line.partition("=")
            v = v.strip().strip('"').strip("'")
            os.environ.setdefault(k.strip(), v)
        elif line.startswith("sk-or-") or line.startswith("sk-"):
            os.environ.setdefault("OPENROUTER_API_KEY", line)


_load_env_file()

from openai import OpenAI   # noqa: E402 — import after env loading

from dewey_to_cnu import section_display_name   # noqa: E402


# ───────── Configuration / 配置 ─────────
# Default model on OpenRouter (can be overridden with --model)
# OpenRouter 上的默认模型 (可用 --model 覆盖)
MODEL = "anthropic/claude-opus-4.6"

# Knowledge base path: official CNU sections only.
# 知识库路径: 只使用官方 CNU section.
KB_PATH = Path(__file__).parent / "cnu_knowledge_base_official.json"

# OpenAI-compatible client pointing at OpenRouter
# OpenAI 兼容的 client, base_url 指向 OpenRouter
client = OpenAI(
    api_key=os.environ.get("OPENROUTER_API_KEY", ""),
    base_url="https://openrouter.ai/api/v1",
)


# ───────── Prompt construction / Prompt 构造 ─────────
def load_kb() -> list[dict]:
    """Load and sort the knowledge base by section code.
    加载知识库, 按 section 代码排序.
    """
    kb = json.loads(KB_PATH.read_text(encoding="utf-8"))
    sections = kb.get("sections", [])
    sections.sort(key=lambda s: s["code_section"])
    return sections


def build_system_prompt(kb: list[dict]) -> str:
    """Compose a system prompt listing the official CNU sections.
    生成列出官方 CNU section 的 system prompt.
    """
    lines = [
        "You are an expert classifier for French CNU (Conseil National des Universités) sections.",
        "Given a course description in English or French, identify which CNU section(s) it belongs to.",
        "",
        f"The {len(kb)} official CNU sections are listed below.",
        "Each line contains: code, English display name / official French name, and official group.",
        "",
    ]
    for s in kb:
        code = s["code_section"]
        display_name = section_display_name(code, s.get("section_fr"))
        lines.append(
            f"  {code} {display_name}"
            f"  - group {s.get('code_groupe', '?')}: {s.get('groupe_fr', '?')}"
        )
    lines += [
        "",
        "Respond with ONLY a JSON object (no markdown, no explanation) in this exact form:",
        '  {"sections": ["XX"]}              for single-discipline courses',
        '  {"sections": ["XX", "YY"]}        for interdisciplinary courses (up to 3 codes)',
        "",
        "List sections most relevant first. Use exact 2-digit codes from the list above.",
    ]
    return "\n".join(lines)


def allowed_codes(kb: list[dict]) -> set[str]:
    """Return the set of official CNU section codes."""
    return {s["code_section"] for s in kb}


def clean_codes(codes: list, allowed: set[str]) -> list[str]:
    """Normalize model output and keep only official CNU section codes."""
    cleaned: list[str] = []
    for code in codes:
        code = str(code).strip().zfill(2)
        if code in allowed and code not in cleaned:
            cleaned.append(code)
        if len(cleaned) >= 3:
            break
    return cleaned


# ───────── Classification core / 分类核心 ─────────
def classify(description: str, system_prompt: str) -> list[str]:
    """Call the LLM and parse the returned section codes.
    调用 LLM 并解析返回的 section 代码列表.

    Returns / 返回: e.g. ['27', '26']
    """
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": description},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
        max_tokens=80,
    )
    content = resp.choices[0].message.content or ""
    try:
        return json.loads(content).get("sections", [])[:3]
    except json.JSONDecodeError:
        # Fallback: regex-extract any 2-digit codes wrapped in quotes
        # 兜底: 用正则从文本里抓带引号的 2 位数字代码
        return re.findall(r'"(\d{2})"', content)[:3]


# ───────── Interactive REPL / 交互模式 ─────────
def interactive_mode(system_prompt: str, kb: list[dict]) -> None:
    """Read a description, classify it, repeat. Exit on 'exit' / Ctrl-D.
    循环读入描述 → 分类 → 继续. 输入 exit 或 Ctrl-D 退出.
    """
    by_code = {
        s["code_section"]: section_display_name(
            s["code_section"],
            s.get("section_fr"),
        )
        for s in kb
    }
    official_codes = allowed_codes(kb)
    print(f"Model: {MODEL}")
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
        try:
            codes = clean_codes(classify(text, system_prompt), official_codes)
        except Exception as e:
            print(f"  Error: {e}\n")
            continue
        if not codes:
            print("  (no matching section)\n")
            continue
        for c in codes:
            print(f"  {c}  {by_code.get(c, '?')}")
        print()
    print("Bye!")


# ───────── CLI entry point / 命令行入口 ─────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="CNU course-description classifier via OpenRouter (Claude Opus 4.6)."
    )
    parser.add_argument("text", nargs="?",
                        help="Course description (wrap in quotes). "
                             "If omitted, enter interactive mode.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the system prompt and exit (no API call).")
    parser.add_argument("--model",
                        help="Override default model, e.g. google/gemini-2.5-flash-lite")
    args = parser.parse_args()

    if args.model:
        global MODEL
        MODEL = args.model

    kb = load_kb()
    system_prompt = build_system_prompt(kb)
    official_codes = allowed_codes(kb)

    if args.dry_run:
        print(system_prompt)
        print(f"\n[system prompt length: {len(system_prompt)} chars "
              f"approx. {len(system_prompt) // 4} tokens]")
        return

    if not os.environ.get("OPENROUTER_API_KEY"):
        sys.exit(
            "Error: OPENROUTER_API_KEY not found.\n"
            "  Option 1: Create a .env file in this directory:\n"
            "            OPENROUTER_API_KEY=sk-or-v1-your-key\n"
            "  Option 2: export OPENROUTER_API_KEY=sk-or-v1-your-key"
        )

    if args.text:
        # One-shot classification / 单条分类
        codes = clean_codes(classify(args.text, system_prompt), official_codes)
        by_code = {
            s["code_section"]: section_display_name(
                s["code_section"],
                s.get("section_fr"),
            )
            for s in kb
        }
        print(f"Model: {MODEL}")
        print(f"Description: {args.text}")
        print("Prediction:")
        for c in codes:
            print(f"  {c}  {by_code.get(c, '?')}")
    else:
        # Interactive mode / 交互模式
        interactive_mode(system_prompt, kb)


if __name__ == "__main__":
    main()
