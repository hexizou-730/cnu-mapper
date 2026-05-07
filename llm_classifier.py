"""
CNU Mapper - LLM-based Classifier
==================================

Classify university course descriptions (EN / FR / mixed) into one or more
CNU (Conseil National des Universités) sections, using Claude Opus 4.6
accessed via the OpenRouter API.

This script is intentionally separate from the DDC-based sklearn pipeline.
It does not train a local model. Instead, it builds a prompt from the official
CNU section list, sends the user's course description to an OpenRouter model,
and parses the returned JSON section codes.

Usage:
    conda activate cnu_mapper
    python llm_classifier.py                    # interactive mode
    python llm_classifier.py "description..."   # one-shot classification
    python llm_classifier.py --dry-run          # preview prompt only
    python llm_classifier.py --model <slug>     # switch OpenRouter model

API key resolution order:
    1. OPENROUTER_API_KEY environment variable
    2. .env file in the project directory
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path


# ───────── .env auto-loader ─────────
def _load_env_file() -> None:
    """Load an OpenRouter API key from a local .env file if present.

    The project keeps .env out of Git, but allowing this local file makes the
    CLI easier to run during demos. Two formats are accepted:

      OPENROUTER_API_KEY=sk-or-v1-xxx
      sk-or-v1-xxx

    The second form is tolerated so a user can paste only the key into .env.
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


# ───────── Configuration ─────────
# Default model on OpenRouter (can be overridden with --model)
MODEL = "anthropic/claude-opus-4.6"

# Knowledge base path: official CNU sections only.
KB_PATH = Path(__file__).parent / "cnu_knowledge_base_official.json"

# OpenRouter exposes an OpenAI-compatible chat completions API.
client = OpenAI(
    api_key=os.environ.get("OPENROUTER_API_KEY", ""),
    base_url="https://openrouter.ai/api/v1",
)


# ───────── Prompt construction ─────────
def load_kb() -> list[dict]:
    """Load the official CNU knowledge base.

    The official JSON is a dictionary with metadata and a `sections` list. This
    function returns only the section rows, sorted by code, because that is the
    form needed for prompt construction and output validation.
    """
    kb = json.loads(KB_PATH.read_text(encoding="utf-8"))
    sections = kb.get("sections", [])
    sections.sort(key=lambda s: s["code_section"])
    return sections


def build_system_prompt(kb: list[dict]) -> str:
    """Compose the system prompt sent to the LLM.

    The prompt explicitly lists the allowed label space. Each row includes the
    official CNU code, an English display name maintained by the project, the
    official French section name, and the official group. This reduces the
    chance that the model invents codes outside the CNU nomenclature.
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
    """Return the allowed official CNU section codes."""
    return {s["code_section"] for s in kb}


def clean_codes(codes: list, allowed: set[str]) -> list[str]:
    """Normalize raw model output and keep only official CNU codes.

    The model is instructed to return JSON, but external APIs can still return
    unexpected types or duplicate labels. This function is a small safety layer:
    it zero-pads numeric-looking codes, removes duplicates, filters non-official
    codes, and caps the final answer at three sections.
    """
    cleaned: list[str] = []
    for code in codes:
        code = str(code).strip().zfill(2)
        if code in allowed and code not in cleaned:
            cleaned.append(code)
        if len(cleaned) >= 3:
            break
    return cleaned


# ───────── Classification core ─────────
def classify(description: str, system_prompt: str) -> list[str]:
    """Call the OpenRouter model and parse returned CNU section codes.

    The request uses `response_format={"type": "json_object"}` to encourage a
    machine-readable response. If the returned content is not valid JSON, the
    fallback regex extracts quoted two-digit codes so the CLI can still recover
    a useful answer.
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
        return re.findall(r'"(\d{2})"', content)[:3]


# ───────── Interactive REPL ─────────
def interactive_mode(system_prompt: str, kb: list[dict]) -> None:
    """Run a small command-line REPL for manual classification.

    The same prompt and code-cleaning logic are reused for every input so that
    interactive predictions behave like one-shot predictions.
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
            # Keep the post-processing identical to one-shot mode.
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


# ───────── CLI entry point ─────────
def main() -> None:
    """Parse CLI arguments and run dry-run, one-shot, or interactive mode."""
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
        # Dry-run is useful for reviewing the exact label-space prompt without
        # spending API credits or requiring an API key.
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
        # One-shot classification for scripts, examples, and quick checks.
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
        # Interactive mode for manually trying several descriptions.
        interactive_mode(system_prompt, kb)


if __name__ == "__main__":
    main()
