"""
Download the EntityQuestions dataset (Sciavolino et al., 2021).

EntityQuestions contains simple entity-centric QA pairs drawn from Wikidata triples,
organized by relation type (Wikidata property ID).  We download the official release
from the Princeton-NLP GitHub repository and restructure it into a flat JSONL file.

Each output record:
  {
    "question": str,        natural-language question string
    "answers":  [str],      list of gold answer strings
    "relation": str,        Wikidata property ID, e.g. "P36"
    "subject":  str | null, entity extracted from question (best-effort regex parse)
    "split":    str,        "train", "dev", or "test"
  }

Usage:
    python data/download.py --output_dir data/raw
"""

import os
import re
import json
import argparse
import zipfile
import urllib.request
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# EntityQuestions download URL (Princeton NLP project page — contains actual data)
# GitHub repo contains only code; data is hosted separately.
# ──────────────────────────────────────────────────────────────────────────────
ENTITY_QUESTIONS_URL = (
    "https://nlp.cs.princeton.edu/projects/entity-questions/dataset.zip"
)

# ──────────────────────────────────────────────────────────────────────────────
# Regex patterns for extracting subjects from question strings.
# These are best-effort; a None result means the subject could not be parsed.
# Keys are Wikidata property IDs.
# ──────────────────────────────────────────────────────────────────────────────
SUBJECT_PATTERNS: dict[str, list[str]] = {
    "P17":   [r"(?:which|what) country is (.+?) located in\?"],
    "P19":   [r"(?:in which city|where) was (.+?) born\?",
              r"what (?:is|was) the (?:birthplace|place of birth) of (.+?)\?"],
    "P20":   [r"(?:in which city|where) did (.+?) die\?",
              r"what (?:is|was) the (?:death ?place|place of death) of (.+?)\?"],
    "P36":   [r"what is the capital (?:city )?of (.+?)\?"],
    "P69":   [r"where was (.+?) educated\?"],
    "P131":  [r"where is (.+?) located\?"],
    "P159":  [r"where is the headquarters? of (.+?)\?"],
    "P276":  [r"where is (.+?) located\?"],
    "P495":  [r"(?:which|what) country was (.+?) (?:created|made|produced|developed) in\?"],
    "P740":  [r"where was (.+?) (?:formed|founded|established)\?"],
}

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _extract_subject(question: str, relation: str) -> str | None:
    """Attempt to extract the subject entity from a question string."""
    patterns = SUBJECT_PATTERNS.get(relation, [])
    q = question.strip().lower()
    for pattern in patterns:
        m = re.search(pattern, q, re.IGNORECASE)
        if m:
            # Recover original casing from the original question
            start, end = m.span(1)
            subject_lower = q[start:end]
            # Find the same span in the original (case-preserved) question
            orig_lower = question.lower()
            idx = orig_lower.find(subject_lower)
            if idx >= 0:
                return question[idx: idx + len(subject_lower)]
    return None


def _download_zip(url: str, dest: Path) -> Path:
    """Download url to dest directory, return path to downloaded file."""
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / "entityquestions.zip"
    if zip_path.exists():
        print(f"  Cache hit: {zip_path}")
        return zip_path
    print(f"  Downloading {url} ...")
    urllib.request.urlretrieve(url, zip_path)
    print(f"  Saved to {zip_path}")
    return zip_path


def _iter_jsonl(path: Path):
    """Iterate records from a JSON file that is either a list or JSONL."""
    with open(path) as f:
        text = f.read().strip()
    if text.startswith("["):
        records = json.loads(text)
        yield from records
    else:
        for line in text.splitlines():
            line = line.strip()
            if line:
                yield json.loads(line)


def _find_split_files(root: Path) -> list[tuple[str, str, Path]]:
    """
    Walk the unpacked dataset archive and return (relation_id, split, path) tuples.

    Layout of the official Princeton dataset.zip:
        dataset/
            train/ P17.train.json  P19.train.json ...
            dev/   P17.dev.json    ...
            test/  P17.test.json   ...
    """
    result: list[tuple[str, str, Path]] = []
    for json_file in sorted(root.rglob("*.json")):
        # Skip one-off / bucket files — we only want the main splits
        if "one-off" in json_file.parts:
            continue
        stem_parts = json_file.stem.split(".")         # e.g. ["P17", "train"]
        if len(stem_parts) != 2:
            continue
        rel_id, split = stem_parts
        if not re.fullmatch(r"P\d+", rel_id):
            continue
        if split not in ("train", "dev", "test"):
            continue
        result.append((rel_id, split, json_file))
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Main download + conversion
# ──────────────────────────────────────────────────────────────────────────────

def download_entity_questions(output_dir: str) -> Path:
    """
    Download EntityQuestions, convert to flat JSONL, return the output path.

    Returns
    -------
    Path to the written JSONL file (``data/raw/entity_questions.jsonl``).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "entity_questions.jsonl"

    if out_path.exists():
        print(f"Already exists: {out_path} — skipping download.")
        return out_path

    # 1. Download archive
    zip_path = _download_zip(ENTITY_QUESTIONS_URL, out_dir / "_tmp")

    # 2. Unpack
    unpack_dir = out_dir / "_tmp" / "unpacked"
    unpack_dir.mkdir(parents=True, exist_ok=True)
    print("  Unpacking archive …")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(unpack_dir)

    # 3. Locate split files
    split_files = _find_split_files(unpack_dir)
    if not split_files:
        raise FileNotFoundError(
            "Could not find per-relation JSON files in the unpacked archive. "
            "Expected layout: dataset/train/P*.train.json, dataset/test/P*.test.json …"
        )
    relations = sorted({rel for rel, _, _ in split_files})
    print(f"  Found {len(relations)} relation(s): {relations}")

    # 4. Convert to flat JSONL
    total = 0
    with open(out_path, "w") as fout:
        for relation, split, json_file in split_files:
            for record in _iter_jsonl(json_file):
                question = record.get("question", "")
                answers  = record.get("answers", [])
                if not question or not answers:
                    continue
                subject = _extract_subject(question, relation)
                out_record = {
                    "question": question,
                    "answers":  answers,
                    "relation": relation,
                    "subject":  subject,
                    "split":    split,
                }
                fout.write(json.dumps(out_record) + "\n")
                total += 1

    print(f"  Wrote {total:,} records → {out_path}")
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download EntityQuestions dataset.")
    parser.add_argument(
        "--output_dir", default="data/raw",
        help="Directory to save raw JSONL (default: data/raw)"
    )
    args = parser.parse_args()
    path = download_entity_questions(args.output_dir)
    print(f"\nDone → {path}")
