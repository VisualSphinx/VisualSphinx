"""Merge question metadata with reviewer scores extracted via regex."""

import json
import re
import sys
from pathlib import Path
import argparse


def parse_args():
    """CLI parameters."""
    p = argparse.ArgumentParser(
        description="Merge questions.json with ranking strings and extract scores."
    )
    p.add_argument(
        "--scores",
        default="./data/step4/Dataset_style_1_4options/4.4_puzzles_scoring.json",
    )
    p.add_argument(
        "--questions",
        default="./data/step4/Dataset_style_1_4options/4.3_puzzles_filtered.json",
    )
    p.add_argument(
        "--output",
        default="./data/step4/Dataset_style_1_4options/4.6_puzzles_with_scoring.json",
    )
    return p.parse_args()


def extract_score(score_string: str, pattern: re.Pattern):
    """Return int captured by pattern or None."""
    if not score_string:
        return None
    m = pattern.search(score_string)
    if m:
        try:
            return int(m.group(1))
        except (ValueError, IndexError):
            return None
    return None


def main(scores_path: Path, questions_path: Path, output_path: Path):
    """Load both JSONs, merge on id, add numeric scores, write result."""
    print(f"Loading scores from: {scores_path}")
    with open(scores_path, "r", encoding="utf-8") as f:
        scores_data = json.load(f)

    print(f"Loading questions from: {questions_path}")
    with open(questions_path, "r", encoding="utf-8") as f:
        questions_data = json.load(f)

    scores_map = {item["id"]: item.get("score") for item in scores_data if "id" in item}
    print(f"Score map size: {len(scores_map)}")

    re_reason = re.compile(r"Reasonableness:\s*(\d+)")
    re_readab = re.compile(r"Readability:\s*(\d+)")

    merged, seen = [], set()
    print("Merging data and extracting scores...")
    for q in questions_data:
        iid = q.get("id")
        if iid is None or iid in seen:
            print(f"Warning: skipping item with missing/duplicate id: {iid}")
            continue
        seen.add(iid)
        s = scores_map.get(iid)
        merged.append(
            {
                **q,
                "Reasonableness": extract_score(s, re_reason),
                "Readability": extract_score(s, re_readab),
            }
        )

    print(f"Writing merged data to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    print("Done.")


if __name__ == "__main__":
    a = parse_args()
    main(Path(a.scores), Path(a.questions), Path(a.output))
