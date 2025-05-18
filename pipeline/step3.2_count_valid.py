#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze folders of generated scripts by ID and attempt, count test successes and image counts,
then output IDs meeting specified criteria.
"""

import argparse
import json
import os
import re
from pathlib import Path
from collections import defaultdict


def parse_args():
    """
    Define and parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Analyze script folders and find valid IDs based on test results and image counts"
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="./data/step3/3.1_all_scripts_style_1",
        help="Root directory containing script subfolders",
    )
    parser.add_argument(
        "--correct_dir",
        type=str,
        default="output_correct",
        help="Subdirectory name for correct script outputs",
    )
    parser.add_argument(
        "--incorrect_dir",
        type=str,
        default="output_incorrect",
        help="Subdirectory name for incorrect script outputs",
    )
    parser.add_argument(
        "--correct_count",
        type=int,
        default=5,
        help="Required number of correct images to meet criteria",
    )
    parser.add_argument(
        "--incorrect_count",
        type=int,
        default=3,
        help="Required number of incorrect images to meet criteria",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="./data/step3/3.2_valid_style_1.json",
        help="Path to save the resulting valid IDs JSON",
    )
    return parser.parse_args()


def count_images(directory: str) -> int:
    """
    Count image files in a directory based on common extensions.
    """
    if not os.path.isdir(directory):
        return 0
    extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"}
    return sum(
        1 for f in os.listdir(directory) if os.path.splitext(f.lower())[1] in extensions
    )


def analyze_script_folders(
    root_dir: str,
    correct_dir: str,
    incorrect_dir: str,
    req_correct: int,
    req_incorrect: int,
) -> dict:
    """
    Walk through subfolders of the form <ID>_<attempt>, determine max attempt per ID,
    check test results and count images, and return IDs meeting the given criteria.
    """
    max_attempts = defaultdict(int)
    success_count = 0
    valid_ids = []

    for entry in os.scandir(root_dir):
        if not entry.is_dir():
            continue
        m = re.match(r"^(\d+)_(\d+)$", entry.name)
        if not m:
            continue
        id_str, attempt_str = m.groups()
        attempt = int(attempt_str)
        if attempt > max_attempts[id_str]:
            max_attempts[id_str] = attempt

    for id_str, attempt in max_attempts.items():
        folder = Path(root_dir) / f"{id_str}_{attempt}"
        result_file = folder / "test_result.txt"
        if result_file.exists() and "success" in result_file.read_text().lower():
            success_count += 1
        correct_count = count_images(folder / correct_dir)
        incorrect_count = count_images(folder / incorrect_dir)
        if correct_count == req_correct and incorrect_count == req_incorrect:
            valid_ids.append(id_str)

    print(f"Total unique IDs: {len(max_attempts)}")
    print(f"Successful test count: {success_count}")
    print(f"IDs meeting criteria: {len(valid_ids)}")

    return {"ids": valid_ids}


def main():
    """
    Main entry point: parse arguments, run analysis, and save results.
    """
    args = parse_args()
    result = analyze_script_folders(
        args.root_dir,
        args.correct_dir,
        args.incorrect_dir,
        args.correct_count,
        args.incorrect_count,
    )
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
