"""
Merge 2.2_evolved_rules and 2.2_evolved_rules_others from two JSON files, reassign IDs from zero,
and remap parent IDs accordingly.
"""

import argparse
import copy
import json
import os


def parse_args():
    """
    Define and parse command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file1_path", type=str, default="./data/step2/2.2_evolved_rules.json"
    )
    parser.add_argument(
        "--file2_path", type=str, default="./data/step2/2.2_evolved_rules_others.json"
    )
    parser.add_argument(
        "--output_path", type=str, default="./data/step2/2.3_rules.json"
    )
    return parser.parse_args()


def merge_two_json_files(file1_path: str, file2_path: str, output_path: str):
    """
    Load rule lists from file1_path and file2_path, merge them,
    renumber IDs starting from zero, and remap parent IDs.
    """

    def load_rules_from_file(path: str):
        data = json.load(open(path, "r", encoding="utf-8"))
        seed_rules = data.get("seed_rules", [])
        evolved_rules = data.get("evolved_rules", [])
        return seed_rules + evolved_rules

    old_rules_1 = load_rules_from_file(file1_path)
    old_rules_2 = load_rules_from_file(file2_path)

    old_to_new_1 = {}
    old_to_new_2 = {}
    new_id = 0

    for rule in old_rules_1:
        old_to_new_1[rule["id"]] = new_id
        new_id += 1
    for rule in old_rules_2:
        old_to_new_2[rule["id"]] = new_id
        new_id += 1

    def convert_rule(old_rule: dict, this_map: dict, other_map: dict) -> dict:
        new_rule = copy.deepcopy(old_rule)
        old_id = old_rule["id"]
        new_rule["id"] = this_map[old_id]
        new_parents = []
        for p in old_rule.get("parents", []):
            if p in this_map:
                new_parents.append(this_map[p])
            elif p in other_map:
                new_parents.append(other_map[p])
        new_rule["parents"] = new_parents
        return new_rule

    merged = []
    for r in old_rules_1:
        merged.append(convert_rule(r, old_to_new_1, old_to_new_2))
    for r in old_rules_2:
        merged.append(convert_rule(r, old_to_new_2, old_to_new_1))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"Merged {len(merged)} rules to {output_path}")


def main(args):
    """
    Main entry point.
    """
    merge_two_json_files(args.file1_path, args.file2_path, args.output_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)
