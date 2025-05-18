import json

summary_file = "./data/step3/3.3_phash_summary.jsonl"
total = 0
dup_only = 0
blank_only = 0
dup_and_blank = 0

with open(summary_file, "r", encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)
        total += 1
        has_dup = rec.get("has_duplicate", False)
        has_blank = rec.get("has_blank", False)

        if has_dup and has_blank:
            dup_and_blank += 1
        elif has_dup:
            dup_only += 1
        elif has_blank:
            blank_only += 1

print(f"Total ids                : {total}")
print(f"Duplicate only           : {dup_only}")
print(f"Blank only               : {blank_only}")
print(f"Duplicate & Blank (both) : {dup_and_blank}")
