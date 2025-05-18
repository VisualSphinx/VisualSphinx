import argparse, json


def get_args():
    ap = argparse.ArgumentParser("merge & filter by phash summary file")
    ap.add_argument(
        "--summary",
        default="./data/step3/3.3_phash_summary.jsonl",
        help="summary_*.jsonl",
    )
    ap.add_argument(
        "--input",
        default="./data/step3/3.2_valid_style_1.jsonl",
        help="original items JSON",
    )
    ap.add_argument("--output", default="items_filtered.json")
    return ap.parse_args()


args = get_args()

summary = {}
with open(args.summary, "r", encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)
        summary[int(rec["id"])] = rec  # id → {has_blank, has_duplicate}

with open(args.input, "r", encoding="utf-8") as f:
    items = json.load(f)  # list[dict]

result = []
for item in items:
    sid = int(item["id"])
    meta = summary.get(sid)
    if meta is None:
        continue
    if meta["has_blank"]:
        continue
    item["has_duplicate"] = bool(meta["has_duplicate"])
    result.append(item)

with open(args.output, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f"Saved {len(result)} items → {args.output}")
