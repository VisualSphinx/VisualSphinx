"""Batch-evaluate items with OpenAI’s Batch API and checkpointed retries."""

import json
import re
import sys
import base64
import os
import time
import tempfile
from pathlib import Path
from typing import List, Dict
from collections import deque
import argparse

from jinja2 import Environment, FileSystemLoader
from tqdm import tqdm
from openai import OpenAI
from api_config import OpenAI_API_KEY as API_KEY


# ----------------------------------------------------------------------
# args / config
# ----------------------------------------------------------------------
def parse_args():
    """Return CLI args (all tunables surfaced here)."""
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt-4.1-mini-2025-04-14")
    p.add_argument("--chunk_size", type=int, default=500)
    p.add_argument("--max_active_batches", type=int, default=32)
    p.add_argument("--max_retries", type=int, default=5)
    p.add_argument("--ckpt_interval", type=int, default=200)
    p.add_argument("--poll_interval", type=int, default=30)
    p.add_argument(
        "--input",
        default="./data/step4/Dataset_style_1_4options/4.3_puzzles_filtered.json",
    )
    p.add_argument(
        "--output",
        default="./data/step4/Dataset_style_1_4options/4.4_puzzles_scoring.json",
    )
    p.add_argument("--template", default="./Prompts/step4.4_scoring.md")
    p.add_argument(
        "--raw",
        default="./data/step4/Dataset_style_1_4options/4.4_puzzles_scoring_raw.json",
    )
    return p.parse_args()


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
FINAL_REGEX = re.compile(r"<final_scores>(.*?)</final_scores>", re.DOTALL)

client = OpenAI(api_key=API_KEY)


def build_prompt(item: Dict, tpl, json_dir: Path) -> List[Dict]:
    """Render Jinja2 + image into multimodal messages."""
    if "correct_answer" not in item:
        return []
    rendered = tpl.render(
        question=item["prompt"],
        answer=item["correct_answer"],
        rules="\n".join(item.get("explanation", [])),
    )
    prefix, suffix = rendered.split("<!--SPLIT-->", 1)
    img = (
        "data:image/png;base64,"
        + base64.b64encode((json_dir / item["image"]).read_bytes()).decode()
    )
    return [
        {"type": "image_url", "image_url": {"url": img}},
        {"type": "text", "text": prefix + suffix},
    ]


def atomic_write(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
    tmp.replace(path)


def submit_batch(msgs: List[List[Dict]], model: str) -> str:
    """Upload JSONL & launch Batch; return job ID."""
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".jsonl") as f:
        for idx, m in enumerate(msgs):
            f.write(
                json.dumps(
                    {
                        "custom_id": f"req-{idx}",
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": model,
                            "messages": m,
                            "response_format": {"type": "text"},
                            "temperature": 0.2,
                        },
                    }
                )
                + "\n"
            )
        jp = Path(f.name)
    try:
        file_obj = client.files.create(file=open(jp, "rb"), purpose="batch")
        job = client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        return job.id
    finally:
        jp.unlink(missing_ok=True)


def parse_output(file_id: str, items: List[Dict], raw_fh) -> Dict[str, List[Dict]]:
    """Stream-parse Batch output into success / failure lists."""
    succ, fail = [], []
    try:
        stream = client.files.content(file_id)
        for ln, line in enumerate(stream.iter_lines(), 1):
            rec = json.loads(line)
            cid = rec.get("custom_id", "")
            if not cid.startswith("req-"):
                continue
            idx = int(cid.split("-")[-1])
            if idx >= len(items):
                continue
            itm = items[idx]
            resp = rec.get("response", {})
            status, body_raw = resp.get("status_code"), resp.get("body")
            if status is None or body_raw is None:
                fail.append(
                    {
                        "id": itm["id"],
                        "error": "Missing status/body",
                        "original_item": itm,
                    }
                )
                continue
            body = json.loads(body_raw) if isinstance(body_raw, str) else body_raw
            if status == 200:
                if not body or not body.get("choices"):
                    fail.append(
                        {
                            "id": itm["id"],
                            "error": "Empty choices",
                            "original_item": itm,
                        }
                    )
                    continue
                txt = body["choices"][0]["message"].get("content")
                if txt is None:
                    fail.append(
                        {"id": itm["id"], "error": "No content", "original_item": itm}
                    )
                    continue
                m = FINAL_REGEX.search(txt)
                if m:
                    succ.append({"id": itm["id"], "score": m.group(1).strip()})
                    raw_fh.write(
                        json.dumps({"id": itm["id"], "raw": txt}, ensure_ascii=False)
                        + "\n"
                    )
                else:
                    fail.append(
                        {"id": itm["id"], "error": "Tag missing", "original_item": itm}
                    )
            else:
                err = body.get("error", {}).get("message", f"status {status}")
                fail.append({"id": itm["id"], "error": err, "original_item": itm})
        raw_fh.flush()
        return {"successes": succ, "failures": fail}
    except Exception as e:
        print(f"[Error] parse_output failed: {e}")
        raise


# ----------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------
def main():
    """Entry."""
    import argparse  # kept local to satisfy parse_args above

    args = parse_args()
    CHUNK_SIZE = args.chunk_size
    MAX_ACTIVE_BATCHES = args.max_active_batches
    MAX_RETRIES = args.max_retries
    CKPT_INTERVAL = args.ckpt_interval
    POLL_INTERVAL = args.poll_interval

    inp_fp, out_fp, tpl_fp = Path(args.input), Path(args.output), Path(args.template)
    raw_fp = (
        Path(args.raw)
        if args.raw
        else out_fp.with_name("raw_" + inp_fp.stem + ".jsonl")
    )
    env = Environment(loader=FileSystemLoader(tpl_fp.parent))
    tpl = env.get_template(tpl_fp.name)
    items_all = json.loads(inp_fp.read_text())
    json_dir = inp_fp.parent

    results, done = [], set()
    if out_fp.exists():
        results = json.loads(out_fp.read_text())
        done = {r["id"] for r in results}
        print(f"[Resume] {len(done)} done.")

    pending = deque({**it, "attempt": 1} for it in items_all if it["id"] not in done)
    chunks = deque()
    while pending:
        chunk = [pending.popleft() for _ in range(min(len(pending), CHUNK_SIZE))]
        chunks.append(chunk)
    active = []
    proc = [len(done)]

    with raw_fp.open("a", encoding="utf-8") as raw_fh, tqdm(
        total=len(items_all), desc="Progress", initial=proc[0]
    ) as pbar:
        while chunks or active:
            while chunks and len(active) < MAX_ACTIVE_BATCHES:
                batch_items = [it for it in chunks.popleft() if it["id"] not in done]
                if not batch_items:
                    continue
                msgs = [
                    [{"role": "user", "content": build_prompt(it, tpl, json_dir)}]
                    for it in batch_items
                ]
                try:
                    jid = submit_batch(msgs, args.model)
                    active.append({"id": jid, "items": batch_items})
                    print(f"[Launch] {jid} ({len(batch_items)} items)")
                except Exception as e:
                    print(f"[Submit-Error] {e}")
                    for it in batch_items:
                        if it["attempt"] < MAX_RETRIES:
                            it["attempt"] += 1
                            pending.appendleft(it)
                        else:
                            results.append(
                                {"id": it["id"], "error": f"submit fail: {e}"}
                            )
                            done.add(it["id"])
                            pbar.update(1)
                            proc[0] += 1

            if not active:
                time.sleep(min(POLL_INTERVAL, 5) if chunks else 0.1)
                continue

            finished = []
            for job in active:
                try:
                    info = client.batches.retrieve(job["id"])
                except Exception as e:
                    print(f"[API] retrieve {job['id']} err: {e}")
                    continue
                if info.status == "completed":
                    if not info.output_file_id:
                        err = "no output file"
                        for it in job["items"]:
                            if it["id"] in done:
                                continue
                            if it["attempt"] < MAX_RETRIES:
                                it["attempt"] += 1
                                pending.appendleft(it)
                            else:
                                results.append({"id": it["id"], "error": err})
                                done.add(it["id"])
                                pbar.update(1)
                                proc[0] += 1
                        finished.append(job)
                        continue
                    try:
                        pr = parse_output(info.output_file_id, job["items"], raw_fh)
                    except Exception as e:
                        print(f"[Parse-Error] {e}")
                        for it in job["items"]:
                            if it["id"] in done:
                                continue
                            if it["attempt"] < MAX_RETRIES:
                                it["attempt"] += 1
                                pending.appendleft(it)
                            else:
                                results.append({"id": it["id"], "error": str(e)})
                                done.add(it["id"])
                                pbar.update(1)
                                proc[0] += 1
                        finished.append(job)
                        continue
                    for s in pr["successes"]:
                        if s["id"] not in done:
                            results.append(s)
                            done.add(s["id"])
                            pbar.update(1)
                            proc[0] += 1
                    for f in pr["failures"]:
                        it = f["original_item"]
                        if it["id"] in done:
                            continue
                        if it["attempt"] < MAX_RETRIES:
                            it["attempt"] += 1
                            pending.appendleft(it)
                        else:
                            results.append({"id": it["id"], "error": f["error"]})
                            done.add(it["id"])
                            pbar.update(1)
                            proc[0] += 1
                    finished.append(job)
                elif info.status in {"failed", "expired", "cancelled"}:
                    err = f"batch {info.status}"
                    for it in job["items"]:
                        if it["id"] in done:
                            continue
                        if it["attempt"] < MAX_RETRIES:
                            it["attempt"] += 1
                            pending.appendleft(it)
                        else:
                            results.append({"id": it["id"], "error": err})
                            done.add(it["id"])
                            pbar.update(1)
                            proc[0] += 1
                    finished.append(job)
            if finished:
                active = [j for j in active if j not in finished]
            if (
                proc[0] // CKPT_INTERVAL > (proc[0] - len(finished)) // CKPT_INTERVAL
                and proc[0]
            ):
                atomic_write(out_fp, results)
                print(f"[CKPT] at {proc[0]}.")
            time.sleep(POLL_INTERVAL if (active or chunks) else 0)

    atomic_write(out_fp, results)
    print(f"[Done] {proc[0]} items → {out_fp}")


if __name__ == "__main__":
    main()
