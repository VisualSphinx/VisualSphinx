"""
Batch puzzle classifier using the Anthropic β batch API and external Markdown templates.
"""

import argparse
import base64
import json
import mimetypes
import os
import threading
import time
import re
from pathlib import Path

import anthropic
from jinja2 import Environment, FileSystemLoader
from tqdm import tqdm
from api_config import Anthropic_API_KEY


def parse_args():
    """
    Define and parse command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=str, default="./data/step1_filtered/1.5_questions.json"
    )
    parser.add_argument(
        "--output", type=str, default="./data/step1_filtered/1.8_rule_class.json"
    )
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--template", type=str, default="Prompts/step1.8_classify.md")
    parser.add_argument("--model", type=str, default="claude-3-7-sonnet-20250219")
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument("--image_dir", type=str, default="")
    return parser.parse_args()


def encode_image(path: str) -> tuple[str, str]:
    """
    Encode an image file to a base64 string and determine its MIME type.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    mime, _ = mimetypes.guess_type(path)
    mime = mime or ("image/png" if path.lower().endswith(".png") else "image/jpeg")
    data = base64.b64encode(Path(path).read_bytes()).decode("utf-8")
    return data, mime


class PuzzleClassifier:
    """
    Class to classify puzzles in batches using Anthropic β API and Markdown templates.
    """

    def __init__(
        self,
        api_key: str,
        template_path: str,
        model: str,
        batch_size: int,
        max_tokens: int,
        temperature: float,
    ):
        """
        Initialize classifier with API key, template path, and model parameters.
        """
        self.client = anthropic.Anthropic(api_key=Anthropic_API_KEY)
        self.model = model
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.results: dict[int, dict] = {}
        self.lock = threading.Lock()

        tpl_p = Path(template_path)
        if not tpl_p.is_file():
            raise FileNotFoundError(f"Template not found: {tpl_p}")
        self.tpl_env = Environment(loader=FileSystemLoader(tpl_p.parent))
        self.template = self.tpl_env.get_template(tpl_p.name)

    def _build_content(self, item: dict, img_b64: str, mime: str) -> list[dict]:
        """
        Render the Markdown template with puzzle text and image.
        """
        options_block = "\n".join(
            f'"{k}": "{v}"' for k, v in item.get("options", {}).items()
        )
        md = self.template.render(
            prompt=item.get("prompt", ""),
            options_block=options_block,
            explanation=item.get("explanation", ""),
            correct_answer=item.get("correct_answer", ""),
        )
        pre, suf = md.split("<!--PUZZLE_SPLIT-->")
        return [
            {"type": "text", "text": pre},
            {
                "type": "image",
                "source": {"type": "base64", "media_type": mime, "data": img_b64},
            },
            {"type": "text", "text": suf},
        ]

    def _prepare_requests(self, items: list[dict]) -> tuple[list[dict], dict]:
        """
        Prepare batch API requests and map custom IDs to items.
        """
        reqs = []
        idmap = {}
        for it in items:
            pid = it["id"]
            if pid in self.results:
                continue
            try:
                b64, mime = encode_image(it["image"])
            except Exception as e:
                print(f"[skip] encode error {pid}: {e}")
                continue
            content = self._build_content(it, b64, mime)
            reqs.append(
                {
                    "custom_id": str(pid),
                    "params": {
                        "model": self.model,
                        "max_tokens": self.max_tokens,
                        "temperature": self.temperature,
                        "messages": [{"role": "user", "content": content}],
                    },
                }
            )
            idmap[str(pid)] = it
        return reqs, idmap

    def process(self, items: list[dict], save_interval: int, outfile: str):
        """
        Process items in batches: load, send requests, poll, parse responses, and save.
        """
        self._load(outfile)
        todo = [it for it in items if it["id"] not in self.results]
        if not todo:
            print("Nothing new.")
            return

        for idx in range(0, len(todo), self.batch_size):
            chunk = todo[idx : idx + self.batch_size]
            print(
                f"\nBatch {idx//self.batch_size+1}/"
                f"{(len(todo)+self.batch_size-1)//self.batch_size} size={len(chunk)}"
            )
            reqs, idmap = self._prepare_requests(chunk)
            if not reqs:
                print("No valid requests, skip.")
                continue
            try:
                batch = self.client.beta.messages.batches.create(requests=reqs)
                bid = batch.id
                while True:
                    status = self.client.beta.messages.batches.retrieve(
                        bid
                    ).processing_status
                    if status == "ended":
                        break
                    print("  status:", status)
                    time.sleep(8)
                for ent in self.client.beta.messages.batches.results(bid):
                    pid = int(ent.custom_id)
                    if hasattr(ent, "result") and hasattr(ent.result, "message"):
                        txt = "".join(
                            c.text
                            for c in ent.result.message.content
                            if c.type == "text"
                        )

                        def grab(tag: str) -> str:
                            m = re.search(rf"<{tag}>(.*?)</{tag}>", txt, re.DOTALL)
                            return m.group(1).strip() if m else ""

                        self.results[pid] = {
                            "id": pid,
                            "correct_answer": idmap[str(pid)].get("correct_answer", ""),
                            "puzzle_breakdown": grab("puzzle_breakdown"),
                            "question_type": grab("question_type"),
                            "knowledge_point": grab("knowledge_point"),
                            "raw_response": txt,
                        }
                    else:
                        err = getattr(ent, "error", "Unknown error")
                        self.results[pid] = {"id": pid, "error": str(err)}
                        print("  !! failed id", pid, err)
            except Exception as e:
                print("Batch error:", e)
            if (idx // self.batch_size + 1) % save_interval == 0:
                self._save(outfile)

        self._save(outfile)
        ok = sum(1 for v in self.results.values() if "error" not in v)
        print(f"Done. total={len(self.results)} ok={ok} fail={len(self.results)-ok}")

    def _load(self, path: str):
        """
        Load existing results from a JSON file.
        """
        if os.path.exists(path) and os.path.getsize(path):
            try:
                for rec in json.load(open(path, "r", encoding="utf-8")):
                    self.results[rec["id"]] = rec
                print("Resumed", len(self.results), "items.")
            except Exception as e:
                print("⚠ load error:", e)

    def _save(self, path: str):
        """
        Save current results to a JSON file in sorted order.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                sorted(self.results.values(), key=lambda r: r["id"]),
                f,
                ensure_ascii=False,
                indent=2,
            )
        print("Saved", len(self.results), "→", path)


def main(args):
    """
    Main entry point: load input data, resolve image paths, and run classification.
    """
    data = json.load(open(args.input, "r", encoding="utf-8"))
    if isinstance(data, dict):
        data = [data]

    json_dir = Path(args.input).parent
    for it in data:
        img = it.get("image", "")
        if img and not os.path.isabs(img):
            p = json_dir / img
            if p.exists():
                it["image"] = str(p)

    if args.image_dir:
        for it in data:
            if not os.path.exists(it["image"]):
                cand = Path(args.image_dir) / Path(it["image"]).name
                if cand.exists():
                    it["image"] = str(cand)

    classifier = PuzzleClassifier(
        api_key=args.api_key,
        template_path=args.template,
        model=args.model,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    classifier.process(data, args.save_interval, args.output)


if __name__ == "__main__":
    args = parse_args()
    main(args)
