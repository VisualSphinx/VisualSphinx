"""
Batch extractor that calls the Anthropic Claude batch API using Markdown templates
and image insertion markers. Renders prompts and processes puzzles in batches.
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
    parser.add_argument("--input", default="./data/step1_filtered/1.5_questions.json")
    parser.add_argument("--output", default="./data/step1_filtered/1.6_rules_raw.json")
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--example_image", default="./data/step1/images/image_6.png")
    parser.add_argument("--template", default="./Prompts/step1.6_abstract.md")
    parser.add_argument("--batch_size", type=int, default="20")
    parser.add_argument("--save_interval", type=int, default="20")
    parser.add_argument("--max_tokens", type=int, default="20000")
    parser.add_argument("--temperature", type=float, default="1.0")
    parser.add_argument("--model", type=str, default="claude-3-7-sonnet-20250219")
    return parser.parse_args()


class BatchRegularityExtractor:
    """
    Batch extractor that processes items by rendering prompts with a Markdown template,
    inserting example and puzzle images, and calling the Anthropic API in batches.
    """

    def __init__(
        self,
        api_key: str,
        template_path: str,
        example_image_path: str,
        model: str = "",
        batch_size: int = "",
        max_tokens: int = "",
        temperature: float = "",
    ):
        """
        Initialize extractor with API key, template path, example image path,
        and model parameters.
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

        self.example_b64, self.example_mime = self._encode_image(example_image_path)

    @staticmethod
    def _encode_image(path: str) -> tuple[str, str]:
        """
        Encode image from path to base64 and return (data, mime type).
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")
        mime, _ = mimetypes.guess_type(path)
        mime = mime or ("image/png" if path.lower().endswith(".png") else "image/jpeg")
        data = base64.b64encode(Path(path).read_bytes()).decode("utf-8")
        return data, mime

    def _build_content(
        self, item: dict, puzzle_b64: str, puzzle_mime: str
    ) -> list[dict]:
        """
        Build the content list by rendering the template and inserting example and puzzle images.
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
        pre, rest = md.split("<!--EXAMPLE_SPLIT-->")
        mid, suf = rest.split("<!--PUZZLE_SPLIT-->")
        return [
            {"type": "text", "text": pre},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": self.example_mime,
                    "data": self.example_b64,
                },
            },
            {"type": "text", "text": mid},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": puzzle_mime,
                    "data": puzzle_b64,
                },
            },
            {"type": "text", "text": suf},
        ]

    def _prepare_batch_requests(self, items: list[dict]):
        """
        Prepare batch request payloads and map custom IDs to items.
        """
        reqs, id_map = [], {}
        for itm in items:
            pid = itm["id"]
            if pid in self.results:
                continue
            try:
                p_b64, p_mime = self._encode_image(itm["image"])
            except Exception as e:
                print(f"[skip] encode image error {pid}: {e}")
                continue
            content = self._build_content(itm, p_b64, p_mime)
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
            id_map[str(pid)] = itm
        return reqs, id_map

    def process_batch(self, items: list[dict], save_interval: int, out_path: str):
        """
        Process items in batches: load existing results, send requests, poll for completion,
        parse responses, and save at intervals.
        """
        self._load(out_path)
        todo = [x for x in items if x["id"] not in self.results]
        if not todo:
            print("Nothing to do.")
            return

        for idx in range(0, len(todo), self.batch_size):
            chunk = todo[idx : idx + self.batch_size]
            print(
                f"\nBatch {idx//self.batch_size+1}/"
                f"{(len(todo)+self.batch_size-1)//self.batch_size} size={len(chunk)}"
            )
            reqs, id_map = self._prepare_batch_requests(chunk)
            if not reqs:
                print("No valid requests, skip.")
                continue

            try:
                batch = self.client.beta.messages.batches.create(requests=reqs)
                bid = batch.id
                print("Batch id:", bid)

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

                        def take(tag):
                            m = re.search(rf"<{tag}>(.*?)</{tag}>", txt, re.DOTALL)
                            return m.group(1).strip() if m else ""

                        self.results[pid] = {
                            "id": pid,
                            "correct_answer": id_map[str(pid)].get(
                                "correct_answer", ""
                            ),
                            "detailed_analysis": take("detailed_analysis"),
                            "puzzle_breakdown": take("puzzle_breakdown"),
                            "key_points": [
                                ln.strip().lstrip("- ")
                                for ln in take("key_points").splitlines()
                                if ln.strip() and ln.strip() != "-"
                            ],
                            "raw_response": txt,
                        }
                    else:
                        err = getattr(ent, "error", "Unknown error")
                        self.results[pid] = {"id": pid, "error": str(err)}
                        print("  !! failed id", pid, err)
            except Exception as e:
                print("Batch error:", e)

            if (idx // self.batch_size + 1) % save_interval == 0:
                self._save(out_path)

        self._save(out_path)
        ok = sum(1 for v in self.results.values() if "error" not in v)
        print(f"Done. total={len(self.results)} ok={ok} failed={len(self.results)-ok}")

    def _load(self, path: str):
        """
        Load existing results from the output JSON file if present.
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
        Save current results to the output JSON file in sorted order.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                sorted(self.results.values(), key=lambda r: r["id"]),
                f,
                ensure_ascii=False,
                indent=2,
            )
        print("Saved", len(self.results), "items →", path)


def main(args):
    """
    Main entry point: load input data, resolve image paths, and run the extractor.
    """
    data = json.load(open(args.input, "r", encoding="utf-8"))
    if isinstance(data, dict):
        data = [data]

    json_dir = Path(args.input).parent
    for itm in data:
        img = itm.get("image", "")
        if img and not os.path.isabs(img):
            abs_path = json_dir / img
            if abs_path.exists():
                itm["image"] = str(abs_path)
            else:
                print(f"⚠ Image not found even after join: {abs_path}")

    extractor = BatchRegularityExtractor(
        api_key=args.api_key,
        template_path=args.template,
        example_image_path=args.example_image,
        model=args.model,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    extractor.process_batch(data, args.save_interval, args.output)


if __name__ == "__main__":
    args = parse_args()
    main(args)
