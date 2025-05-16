#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rank rules by calling an external API with retries, concurrency,
exponential back-off, and robust checkpointing.
"""

import argparse
import json
import os
import pickle
import re
import signal
import sys
import time
import threading
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed

from together import Together
from jinja2 import Environment, FileSystemLoader
from tqdm import tqdm

# -------------------- API key discovery --------------------
try:
    from api_config import TOGETHER_API_KEY as API_KEY
except Exception:
    API_KEY = os.getenv("TOGETHER_API_KEY", "")


# -------------------- CLI argument parsing --------------------
def parse_args() -> argparse.Namespace:
    """Define and parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", default="./data/step2/2.3_rules.json")
    parser.add_argument("--output_json", default="./data/step2/2.6_rules_scoring.json")
    parser.add_argument("--api_key", default=API_KEY)
    parser.add_argument("--use_api", type=bool, default=True)
    parser.add_argument("--prompt_dir", default="Prompts")
    parser.add_argument("--max_workers", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument(
        "--checkpoint_path",
        default="./data/scoring_checkpoint/scoring_checkpoint.pkl",
    )
    parser.add_argument("--checkpoint_interval", type=int, default=1000)
    parser.add_argument("--max_retries", type=int, default=10)
    parser.add_argument("--base_delay", type=float, default=2.0)
    parser.add_argument("--backoff_factor", type=float, default=2.0)
    return parser.parse_args()


# -------------------- Core ranking system --------------------
class RuleRanker:
    """Ranking system with concurrency, retries, and checkpointing."""

    def __init__(
        self,
        *,
        input_json: str,
        output_json: str,
        api_key: str,
        use_api: bool,
        prompt_dir: str,
        max_workers: int,
        batch_size: int,
        checkpoint_path: str,
        checkpoint_interval: int,
        max_retries: int,
        base_delay: float,
        backoff_factor: float,
    ):

        # ---- config ----
        self.input_json = input_json
        self.output_json = output_json
        self.use_api = use_api and bool(api_key)
        self.api_key = api_key
        self.prompt_dir = prompt_dir
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.checkpoint_path = checkpoint_path
        self.checkpoint_interval = checkpoint_interval
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.backoff_factor = backoff_factor

        # ---- API client (optional) ----
        self.client: Optional[Together] = None
        if self.use_api:
            try:
                self.client = Together(api_key=self.api_key)
            except Exception:
                print(
                    "[Warn] Could not init Together client; switching to offline mode."
                )
                self.use_api = False

        # ---- data & counters ----
        self.rules: List[Dict] = []
        self.current_index = 0
        self.api_calls = 0
        self.success_count = 0
        self.fail_count = 0
        self.lock = threading.Lock()  # protect counters

        # ---- heavy I/O ----
        self._load_input()
        self.prompt_base = self._load_template()
        self._load_checkpoint()

        # ---- signals ----
        self._setup_signal_handlers()

    # ---------- I/O & checkpoint -------------------------------------------
    def _load_input(self):
        with open(self.input_json, encoding="utf-8") as f:
            self.rules = json.load(f)

    def _save_output(self):
        with open(self.output_json, "w", encoding="utf-8") as f:
            json.dump(self.rules, f, ensure_ascii=False, indent=2)

    def _load_checkpoint(self):
        if not os.path.exists(self.checkpoint_path):
            return
        try:
            with open(self.checkpoint_path, "rb") as f:
                state = pickle.load(f)
            self.current_index = state["current_index"]
            self.rules = state["rules"]
            self.api_calls = state["api_calls"]
            self.success_count = state["success_count"]
            self.fail_count = state["fail_count"]
            print(f"[Resume] checkpoint at index {self.current_index}")
        except Exception:
            print("[Warn] checkpoint corrupt, starting fresh.")

    def _save_checkpoint(self):
        Path(self.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        state = {
            "current_index": self.current_index,
            "rules": self.rules,
            "api_calls": self.api_calls,
            "success_count": self.success_count,
            "fail_count": self.fail_count,
        }
        with open(self.checkpoint_path, "wb") as f:
            pickle.dump(state, f)
        with open(
            Path(self.checkpoint_path).with_suffix(".json"), "w", encoding="utf-8"
        ) as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    # ---------- signals ----------------------------------------------------
    def _setup_signal_handlers(self):
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)

    def _handle_interrupt(self, *_):
        print("\n[Interrupt] saving checkpoint…")
        self._save_checkpoint()
        sys.exit(0)

    # ---------- prompt helpers --------------------------------------------
    def _load_template(self) -> str:
        env = Environment(loader=FileSystemLoader(self.prompt_dir))
        return env.get_template("step2.6_scoring.md").render()

    def _generate_prompt(self, rule: Dict) -> str:
        text = "\n".join(rule.get("rule_content", []))
        return self.prompt_base.replace("{RULE}", text)

    # ---------- API & parsing ---------------------------------------------
    def _call_api(self, prompt: str) -> Optional[str]:
        if not self.use_api:
            return None
        for attempt in range(self.max_retries):
            try:
                with self.lock:
                    self.api_calls += 1
                resp = self.client.chat.completions.create(
                    model="deepseek-ai/DeepSeek-V3",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                )
                return resp.choices[0].message.content
            except Exception as e:
                time.sleep(self.base_delay * (self.backoff_factor**attempt))
                if attempt == self.max_retries - 1:
                    print(f"[Warn] API failed permanently: {e}")
        return None

    _FS = re.compile(r"<format_score>\s*(\d{1,2})\s*</format_score>", re.I | re.S)
    _CQ = re.compile(r"<content_quality>\s*(\d{1,2})\s*</content_quality>", re.I | re.S)
    _FE = re.compile(r"<feasibility>\s*(\d{1,2})\s*</feasibility>", re.I | re.S)

    @classmethod
    def _parse_scores(cls, text: str) -> Optional[Tuple[int, int, int]]:
        try:
            return (
                int(cls._FS.search(text).group(1)),
                int(cls._CQ.search(text).group(1)),
                int(cls._FE.search(text).group(1)),
            )
        except Exception:
            return None

    def _process_one(self, rule: Dict) -> Optional[Tuple[int, int, int]]:
        prompt = self._generate_prompt(rule)
        for _ in range(3):  # parsing retries
            res = self._call_api(prompt)
            if res:
                scores = self._parse_scores(res)
                if scores:
                    return scores
        return None

    # ---------- main driver -----------------------------------------------
    def rank_all_rules(self):
        total = len(self.rules)
        pbar = tqdm(total=total, initial=self.current_index, desc="Scoring rules")

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            while self.current_index < total:
                start, end = self.current_index, min(
                    self.current_index + self.batch_size, total
                )
                futures = {
                    ex.submit(self._process_one, self.rules[i]): i
                    for i in range(start, end)
                }

                for fut in as_completed(futures):
                    idx = futures[fut]
                    scores = fut.result()
                    if scores:
                        fs, cq, fe = scores
                        self.rules[idx]["format_score"] = fs
                        self.rules[idx]["content_quality_score"] = cq
                        self.rules[idx]["feasibility_score"] = fe
                        self.success_count += 1
                    else:
                        self.rules[idx].update(
                            format_score=-1,
                            content_quality_score=-1,
                            feasibility_score=-1,
                        )
                        self.fail_count += 1

                    pbar.update(1)  # ← 立即推进 1

                self.current_index = end

                if self.current_index % self.checkpoint_interval == 0:
                    self._save_checkpoint()

        pbar.close()
        self._save_checkpoint()
        self._save_output()
        print(
            f"[Done] success={self.success_count}  fail={self.fail_count}  api_calls={self.api_calls}"
        )


# -------------------- CLI entry --------------------------------------------
def main():
    args = parse_args()
    RuleRanker(
        input_json=args.input_json,
        output_json=args.output_json,
        api_key=args.api_key,
        use_api=args.use_api,
        prompt_dir=args.prompt_dir,
        max_workers=args.max_workers,
        batch_size=args.batch_size,
        checkpoint_path=args.checkpoint_path,
        checkpoint_interval=args.checkpoint_interval,
        max_retries=args.max_retries,
        base_delay=args.base_delay,
        backoff_factor=args.backoff_factor,
    ).rank_all_rules()


if __name__ == "__main__":
    main()
