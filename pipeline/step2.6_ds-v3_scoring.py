"""
Rank rules by calling an external API with retries and checkpointing.
"""

import argparse
import json
import os
import pickle
import random
import re
import signal
import sys
import time
import concurrent.futures
from pathlib import Path
from together import Together
from jinja2 import Environment, FileSystemLoader
from tqdm import tqdm


def parse_args():
    """
    Define and parse command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str, default="")
    parser.add_argument("--output_json", type=str, default="")
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--use_api", type=bool, default=False)
    parser.add_argument("--prompt_dir", type=str, default="Prompts")
    parser.add_argument("--max_workers", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--checkpoint_path", type=str, default="rank_checkpoint.pkl")
    parser.add_argument("--checkpoint_interval", type=int, default=1000)
    parser.add_argument("--max_retries", type=int, default=10)
    parser.add_argument("--base_delay", type=float, default=2.0)
    parser.add_argument("--backoff_factor", type=float, default=2.0)
    return parser.parse_args()


class RuleRanker:
    """
    System for ranking rules via API calls with exponential backoff and checkpoint support.
    """

    def __init__(
        self,
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
        """
        Initialize RuleRanker with configuration parameters.
        """
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

        self.client = None
        if self.use_api:
            try:
                self.client = Together(api_key=self.api_key)
            except Exception:
                self.use_api = False

        self.rules = []
        self.current_index = 0
        self.api_calls = 0
        self.success_count = 0
        self.fail_count = 0

        self._load_input()
        self._load_checkpoint()
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """
        Setup signal handlers for graceful interruption.
        """
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)

    def _handle_interrupt(self, sig, frame):
        """
        Handle interrupt by saving checkpoint and exiting.
        """
        self._save_checkpoint()
        sys.exit(0)

    def _load_input(self):
        """
        Load rules from the input JSON file.
        """
        with open(self.input_json, "r", encoding="utf-8") as f:
            self.rules = json.load(f)

    def _save_output(self):
        """
        Save ranked rules to the output JSON file.
        """
        with open(self.output_json, "w", encoding="utf-8") as f:
            json.dump(self.rules, f, ensure_ascii=False, indent=2)

    def _load_checkpoint(self):
        """
        Load progress from checkpoint if available.
        """
        if os.path.exists(self.checkpoint_path):
            try:
                with open(self.checkpoint_path, "rb") as f:
                    state = pickle.load(f)
                self.current_index = state["current_index"]
                self.rules = state["rules"]
                self.api_calls = state["api_calls"]
                self.success_count = state["success_count"]
                self.fail_count = state["fail_count"]
            except Exception:
                pass

    def _save_checkpoint(self):
        """
        Save current progress to checkpoint files (pickle and JSON).
        """
        state = {
            "current_index": self.current_index,
            "rules": self.rules,
            "api_calls": self.api_calls,
            "success_count": self.success_count,
            "fail_count": self.fail_count,
        }
        with open(self.checkpoint_path, "wb") as f:
            pickle.dump(state, f)
        json_path = Path(self.checkpoint_path).with_suffix(".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def _load_template(self) -> str:
        """
        Load and return the ranking prompt template.
        """
        env = Environment(loader=FileSystemLoader(self.prompt_dir))
        template = env.get_template("rank.md")
        return template.render()

    def _generate_prompt(self, rule: dict) -> str:
        """
        Insert rule content into the prompt template.
        """
        base = self._load_template()
        content = rule.get("rule_content", [])
        text = "\n".join(content) if isinstance(content, list) else str(content)
        return base.replace("{RULE}", text)

    def _call_api(self, prompt: str) -> str | None:
        """
        Call API with retries and exponential backoff.
        """
        for attempt in range(self.max_retries):
            try:
                self.api_calls += 1
                resp = self.client.chat.completions.create(
                    model="deepseek-ai/DeepSeek-V3",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                )
                return resp.choices[0].message.content
            except Exception:
                if attempt < self.max_retries - 1:
                    time.sleep(self.base_delay * (self.backoff_factor**attempt))
        return None

    def _parse_scores(self, response: str) -> tuple[int, int, int] | None:
        """
        Extract format, content quality, and feasibility scores from response.
        """
        try:
            fs = int(
                re.search(r"<format_score>\s*(\d+)\s*</format_score>", response).group(
                    1
                )
            )
            cq = int(
                re.search(
                    r"<content_quality>\s*(\d+)\s*</content_quality>", response
                ).group(1)
            )
            fe = int(
                re.search(r"<feasibility>\s*(\d+)\s*</feasibility>", response).group(1)
            )
            return fs, cq, fe
        except Exception:
            return None

    def _process_one(self, rule: dict) -> tuple[int, int, int] | None:
        """
        Generate prompt and obtain scores for a single rule.
        """
        for _ in range(3):
            prompt = self._generate_prompt(rule)
            resp = self._call_api(prompt)
            if resp:
                scores = self._parse_scores(resp)
                if scores:
                    return scores
        return None

    def rank_all_rules(self):
        """
        Rank all rules, processing in batches with concurrency and checkpointing.
        """
        total = len(self.rules)
        pbar = tqdm(total=total, initial=self.current_index, desc="Scoring rules")
        while self.current_index < total:
            end = min(self.current_index + self.batch_size, total)
            batch = [(i, self.rules[i]) for i in range(self.current_index, end)]
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_workers
            ) as exe:
                futures = {exe.submit(self._process_one, r): i for i, r in batch}
                for fut in concurrent.futures.as_completed(futures):
                    idx = futures[fut]
                    res = fut.result()
                    if res:
                        fs, cq, fe = res
                        self.rules[idx]["format_score"] = fs
                        self.rules[idx]["content_quality_score"] = cq
                        self.rules[idx]["feasibility_score"] = fe
                        self.success_count += 1
                    else:
                        self.rules[idx]["format_score"] = -1
                        self.rules[idx]["content_quality_score"] = -1
                        self.rules[idx]["feasibility_score"] = -1
                        self.fail_count += 1
            self.current_index = end
            pbar.update(end - self.current_index + (end - self.current_index))
            if self.current_index % self.checkpoint_interval < self.batch_size:
                self._save_checkpoint()
        pbar.close()
        self._save_checkpoint()
        self._save_output()


def main():
    """
    Entry point: parse args and start ranking.
    """
    args = parse_args()
    ranker = RuleRanker(
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
    )
    ranker.rank_all_rules()


if __name__ == "__main__":
    main()
