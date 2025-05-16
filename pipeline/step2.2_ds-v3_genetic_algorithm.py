"""
Tag‑based rule evolution system with genetic operators, parallel Together API
calls, checkpoint/resume support, and island‑model migration.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import re
import signal
import sys
import threading
import time
import concurrent.futures
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple
from tqdm import tqdm


from jinja2 import Environment, FileSystemLoader
from together import Together

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

try:
    from api_config import TOGETHER_API_KEY as API_KEY
except Exception:
    API_KEY = os.getenv("TOGETHER_API_KEY", "")


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:  # noqa: WPS231 — simple factory
    """Parse command‑line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments ready for consumption by :pyfunc:`main`.
    """
    p = argparse.ArgumentParser("Tag‑based GA for rule evolution")
    # core GA & I/O
    p.add_argument(
        "--input_json_file",
        type=str,
        default="./data/step2/2.1_eight_classes.json",
    )  # or ./data/step2/2.1_others.json
    p.add_argument(
        "--output_filename", type=str, default="./data/step2/2.2_evolved_rules.json"
    )  # or ./data/step2/2.2_evolved_rules_others.json

    p.add_argument("--generations", type=int, default=9)
    p.add_argument("--separate_output", action="store_true")
    # hyper‑parameters
    p.add_argument("--mutation_probability", type=float, default=0.15)
    p.add_argument("--migration_interval", type=int, default=3)
    p.add_argument("--elite_percentage", type=float, default=0.1)
    p.add_argument("--growth_factor", type=float, default=1.20)
    # runtime settings
    p.add_argument("--use_api", action="store_true")
    p.add_argument("--max_workers", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=120)
    # checkpointing
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--checkpoint_interval", type=int, default=1)
    p.add_argument("--resume_from", type=str, default="")
    p.add_argument("--list_checkpoints", action="store_true")
    # prompts / keys
    p.add_argument("--prompt_dir", type=str, default="Prompts")
    p.add_argument("--api_key", type=str, default=API_KEY)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class TagBasedRuleEvolutionSystem:
    """Island‑model genetic algorithm specialised for rule strings."""

    # ‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑
    # Construction / initialisation helpers
    # ‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑

    def __init__(
        self,
        input_json_file: str,
        mutation_probability: float,
        migration_interval: int,
        elite_percentage: float,
        growth_factor: float,
        api_key: str,
        prompt_dir: str,
        use_api: bool,
        max_workers: int,
        batch_size: int,
        checkpoint_dir: str,
        checkpoint_interval: int,
    ) -> None:  # noqa: WPS211 — long but clear
        self.input_json_file = input_json_file
        self.mutation_probability = mutation_probability
        self.migration_interval = migration_interval
        self.elite_percentage = elite_percentage
        self.growth_factor = growth_factor
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval
        self.is_resumed = False

        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # API + template env --------------------------------------------------
        self.api_key = api_key
        self.use_api = use_api and bool(api_key)
        self.client: Together | None = None
        self.template_env: Environment | None = None
        self.prompt_dir = prompt_dir
        if self.use_api:
            try:
                self.client = Together(api_key=api_key)
                if Path(prompt_dir).exists():
                    self.template_env = Environment(loader=FileSystemLoader(prompt_dir))
            except Exception:  # fall back silently — simulation mode
                self.use_api = False

        # Evolution state -----------------------------------------------------
        self.next_id: int = 0
        self.all_rules: Dict[int, Dict[str, Any]] = {}
        self.tag_to_island: Dict[Tuple[str, str], int] = {}
        self.islands: List[List[Dict[str, Any]]] = []
        self.island_count: int = 0
        self.used_as_parent: set[int] = set()
        self.generation_stats: defaultdict[int, Dict[str, Any]] = defaultdict(dict)
        self.current_generation: int = 0
        self.question_types: set[str] = set()
        self.knowledge_points: set[str] = set()
        self.tag_combinations: set[Tuple[str, str]] = set()

        # API monitoring ------------------------------------------------------
        self.api_calls: Dict[str, Any] = defaultdict(int)
        self.api_calls["total_time"] = 0.0

        self._setup_signal_handlers()
        self._load_and_initialize_seeds()

    # ---------------------------------------------------------------------
    # API helpers (with exponential back‑off)
    # ---------------------------------------------------------------------

    def _execute_api_call_with_exponential_backoff(
        self,
        prompt: str,
        max_retries: int = 10,
        base_delay: float = 2.0,
        backoff_factor: float = 2.0,
    ) -> str:
        """Wrapper for Together API with exponential back‑off."""
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="deepseek-ai/DeepSeek-V3",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                )
                content = response.choices[0].message.content
                if content and content.strip():
                    return content
            except Exception as exc:  # noqa: WPS440 — broad catch → simulation
                print(f"[API] Attempt {attempt+1} failed: {exc}")
            if attempt < max_retries - 1:
                time.sleep(base_delay * (backoff_factor**attempt))
        return "__FALLBACK__"

    # ---------------------------------------------------------------------
    # Signal handling
    # ---------------------------------------------------------------------

    def _setup_signal_handlers(self) -> None:  # noqa: D401
        """Install SIGINT/SIGTERM handlers to allow graceful exit."""
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)

    def _handle_interrupt(self, *_: Any) -> None:  # noqa: D401
        """Save checkpoint and exit upon external interrupt."""
        print("\n[INTERRUPT] Saving checkpoint…")
        self.save_checkpoint(f"interrupted_gen{self.current_generation}")
        sys.exit(0)

    # ---------------------------------------------------------------------
    # Checkpointing helpers
    # ---------------------------------------------------------------------

    def save_checkpoint(self, name: str | None = None) -> None:  # noqa: WPS231
        """Serialise internal state to *checkpoint_dir* (*pickle* + JSON index)."""
        ts = time.strftime("%Y%m%d_%H%M%S")
        name = name or f"checkpoint_gen{self.current_generation}_{ts}"
        pkl_path = Path(self.checkpoint_dir) / f"{name}.pkl"
        state = {  # pure python primitive container ➜ pickle‑safe
            "mutation_probability": self.mutation_probability,
            "migration_interval": self.migration_interval,
            "elite_percentage": self.elite_percentage,
            "growth_factor": self.growth_factor,
            "max_workers": self.max_workers,
            "batch_size": self.batch_size,
            "next_id": self.next_id,
            "all_rules": self.all_rules,
            "islands": self.islands,
            "island_count": self.island_count,
            "tag_to_island": self.tag_to_island,
            "used_as_parent": self.used_as_parent,
            "generation_stats": dict(self.generation_stats),
            "current_generation": self.current_generation,
            "question_types": self.question_types,
            "knowledge_points": self.knowledge_points,
            "tag_combinations": self.tag_combinations,
            "api_calls": dict(self.api_calls),
            "input_json_file": self.input_json_file,
            "checkpoint_interval": self.checkpoint_interval,
        }
        with pkl_path.open("wb") as fp:
            pickle.dump(state, fp)
        # add human‑readable index
        index_path = pkl_path.with_name(f"{name}_index.json")
        with index_path.open("w", encoding="utf‑8") as fp:
            json.dump(
                {
                    "timestamp": ts,
                    "generation": self.current_generation,
                    "total_rules": len(self.all_rules),
                    "island_count": self.island_count,
                    "api_calls": dict(self.api_calls),
                    "checkpoint_file": str(pkl_path),
                },
                fp,
                ensure_ascii=False,
                indent=2,
            )
        print(f"[CHECKPOINT] Saved → {pkl_path}")

    @classmethod
    def load_checkpoint(
        cls,
        path: str,
        api_key: str,
        prompt_dir: str,
        use_api: bool,
    ) -> "TagBasedRuleEvolutionSystem":
        """Restore system from *path* pickle; API key / settings may differ."""
        with Path(path).open("rb") as fp:
            state: Dict[str, Any] = pickle.load(fp)
        inst = cls(
            input_json_file=state["input_json_file"],
            mutation_probability=state["mutation_probability"],
            migration_interval=state["migration_interval"],
            elite_percentage=state["elite_percentage"],
            growth_factor=state["growth_factor"],
            api_key=api_key,
            prompt_dir=prompt_dir,
            use_api=use_api,
            max_workers=state["max_workers"],
            batch_size=state["batch_size"],
            checkpoint_dir=Path(path).parent.as_posix(),
            checkpoint_interval=state.get("checkpoint_interval", 1),
        )
        # overwrite dynamic state ------------------------------------------------
        inst.next_id = state["next_id"]
        inst.all_rules = state["all_rules"]
        inst.islands = state["islands"]
        inst.island_count = state["island_count"]
        inst.tag_to_island = state["tag_to_island"]
        inst.used_as_parent = state["used_as_parent"]
        inst.generation_stats = defaultdict(dict, state["generation_stats"])
        inst.current_generation = state["current_generation"]
        inst.question_types = state["question_types"]
        inst.knowledge_points = state["knowledge_points"]
        inst.tag_combinations = state["tag_combinations"]
        inst.api_calls = defaultdict(int, state["api_calls"])
        inst.is_resumed = True
        print(f"[CHECKPOINT] Resumed → Gen {inst.current_generation}")
        return inst

    @classmethod
    def list_checkpoints(cls, directory: str) -> List[Dict[str, Any]]:
        """Return list of checkpoints found in *directory* sorted by gen desc."""
        dir_path = Path(directory)
        if not dir_path.exists():
            return []
        cps = []
        for idx_file in dir_path.glob("*_index.json"):
            try:
                idx_data = json.load(idx_file.open("r", encoding="utf‑8"))
                pkl_path = Path(idx_data.get("checkpoint_file", ""))
                if pkl_path.exists():
                    cps.append(
                        {
                            "generation": idx_data.get("generation", 0),
                            "total_rules": idx_data.get("total_rules", 0),
                            "timestamp": idx_data.get("timestamp", ""),
                            "checkpoint_file": pkl_path.as_posix(),
                        }
                    )
            except Exception:  # corrupt file → skip
                continue
        return sorted(cps, key=lambda d: d["generation"], reverse=True)

    # ---------------------------------------------------------------------
    # Seed initialisation
    # ---------------------------------------------------------------------

    def _load_and_initialize_seeds(self) -> None:  # noqa: WPS231
        """Load seeds from *input_json_file* and distribute them into islands."""
        if self.is_resumed:
            return
        data = json.load(Path(self.input_json_file).open("r", encoding="utf‑8"))
        tag_groups: defaultdict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(
            list
        )
        for seed in data:
            qt = seed.get("question_type", "unknown")
            kp = seed.get("knowledge_point", "unknown")
            combo = (qt, kp)
            self.question_types.add(qt)
            self.knowledge_points.add(kp)
            self.tag_combinations.add(combo)
            content = seed.get("seed_rule", [])
            if not isinstance(content, list):
                content = [content]
            rule = {
                "id": seed.get("seed_id", self.next_id),
                "rule_content": content,
                "generation": 0,
                "parents": [],
                "mutated": False,
                "times_used": 0,
                "ancestry_depth": 0,
                "creation_method": "seed",
                "question_type": qt,
                "knowledge_point": kp,
                "tag_combination": combo,
            }
            tag_groups[combo].append(rule)
            self.next_id = max(self.next_id, rule["id"] + 1)
            self.all_rules[rule["id"]] = rule
        # build islands ----------------------------------------------------
        self.island_count = len(tag_groups)
        self.islands = [[] for _ in range(self.island_count)]
        for idx, (combo, rules) in enumerate(tag_groups.items()):
            self.tag_to_island[combo] = idx
            self.islands[idx] = rules
        self.generation_stats[0] = {
            "total_rules": len(self.all_rules),
            "island_sizes": [len(island) for island in self.islands],
        }

    # ---------------------------------------------------------------------
    # Fitness & parent selection helpers
    # ---------------------------------------------------------------------

    def _fitness(self, rule: Dict[str, Any]) -> float:
        """Composite heuristic mirroring Example 1."""
        usage_score = 1.0 / (1 + 0.2 * rule.get("times_used", 0))
        ancestry_score = min(1.0, 0.5 + 0.1 * rule.get("ancestry_depth", 0))
        random_score = random.uniform(0.8, 1.0)
        mutation_bonus = 1.1 if rule.get("mutated", False) else 1.0
        return (
            0.4 * usage_score + 0.4 * ancestry_score + 0.2 * random_score
        ) * mutation_bonus

    def _select_elite(self, pop: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        count = max(1, int(len(pop) * self.elite_percentage))
        return sorted(pop, key=self._fitness, reverse=True)[:count]

    def _select_parent(
        self, pop: List[Dict[str, Any]]
    ) -> Dict[str, Any]:  # noqa: WPS231
        unused = [r for r in pop if r["times_used"] == 0]
        if unused and random.random() < 0.4:
            chosen = random.choice(unused)
        else:
            weights = [1.0 / (r["times_used"] + 1) for r in pop]
            s = sum(weights)
            probs = [w / s for w in weights]
            chosen = random.choices(pop, weights=probs, k=1)[0]
        chosen["times_used"] += 1
        self.used_as_parent.add(chosen["id"])
        return chosen

    # ---------------------------------------------------------------------
    # Prompt helpers (template or hard‑coded fallback)
    # ---------------------------------------------------------------------

    def _get_template(self, name: str) -> str:
        if self.template_env is None:
            return ""  # fallback handled later
        try:
            return self.template_env.get_template(f"{name}.md").render()
        except Exception:
            return ""

    def _prepare_crossover_prompt(
        self, p1: Dict[str, Any], p2: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        tpl = self._get_template("step2.2_crossover")
        r1 = "\n- " + "\n- ".join(p1["rule_content"])
        r2 = "\n- " + "\n- ".join(p2["rule_content"])
        prompt = tpl.replace("{FIRST_RULE_SET}", r1).replace("{SECOND_RULE_SET}", r2)
        qt, kp = (
            (p1["question_type"], p1["knowledge_point"])
            if random.random() < 0.5
            else (p2["question_type"], p2["knowledge_point"])
        )
        return prompt, {
            "parents": [p1["id"], p2["id"]],
            "question_type": qt,
            "knowledge_point": kp,
            "tag_combination": (qt, kp),
            "ancestry_depth": max(
                p1.get("ancestry_depth", 0), p2.get("ancestry_depth", 0)
            )
            + 1,
        }

    def _prepare_mutation_prompt(
        self, rule: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        tpl = self._get_template("step2.2_mutation")
        rc = "\n- " + "\n- ".join(rule["rule_content"])
        prompt = tpl.replace("{RULE_SET}", rc)
        return prompt, {
            "original_id": rule.get("id", -1),
            "question_type": rule["question_type"],
            "knowledge_point": rule["knowledge_point"],
            "tag_combination": rule["tag_combination"],
            "ancestry_depth": rule.get("ancestry_depth", 0),
        }

    # ---------------------------------------------------------------------
    # Response parsing helpers
    # ---------------------------------------------------------------------

    _CROSS_PATTERN = re.compile(r"<crossover_rules>(.*?)</crossover_rules>", re.DOTALL)
    _MUT_PATTERN = re.compile(r"<mutated_rules>(.*?)</mutated_rules>", re.DOTALL)

    @staticmethod
    def _split_rules(block: str) -> List[str]:
        out = []
        for line in block.split("\n"):
            line = re.sub(r"^[\-\d\.\s]+", "", line).strip()
            if line and not line.startswith(("#", "(")):
                out.append(line)
        return out

    def _extract_crossover(self, txt: str) -> List[str] | None:
        m = self._CROSS_PATTERN.search(txt or "")
        return self._split_rules(m.group(1)) if m else None

    def _extract_mutation(self, txt: str) -> List[str] | None:
        m = self._MUT_PATTERN.search(txt or "")
        return self._split_rules(m.group(1)) if m else None

    # ---------------------------------------------------------------------
    # Post‑processing helpers
    # ---------------------------------------------------------------------

    def _process_crossover_result(
        self, txt: str, info: Dict[str, Any]
    ) -> Dict[str, Any] | None:
        rules = self._extract_crossover(txt)
        if not rules:
            return None
        return {
            "rule_content": rules,
            "generation": self.current_generation,
            "parents": info["parents"],
            "mutated": False,
            "times_used": 0,
            "ancestry_depth": info["ancestry_depth"],
            "creation_method": "api_crossover",
            "question_type": info["question_type"],
            "knowledge_point": info["knowledge_point"],
            "tag_combination": info["tag_combination"],
        }

    def _process_mutation_result(
        self, txt: str, info: Dict[str, Any]
    ) -> Dict[str, Any] | None:
        rules = self._extract_mutation(txt)
        if not rules:
            return None
        return {
            "rule_content": rules,
            "generation": self.current_generation,
            "parents": [info["original_id"]] if info["original_id"] != -1 else [],
            "mutated": True,
            "times_used": 0,
            "ancestry_depth": info["ancestry_depth"],
            "creation_method": "api_mutation",
            "question_type": info["question_type"],
            "knowledge_point": info["knowledge_point"],
            "tag_combination": info["tag_combination"],
        }

    # ---------------------------------------------------------------------
    # Simulation fallbacks (identical logic to Example 1)
    # ---------------------------------------------------------------------

    def _simulate_crossover(
        self, p1: Dict[str, Any], p2: Dict[str, Any]
    ) -> Dict[str, Any]:  # noqa: WPS231
        ancestry_depth = (
            max(p1.get("ancestry_depth", 0), p2.get("ancestry_depth", 0)) + 1
        )
        qt, kp = (
            (p1["question_type"], p1["knowledge_point"])
            if random.random() < 0.5
            else (p2["question_type"], p2["knowledge_point"])
        )
        tag_combo = (qt, kp)
        new_rules = [f"sim_rule_{random.randint(1000, 9999)}"]
        for r in p1["rule_content"] + p2["rule_content"]:
            if r not in new_rules and random.random() < 0.7:
                new_rules.append(r)
        return {
            "rule_content": new_rules,
            "generation": self.current_generation,
            "parents": [p1["id"], p2["id"]],
            "mutated": False,
            "times_used": 0,
            "ancestry_depth": ancestry_depth,
            "creation_method": "sim_crossover",
            "question_type": qt,
            "knowledge_point": kp,
            "tag_combination": tag_combo,
        }

    def _simulate_mutation(self, rule: Dict[str, Any]) -> Dict[str, Any]:
        mtype = random.choices(["small", "medium", "large"], weights=[0.6, 0.3, 0.1])[0]
        new = rule.copy()
        content = new.get("rule_content", [])
        if isinstance(content, list) and content:
            if random.random() < 0.7:
                idx = random.randrange(len(content))
                content[idx] = f"{content[idx]}_{mtype}"
            else:
                content.append(f"new_{mtype}_rule")
        else:
            content = [f"new_{mtype}_rule"]
        new.update(
            {
                "rule_content": content,
                "mutated": True,
                "creation_method": f"sim_mutation_{mtype}",
                "parents": [rule.get("id")] if "id" in rule else [],
            }
        )
        return new

    # ---------------------------------------------------------------------
    # Batched API wrappers (crossover & mutation)
    # ---------------------------------------------------------------------

    def _retry_api_parse_crossover(
        self, prompt: str, info: Dict[str, Any]
    ) -> Dict[str, Any] | None:  # noqa: WPS231
        for _ in range(10):
            txt = self._execute_api_call_with_exponential_backoff(prompt)
            if txt == "__FALLBACK__":
                return None
            rule = self._process_crossover_result(txt, info)
            if rule:
                return rule
        return None

    def batch_api_crossover(
        self, tasks: List[Tuple[Dict[str, Any], Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:  # noqa: WPS231
        if not tasks:
            return []
        if not self.use_api:
            return [self._simulate_crossover(p1, p2) for p1, p2 in tasks]
        t0 = time.time()
        self.api_calls["parallel_batches"] += 1
        out = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            future_map = {
                ex.submit(
                    self._retry_api_parse_crossover,
                    *self._prepare_crossover_prompt(p1, p2),
                ): (p1, p2)
                for p1, p2 in tasks
            }
            for fut in concurrent.futures.as_completed(future_map):
                p1, p2 = future_map[fut]
                self.api_calls["crossover"] += 1
                try:
                    rule = fut.result()
                    if rule:
                        self.api_calls["success"] += 1
                        out.append(rule)
                    else:
                        out.append(self._simulate_crossover(p1, p2))
                        self.api_calls["fallback"] += 1
                except Exception:  # fallback safety
                    out.append(self._simulate_crossover(p1, p2))
                    self.api_calls["failed"] += 1
        self.api_calls["total_time"] += time.time() - t0
        return out

    def _retry_api_parse_mutation(
        self, prompt: str, info: Dict[str, Any]
    ) -> Dict[str, Any] | None:
        for _ in range(10):
            txt = self._execute_api_call_with_exponential_backoff(prompt)
            if txt == "__FALLBACK__":
                return None
            rule = self._process_mutation_result(txt, info)
            if rule:
                return rule
        return None

    def batch_api_mutation(
        self, tasks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:  # noqa: WPS231
        if not tasks:
            return []
        if not self.use_api:
            return [self._simulate_mutation(r) for r in tasks]
        t0 = time.time()
        self.api_calls["parallel_batches"] += 1
        out = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            future_map = {
                ex.submit(
                    self._retry_api_parse_mutation, *self._prepare_mutation_prompt(r)
                ): r
                for r in tasks
            }
            for fut in concurrent.futures.as_completed(future_map):
                orig = future_map[fut]
                self.api_calls["mutation"] += 1
                try:
                    rule = fut.result()
                    if rule:
                        self.api_calls["success"] += 1
                        out.append(rule)
                    else:
                        out.append(self._simulate_mutation(orig))
                        self.api_calls["fallback"] += 1
                except Exception:
                    out.append(self._simulate_mutation(orig))
                    self.api_calls["failed"] += 1
        self.api_calls["total_time"] += time.time() - t0
        return out

    # ---------------------------------------------------------------------
    # Island migration helper
    # ---------------------------------------------------------------------

    def migrate_between_islands(self) -> None:  # noqa: WPS231
        if self.island_count <= 1:
            return
        migrants_per_island = max(1, int(len(self.islands[0]) * 0.05))
        migrants: Dict[int, List[Dict[str, Any]]] = {}
        for idx, island in enumerate(self.islands):
            if len(island) > migrants_per_island:
                chosen = random.sample(island, migrants_per_island)
                migrants[idx] = chosen
                for r in chosen:
                    island.remove(r)
        indices = list(range(self.island_count))
        for src, group in migrants.items():
            dest_candidates = [i for i in indices if i != src]
            if not dest_candidates:
                continue
            dest = random.choice(dest_candidates)
            self.islands[dest].extend(group)
            print(f"[MIGRATE] {len(group)} rules: island {src} → {dest}")

    # ---------------------------------------------------------------------
    # One full GA generation
    # ---------------------------------------------------------------------

    def evolve_one_generation(self) -> Dict[str, Any]:  # noqa: WPS231
        start = time.time()
        self.current_generation += 1
        print(f"[GEN] === Generation {self.current_generation} ===")

        if self.current_generation % self.migration_interval == 0:
            self.migrate_between_islands()

        next_islands = [[] for _ in range(self.island_count)]
        crossover_tasks: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        mutation_candidates: List[Dict[str, Any]] = []

        sizes = [len(i) for i in self.islands]
        for idx, island in enumerate(self.islands):
            if not island:
                continue
            elites = self._select_elite(island)
            next_islands[idx].extend(elites)
            offspring_needed = int(sizes[idx] * self.growth_factor) - len(elites)
            offspring_needed = max(0, offspring_needed)
            for _ in range(offspring_needed):
                p1 = self._select_parent(island)
                alt = [r for r in island if r["id"] != p1["id"]] or island
                p2 = self._select_parent(alt)
                crossover_tasks.append((p1, p2))

        # crossover (batched)
        offspring, cross_done = [], 0
        cross_total = len(crossover_tasks)
        t_cross_start = time.time()

        bar = (
            tqdm(
                total=cross_total,
                desc=f"Gen{self.current_generation}-crossover",
                unit="rule",
            )
            if tqdm
            else None
        )

        for i in range(0, cross_total, self.batch_size):
            batch = crossover_tasks[i : i + self.batch_size]
            results = self.batch_api_crossover(batch)
            offspring.extend(results)
            cross_done += len(results)

            if bar:
                bar.update(len(results))
            else:
                elapsed = time.time() - t_cross_start
                speed = cross_done / elapsed if elapsed else 0
                print(
                    f"  [cross] {cross_done}/{cross_total} " f"| {speed:5.1f} r/s",
                    end="\r",
                )

        if bar:
            bar.close()
        else:
            print()

        for rule in offspring:
            rule["id"] = self.next_id
            self.all_rules[self.next_id] = rule
            self.next_id += 1
        self.save_checkpoint(f"gen{self.current_generation}_after_crossover")

        # gather mutation
        mutation_candidates = [
            r for r in offspring if random.random() < self.mutation_probability
        ]
        mut_total = len(mutation_candidates)
        mut_done = 0
        t_mut_start = time.time()

        bar = (
            tqdm(
                total=mut_total,
                desc=f"Gen{self.current_generation}-mutation",
                unit="rule",
            )
            if tqdm
            else None
        )
        for i in range(0, len(mutation_candidates), self.batch_size):
            batch = mutation_candidates[i : i + self.batch_size]
            results = self.batch_api_mutation(batch)
            for j, new_rule in enumerate(results):
                idx = offspring.index(batch[j])
                offspring[idx] = new_rule
                new_rule["id"] = self.next_id
                self.all_rules[self.next_id] = new_rule
                self.next_id += 1

            mut_done += len(results)
            if bar:
                bar.update(len(results))
            else:
                elapsed = time.time() - t_mut_start
                speed = mut_done / elapsed if elapsed else 0
                print(f"  [mut ] {mut_done}/{mut_total} | {speed:5.1f} r/s", end="\r")

        if bar:
            bar.close()
        else:
            print()

        self.save_checkpoint(f"gen{self.current_generation}_after_mutation")

        for rule in offspring:
            next_islands[self.tag_to_island.get(rule["tag_combination"], 0)].append(
                rule
            )

        self.islands = next_islands
        total_rules = sum(len(i) for i in self.islands)
        elapsed = time.time() - start
        stats = {
            "total_rules": total_rules,
            "island_sizes": [len(i) for i in self.islands],
            "new_rules": len(offspring),
            "generation_time": elapsed,
        }
        self.generation_stats[self.current_generation] = stats
        if self.current_generation % self.checkpoint_interval == 0:
            self.save_checkpoint()
        print(
            f"[GEN] Completed. +{len(offspring)} rules (total {total_rules}) in {elapsed:.2f}s"
        )
        return stats

    # ---------------------------------------------------------------------
    # Multi‑generation driver
    # ---------------------------------------------------------------------

    def evolve_multiple_generations(self, gens: int) -> dict[int, Any]:
        stats: dict[int, Any] = {}
        with tqdm(total=gens, desc="Generations", unit="gen") as bar:
            for _ in range(gens):
                stats[self.current_generation + 1] = self.evolve_one_generation()
                bar.update(1)
                bar.set_postfix(
                    total_rules=sum(len(isle) for isle in self.islands),
                    success=f"{self.api_calls['success']}/{self.api_calls['crossover']+self.api_calls['mutation']}",
                )
        return stats

    # ---------------------------------------------------------------------
    # Export helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def _fmt_time(sec: float) -> str:
        h, rem = divmod(int(sec), 3600)
        m, s = divmod(rem, 60)
        return f"{h}h {m}m {s}s" if h else (f"{m}m {s}s" if m else f"{s}s")

    def _format_rule(self, r: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": r["id"],
            "rule_content": r["rule_content"],
            "generation": r["generation"],
            "parents": r["parents"],
            "mutated": r["mutated"],
            "question_type": r["question_type"],
            "knowledge_point": r["knowledge_point"],
            "times_used": r.get("times_used", 0),
            "creation_method": r.get("creation_method", "unknown"),
        }

    def export_to_json(
        self, out: str, separate: bool = False
    ) -> Dict[str, Any]:  # noqa: WPS231
        if separate:
            island_index: Dict[int, Any] = {}
            base = Path(out).with_suffix("")
            for idx, island in enumerate(self.islands):
                if not island:
                    continue
                tag_info = (
                    f"{island[0]['question_type']}_{island[0]['knowledge_point']}"
                )
                fname = base.with_name(f"{base.name}_island{idx}_{tag_info}.json")
                rules = [self._format_rule(r) for r in island]
                json.dump(
                    rules,
                    fname.open("w", encoding="utf‑8"),
                    ensure_ascii=False,
                    indent=2,
                )
                island_index[idx] = {"filename": fname.name, "rule_count": len(rules)}
            idx_file = Path(out)
            json.dump(
                {
                    "total_rules": len(self.all_rules),
                    "total_generations": self.current_generation,
                    "islands": island_index,
                    "generation_stats": self.generation_stats,
                    "api_stats": self.api_calls,
                },
                idx_file.open("w", encoding="utf‑8"),
                ensure_ascii=False,
                indent=2,
            )
            return {"index": idx_file.as_posix()}
        # flat export --------------------------------------------------------
        payload = {
            "total_rules": len(self.all_rules),
            "total_generations": self.current_generation,
            "tag_stats": {
                "question_types": list(self.question_types),
                "knowledge_points": list(self.knowledge_points),
                "combinations": list(self.tag_combinations),
            },
            "seed_rules": [
                self._format_rule(r)
                for r in self.all_rules.values()
                if r["generation"] == 0
            ],
            "evolved_rules": [
                self._format_rule(r)
                for r in self.all_rules.values()
                if r["generation"] > 0
            ],
            "generation_stats": self.generation_stats,
            "api_stats": self.api_calls,
            "islands": {
                i: {"size": len(island)} for i, island in enumerate(self.islands)
            },
        }
        json.dump(
            payload, Path(out).open("w", encoding="utf‑8"), ensure_ascii=False, indent=2
        )
        return payload


# ---------------------------------------------------------------------------
# High‑level orchestration helpers (akin to Example 1 run_tag_based_evolution)
# ---------------------------------------------------------------------------


def run_evolution_from_args(args: argparse.Namespace) -> None:  # noqa: WPS231
    if args.list_checkpoints:
        cps = TagBasedRuleEvolutionSystem.list_checkpoints(args.checkpoint_dir)
        for i, cp in enumerate(cps, 1):
            print(
                f"{i:2}. gen {cp['generation']:>4} | {cp['total_rules']:>6} rules | {cp['timestamp']} | {Path(cp['checkpoint_file']).name}"
            )
        return

    if args.resume_from:
        system = TagBasedRuleEvolutionSystem.load_checkpoint(
            args.resume_from, args.api_key, args.prompt_dir, args.use_api
        )
    else:
        system = TagBasedRuleEvolutionSystem(
            input_json_file=args.input_json_file,
            mutation_probability=args.mutation_probability,
            migration_interval=args.migration_interval,
            elite_percentage=args.elite_percentage,
            growth_factor=args.growth_factor,
            api_key=args.api_key,
            prompt_dir=args.prompt_dir,
            use_api=args.use_api,
            max_workers=args.max_workers,
            batch_size=args.batch_size,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_interval=args.checkpoint_interval,
        )

    system.evolve_multiple_generations(args.generations)
    system.export_to_json(args.output_filename, args.separate_output)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry‑point."""
    run_evolution_from_args(parse_args())


if __name__ == "__main__":
    main()
