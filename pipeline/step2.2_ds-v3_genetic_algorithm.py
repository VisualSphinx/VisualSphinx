import random
import json
import os
import re
import sys
import time
import signal
import argparse
import pickle
import threading
import concurrent.futures
from collections import defaultdict
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from together import Together

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from api_config import TOGETHER_API_KEY as API_KEY


# Defines and parses command-line arguments for the script.
def parse_args():
    parser = argparse.ArgumentParser(
        description="Tag-Based Rule Evolution System with Parallel API Calls and Resume Capability"
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="run",
        choices=["run", "resume"],
        help="Operation mode: 'run' to start a new evolution, 'resume' to trigger checkpoint resumption.",
    )
    parser.add_argument(
        "--input_json_file",
        type=str,
        default="./data/step2/2.1_eight_classes.json",
        help="Path to the input JSON file containing seed rules.",
        # Coulod also be ./data/step2/2.1_others.json
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="./data/step2/2.2_evolved_rules.json",
        help="Path for the output JSON file for a new run ('run' mode).",
        # Could also be ./data/step2/2.2_evolved_rules_others.json
    )
    parser.add_argument(
        "--generations", type=int, default=9, help="Number of generations to evolve."
    )
    parser.add_argument(
        "--separate_output",
        action="store_true",
        help="Save island outputs separately. Default is False unless this flag is present.",
    )
    parser.add_argument(
        "--prompt_dir",
        type=str,
        default="Prompts",
        help="Directory containing prompt templates.",
    )
    parser.add_argument(
        "--no_api",
        action="store_false",
        dest="use_api_flag",
        help="Disable API usage and run in simulation mode. API is enabled by default if a key is found.",
    )
    parser.set_defaults(use_api_flag=True)
    parser.add_argument(
        "--max_workers",
        type=int,
        default=60,  # Default from user's last full script
        help="Maximum worker threads for parallel API calls.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=120,
        help="Number of rules to process in each API batch.",
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=1,
        help="Save checkpoint every N generations.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./data/ga_checkpoints",
        help="Directory to save and load checkpoints.",
    )
    parser.add_argument(
        "--resume_checkpoint_path",
        type=str,
        default=None,
        help="Path to a specific checkpoint file to resume from (non-interactive resume).",
    )
    parser.add_argument(
        "--resume_generations",
        type=int,
        default=2,
        help="Number of additional generations to run when resuming.",
    )
    parser.add_argument(
        "--resume_output_filename",
        type=str,
        default="./data/step2/2.2_evolved_rules_resume.json",  # Corrected typo from _resuem
        help="Output filename when resuming from a checkpoint.",
    )
    return parser.parse_args()


class TagBasedRuleEvolutionSystem:
    # Initializes the tag-based rule evolution system.
    def __init__(
        self,
        input_json_file,
        mutation_probability,
        migration_interval,
        elite_percentage,
        growth_factor,
        api_key,
        prompt_dir="Prompts",
        use_api=True,
        max_workers=60,  # Default in class signature
        batch_size=120,  # Default in class signature
        checkpoint_dir="checkpoints",  # Default in class signature
        checkpoint_interval=1,  # Default in class signature
    ):
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
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.api_key = api_key
        self.prompt_dir = prompt_dir
        self.use_api = use_api and api_key is not None
        self.template_env = None
        self.client = None

        if self.use_api:
            try:
                self.client = Together(api_key=self.api_key)
                print(
                    f"API client initialized successfully (Parallelism: {self.max_workers}, Batch size: {self.batch_size})"
                )
                if os.path.exists(self.prompt_dir):
                    self.template_env = Environment(
                        loader=FileSystemLoader(self.prompt_dir)
                    )
                    print(
                        f"Successfully loaded prompt template directory: {self.prompt_dir}"
                    )
                else:
                    print(
                        f"Warning: Prompt template directory does not exist: {self.prompt_dir}"
                    )
                    print(
                        "Will use default prompts if template loading fails."
                    )  # Clarified
            except Exception as e:
                print(f"API client initialization failed: {str(e)}")
                print("Will use simulation mode")
                self.use_api = False
        else:
            print("API call disabled, will use simulation mode")

        self.next_id = 0
        self.all_rules = {}
        self.tag_to_island = {}
        self.islands = []
        self.island_count = 0
        self.used_as_parent = set()
        self.generation_stats = defaultdict(dict)
        self.current_generation = 0
        self.question_types = set()
        self.knowledge_points = set()
        self.tag_combinations = set()
        self.api_calls = {
            "crossover": 0,
            "mutation": 0,
            "success": 0,
            "failed": 0,
            "fallback": 0,  # Ensure this key exists from start
            "parallel_batches": 0,
            "total_time": 0.0,
        }
        self._setup_signal_handlers()
        self._load_and_initialize_seeds()

    # Executes API call with exponential backoff, returning '__FALLBACK__' on persistent failure.
    def _execute_api_call_with_exponential_backoff(
        self, prompt, max_retries=10, base_delay=2.0, backoff_factor=2.0
    ):
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="deepseek-ai/DeepSeek-V3",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                )
                content = response.choices[0].message.content
                if content is not None and content.strip():
                    return content
                else:
                    print(
                        f"[Warning] API call attempt {attempt+1} returned empty content, will retry."
                    )
            except Exception as e:
                print(f"[Warning] API call attempt {attempt+1} threw an exception: {e}")

            if attempt < max_retries - 1:
                sleep_time = base_delay * (backoff_factor**attempt)
                print(f"[Warning] Waiting {sleep_time:.2f} seconds before retrying...")
                time.sleep(sleep_time)
        print(
            "[Warning] API call still failed after multiple retries, returning '__FALLBACK__' to use simulation."
        )
        return "__FALLBACK__"

    # Sets up signal handlers to catch interrupts.
    def _setup_signal_handlers(self):
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)

    # Handles interrupt signals (Ctrl+C, etc.) by saving a checkpoint.
    def _handle_interrupt(self, sig, frame):
        print("\n\nInterrupt detected! Saving checkpoint...")
        self.save_checkpoint(f"interrupted_gen{self.current_generation}")
        print(
            "Checkpoint saved, you can use resume_evolution() to restore."
        )  # This function is not standalone
        print("Exiting...")
        exit(0)

    # Saves the current system state to a checkpoint file.
    def save_checkpoint(self, checkpoint_name=None):
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        if checkpoint_name is None:
            checkpoint_name = f"checkpoint_gen{self.current_generation}_{timestamp}"

        checkpoint_path = os.path.join(self.checkpoint_dir, f"{checkpoint_name}.pkl")
        state = {
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
            "api_calls": self.api_calls,
            "input_json_file": self.input_json_file,
            "checkpoint_interval": self.checkpoint_interval,
            "checkpoint_dir": self.checkpoint_dir,  # Save checkpoint_dir used by this instance
        }
        with open(checkpoint_path, "wb") as f:
            pickle.dump(state, f)
        print(f"Checkpoint saved to: {checkpoint_path}")
        index_path = os.path.join(self.checkpoint_dir, f"{checkpoint_name}_index.json")
        index_data = {
            "timestamp": timestamp,
            "generation": self.current_generation,
            "total_rules": len(self.all_rules),
            "island_count": self.island_count,
            "api_calls": self.api_calls,
            "checkpoint_file": checkpoint_path,
        }
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index_data, f, ensure_ascii=False, indent=2)

    # Loads system state from a checkpoint file.
    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_path,
        api_key=None,
        prompt_dir="prompts",
        use_api=True,
        checkpoint_dir_for_loaded_instance=None,  # Allow overriding checkpoint_dir for the new instance
    ):
        print(f"Loading state from checkpoint: {checkpoint_path}")
        with open(checkpoint_path, "rb") as f:
            state = pickle.load(f)

        # Determine checkpoint_dir for the new instance
        # Priority: 1. checkpoint_dir_for_loaded_instance (from CLI), 2. state['checkpoint_dir'] (from pkl), 3. cls default
        effective_checkpoint_dir = checkpoint_dir_for_loaded_instance
        if effective_checkpoint_dir is None:
            effective_checkpoint_dir = state.get(
                "checkpoint_dir", "checkpoints"
            )  # Fallback to "checkpoints" if not in pkl

        instance = cls(
            input_json_file=state["input_json_file"],
            mutation_probability=state["mutation_probability"],
            migration_interval=state["migration_interval"],
            elite_percentage=state["elite_percentage"],
            growth_factor=state["growth_factor"],
            api_key=api_key,
            prompt_dir=prompt_dir,  # Use prompt_dir passed for this session
            use_api=use_api,  # Use use_api passed for this session
            max_workers=state["max_workers"],
            batch_size=state["batch_size"],
            checkpoint_interval=state.get("checkpoint_interval", 1),
            checkpoint_dir=effective_checkpoint_dir,  # Use the determined checkpoint_dir
        )
        instance.next_id = state["next_id"]
        instance.all_rules = state["all_rules"]
        instance.islands = state["islands"]
        instance.island_count = state["island_count"]
        instance.tag_to_island = state["tag_to_island"]
        instance.used_as_parent = state["used_as_parent"]
        instance.generation_stats = defaultdict(dict)
        for gen, stats_data in state["generation_stats"].items():
            instance.generation_stats[gen] = stats_data
        instance.current_generation = state["current_generation"]
        instance.question_types = state["question_types"]
        instance.knowledge_points = state["knowledge_points"]
        instance.tag_combinations = state["tag_combinations"]
        instance.api_calls = state.get(
            "api_calls",
            {  # Provide default for api_calls if not in older checkpoints
                "crossover": 0,
                "mutation": 0,
                "success": 0,
                "failed": 0,
                "fallback": 0,
                "parallel_batches": 0,
                "total_time": 0.0,
            },
        )
        instance.is_resumed = True
        print(
            f"Successfully restored to generation {instance.current_generation} (Total {len(instance.all_rules)} rules)"
        )
        print(
            f"Checkpoint directory for this session set to: {instance.checkpoint_dir}"
        )
        return instance

    # Lists available checkpoints in the specified directory.
    @classmethod
    def list_checkpoints(cls, checkpoint_dir="checkpoints"):
        if not os.path.exists(checkpoint_dir):
            print(f"Checkpoint directory does not exist: {checkpoint_dir}")
            return []
        index_files = [
            f for f in os.listdir(checkpoint_dir) if f.endswith("_index.json")
        ]
        if not index_files:
            print(f"No checkpoint index files found in {checkpoint_dir}")
            return []
        checkpoints = []
        for index_file in index_files:
            index_path = os.path.join(checkpoint_dir, index_file)
            try:
                with open(index_path, "r", encoding="utf-8") as f:
                    index_data = json.load(f)
                checkpoint_file = index_data.get("checkpoint_file")
                if checkpoint_file and os.path.exists(
                    checkpoint_file
                ):  # Ensure .pkl file also exists
                    checkpoints.append(
                        {
                            "index_file": index_file,
                            "checkpoint_file": checkpoint_file,
                            "generation": index_data.get("generation", 0),
                            "total_rules": index_data.get("total_rules", 0),
                            "timestamp": index_data.get("timestamp", "Unknown"),
                        }
                    )
                elif checkpoint_file and not os.path.exists(checkpoint_file):
                    print(
                        f"Warning: Index file {index_file} points to missing checkpoint file {checkpoint_file}"
                    )

            except Exception as e:
                print(f"Error reading index file {index_file}: {str(e)}")
        checkpoints.sort(key=lambda x: x["generation"], reverse=True)
        return checkpoints

    # Loads seed rules from a JSON file and assigns them to islands based on tags.
    def _load_and_initialize_seeds(self):
        if self.is_resumed:
            print("Resuming from checkpoint, skipping seed rule initialization")
            return
        try:
            with open(self.input_json_file, "r", encoding="utf-8") as f:
                seed_data = json.load(f)
            print(f"Loaded {len(seed_data)} seed rules")
            tag_groups = defaultdict(list)
            for seed in seed_data:
                question_type = seed.get("question_type", "unknown")
                knowledge_point = seed.get("knowledge_point", "unknown")
                self.question_types.add(question_type)
                self.knowledge_points.add(knowledge_point)
                tag_combination = (question_type, knowledge_point)
                self.tag_combinations.add(tag_combination)
                rule_content = seed.get("seed_rule", [])
                if not isinstance(rule_content, list):
                    rule_content = [rule_content]
                rule = {
                    "id": seed.get("seed_id", self.next_id),
                    "rule_content": rule_content,
                    "generation": 0,
                    "parents": [],
                    "mutated": False,
                    "times_used": 0,
                    "ancestry_depth": 0,
                    "creation_method": "seed",
                    "question_type": question_type,
                    "knowledge_point": knowledge_point,
                    "tag_combination": tag_combination,
                }
                tag_groups[tag_combination].append(rule)
                self.next_id = max(self.next_id, rule["id"] + 1)
                self.all_rules[rule["id"]] = rule
            self.island_count = len(tag_groups)
            self.islands = [[] for _ in range(self.island_count)]
            print(
                f"Detected {len(self.question_types)} question types, {len(self.knowledge_points)} knowledge points"
            )
            print(
                f"Found {self.island_count} unique tag combinations, will create {self.island_count} islands"
            )
            for i, (tag_combination, rules) in enumerate(tag_groups.items()):
                self.tag_to_island[tag_combination] = i
                self.islands[i] = rules
                print(
                    f"Island {i}: Question Type = {tag_combination[0]}, Knowledge Point = {tag_combination[1]}, Rule count = {len(rules)}"
                )
            self.generation_stats[0] = {
                "total_rules": len(self.all_rules),
                "island_sizes": [len(island) for island in self.islands],
                "tag_stats": {
                    "question_types": list(self.question_types),
                    "knowledge_points": list(self.knowledge_points),
                    "combinations": len(self.tag_combinations),
                },
            }
        except Exception as e:
            print(f"Error loading seed rules: {str(e)}")
            raise

    # Calculates rule fitness for elite selection.
    def calculate_rule_fitness(self, rule):
        usage_score = 1.0 / (1.0 + 0.2 * rule.get("times_used", 0))
        ancestry_score = min(1.0, 0.5 + 0.1 * rule.get("ancestry_depth", 0))
        random_score = random.uniform(0.8, 1.0)
        mutation_bonus = 1.1 if rule.get("mutated", False) else 1.0
        fitness = (
            0.4 * usage_score + 0.4 * ancestry_score + 0.2 * random_score
        ) * mutation_bonus
        return fitness

    # Selects elite rules to be preserved for the next generation.
    def select_elite_rules(self, population, elite_percentage):
        elite_count = max(1, int(len(population) * elite_percentage))
        sorted_population = sorted(
            population, key=self.calculate_rule_fitness, reverse=True
        )
        return sorted_population[:elite_count]

    # Selects a parent rule using a diversity-driven strategy.
    def select_parent(self, population):
        unused_rules = [rule for rule in population if rule["times_used"] == 0]
        if unused_rules and random.random() < 0.4:
            selected_rule = random.choice(unused_rules)
        else:
            weights = []
            for rule_item in population:
                weight = 1.0 / (rule_item["times_used"] + 1)
                weights.append(weight)
            total_weight = sum(weights)
            if total_weight > 0:
                normalized_weights = [w / total_weight for w in weights]
                selected_rule = random.choices(
                    population, weights=normalized_weights, k=1
                )[0]
            else:
                selected_rule = random.choice(population)
        selected_rule["times_used"] += 1
        self.used_as_parent.add(selected_rule["id"])
        return selected_rule

    # Retrieves the specified prompt template content.
    def get_template(self, template_name):
        if self.template_env:
            try:
                template = self.template_env.get_template(f"{template_name}.md")
                return template.render()
            except Exception as e:
                print(
                    f"Warning: Error loading template file {template_name}.md: {str(e)}. Using hardcoded default."
                )
        else:
            print(
                f"Warning: Template environment not initialized. Using hardcoded default for '{template_name}'."
            )

        if template_name == "step2.2_crossover":
            return "Default Crossover Prompt: Combine {FIRST_RULE_SET} and {SECOND_RULE_SET} into <crossover_rules>...</crossover_rules>"  # Example default
        elif template_name == "step2.2_mutation":
            return "Default Mutation Prompt: Mutate {RULE_SET} into <mutated_rules>...</mutated_rules>"  # Example default

        print(
            f"Warning: No template file or specific hardcoded default found for '{template_name}'. Returning empty string."
        )
        return ""

    # Prepares the API prompt for a crossover operation.
    def _prepare_crossover_prompt(self, parent1, parent2):
        template_str = self.get_template("step2.2_crossover")
        if not template_str:
            print(f"Warning: Crossover template is empty!")  # Handle empty template
        first_rule_set = parent1["rule_content"]
        second_rule_set = parent2["rule_content"]
        if isinstance(first_rule_set, list):
            first_rule_set = "\n- " + "\n- ".join(map(str, first_rule_set))
        if isinstance(second_rule_set, list):
            second_rule_set = "\n- " + "\n- ".join(map(str, second_rule_set))

        prompt = template_str  # Start with the template string
        if "{FIRST_RULE_SET}" in prompt:  # Check before replacing
            prompt = prompt.replace("{FIRST_RULE_SET}", str(first_rule_set))
        if "{SECOND_RULE_SET}" in prompt:
            prompt = prompt.replace("{SECOND_RULE_SET}", str(second_rule_set))

        if random.random() < 0.5:
            question_type = parent1["question_type"]
            knowledge_point = parent1["knowledge_point"]
        else:
            question_type = parent2["question_type"]
            knowledge_point = parent2["knowledge_point"]
        ancestry_depth = (
            max(parent1.get("ancestry_depth", 0), parent2.get("ancestry_depth", 0)) + 1
        )
        return prompt, {
            "parents": [parent1["id"], parent2["id"]],
            "question_type": question_type,
            "knowledge_point": knowledge_point,
            "tag_combination": (question_type, knowledge_point),
            "ancestry_depth": ancestry_depth,
        }

    # Prepares the API prompt for a mutation operation.
    def _prepare_mutation_prompt(self, rule):
        template_str = self.get_template("step2.2_mutation")
        if not template_str:
            print(f"Warning: Mutation template is empty!")
        rule_content = rule["rule_content"]
        if isinstance(rule_content, list):
            rule_content = "\n- " + "\n- ".join(map(str, rule_content))

        prompt = template_str
        if "{RULE_SET}" in prompt:  # Check before replacing
            prompt = prompt.replace("{RULE_SET}", str(rule_content))
        original_id = rule.get("id", -1)
        return prompt, {
            "original_id": original_id,
            "question_type": rule["question_type"],
            "knowledge_point": rule["knowledge_point"],
            "tag_combination": rule["tag_combination"],
            "ancestry_depth": rule.get("ancestry_depth", 0),
        }

    # Executes a single API call.
    def _execute_api_call(self, prompt):
        try:
            response = self.client.chat.completions.create(
                model="deepseek-ai/DeepSeek-V3",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API call failed: {str(e)}")
            self.api_calls["failed"] += 1
            return None

    # Extracts crossover rules from the API response text.
    def extract_crossover_rules(self, response_text):
        if not response_text:
            return None
        try:
            pattern = r"<crossover_rules>(.*?)</crossover_rules>"
            match = re.search(pattern, response_text, re.DOTALL)
            if match:
                rules_text = match.group(1).strip()
                rules = []
                for line in rules_text.split("\n"):
                    line = line.strip()
                    line = re.sub(r"^[\-\d\.\s]+", "", line).strip()
                    if line and not line.startswith("#") and not line.startswith("("):
                        rules.append(line)
                return rules
            return None
        except Exception as e:
            print(f"  Error extracting crossover rules: {str(e)}")
            return None

    # Extracts mutated rules from the API response text.
    def extract_mutated_rules(self, response_text):
        if not response_text:
            return None
        try:
            pattern = r"<mutated_rules>(.*?)</mutated_rules>"
            match = re.search(pattern, response_text, re.DOTALL)
            if match:
                rules_text = match.group(1).strip()
                rules = []
                for line in rules_text.split("\n"):
                    line = line.strip()
                    line = re.sub(r"^[\-\d\.\s]+", "", line).strip()
                    if line and not line.startswith("#") and not line.startswith("("):
                        rules.append(line)
                return rules
            return None
        except Exception as e:
            print(f"  Error extracting mutated rules: {str(e)}")
            return None

    # Processes the API response for a crossover operation.
    def _process_crossover_result(self, response_text, parent_info):
        crossover_rules = self.extract_crossover_rules(response_text)
        if crossover_rules:
            return {
                "rule_content": crossover_rules,
                "generation": self.current_generation,
                "parents": parent_info["parents"],
                "mutated": False,
                "times_used": 0,
                "ancestry_depth": parent_info["ancestry_depth"],
                "creation_method": "api_crossover",
                "question_type": parent_info["question_type"],
                "knowledge_point": parent_info["knowledge_point"],
                "tag_combination": parent_info["tag_combination"],
            }
        return None

    # Processes the API response for a mutation operation.
    def _process_mutation_result(self, response_text, rule_info):
        mutated_rules = self.extract_mutated_rules(response_text)
        if mutated_rules:
            parents_list = []
            if rule_info["original_id"] != -1:
                parents_list = [rule_info["original_id"]]
            return {
                "rule_content": mutated_rules,
                "generation": self.current_generation,
                "parents": parents_list,
                "mutated": True,
                "times_used": 0,
                "ancestry_depth": rule_info["ancestry_depth"],
                "creation_method": "api_mutation",
                "question_type": rule_info["question_type"],
                "knowledge_point": rule_info["knowledge_point"],
                "tag_combination": rule_info["tag_combination"],
            }
        return None

    # Simulates a crossover operation when the API is unavailable or fails.
    def _simulate_crossover(self, parent1, parent2):
        crossover_type = random.choice(
            ["StandardCrossover", "HybridCrossover", "InnovativeCombination"]
        )
        ancestry_depth = (
            max(parent1.get("ancestry_depth", 0), parent2.get("ancestry_depth", 0)) + 1
        )
        if random.random() < 0.5:
            question_type = parent1["question_type"]
            knowledge_point = parent1["knowledge_point"]
        else:
            question_type = parent2["question_type"]
            knowledge_point = parent2["knowledge_point"]
        tag_combination = (question_type, knowledge_point)
        parent1_rules = (
            parent1["rule_content"]
            if isinstance(parent1["rule_content"], list)
            else [parent1["rule_content"]]
        )
        parent2_rules = (
            parent2["rule_content"]
            if isinstance(parent2["rule_content"], list)
            else [parent2["rule_content"]]
        )
        combined_rules = [f"New_Simulated_Rule_{random.randint(1000, 9999)}"]
        for rule_item in parent1_rules + parent2_rules:
            if rule_item not in combined_rules and random.random() < 0.7:
                combined_rules.append(rule_item)
        if not combined_rules and parent1_rules:
            combined_rules.append(random.choice(parent1_rules))
        elif not combined_rules and parent2_rules:
            combined_rules.append(random.choice(parent2_rules))
        return {
            "rule_content": combined_rules,
            "generation": self.current_generation,
            "parents": [parent1["id"], parent2["id"]],
            "mutated": False,
            "times_used": 0,
            "ancestry_depth": ancestry_depth,
            "creation_method": f"sim_crossover_{crossover_type}",
            "question_type": question_type,
            "knowledge_point": knowledge_point,
            "tag_combination": tag_combination,
        }

    # Simulates a mutation operation when the API is unavailable or fails.
    def _simulate_mutation(self, rule):
        mutation_types = ["SmallVariation", "MediumVariation", "LargeVariation"]
        mutation_weights = [0.6, 0.3, 0.1]
        mutation_type = random.choices(mutation_types, weights=mutation_weights, k=1)[0]
        mutated_rule = rule.copy()
        current_rule_content = rule.get("rule_content", [])
        if not isinstance(current_rule_content, list):
            current_rule_content = (
                [str(current_rule_content)] if current_rule_content else []
            )

        if current_rule_content:
            if random.random() < 0.7:
                index_to_mutate = random.randint(0, len(current_rule_content) - 1)
                new_mutated_content = current_rule_content.copy()
                new_mutated_content[index_to_mutate] = (
                    f"{current_rule_content[index_to_mutate]}_{mutation_type}"
                )
                mutated_rule["rule_content"] = new_mutated_content
            else:
                new_mutated_content = current_rule_content.copy()
                new_mutated_content.append(f"Newly_Generated_{mutation_type}_Rule")
                mutated_rule["rule_content"] = new_mutated_content
        else:
            mutated_rule["rule_content"] = [f"Newly_Generated_{mutation_type}_Rule"]

        mutated_rule["mutated"] = True
        mutated_rule["creation_method"] = f"sim_mutation_{mutation_type}"
        if "id" in rule and rule["id"] is not None:
            mutated_rule["parents"] = [rule["id"]]
        else:
            mutated_rule["parents"] = []

        for key_to_preserve in [
            "question_type",
            "knowledge_point",
            "tag_combination",
            "ancestry_depth",
        ]:
            if key_to_preserve in rule:
                mutated_rule[key_to_preserve] = rule[key_to_preserve]
            elif key_to_preserve not in mutated_rule:
                if key_to_preserve == "ancestry_depth":
                    mutated_rule[key_to_preserve] = 0
                elif key_to_preserve == "tag_combination":
                    mutated_rule[key_to_preserve] = ("unknown", "unknown")
                else:
                    mutated_rule[key_to_preserve] = "unknown"
        return mutated_rule

    # Retries API call and parsing for crossover, falling back to None on persistent failure.
    def _retry_api_parse_crossover(
        self,
        prompt,
        parent_info,
        max_call_retries=10,
        max_parse_retries=10,
        base_delay=2.0,
        backoff_factor=2.0,
    ):
        for attempt_parse in range(max_parse_retries):
            response_text = self._execute_api_call_with_exponential_backoff(
                prompt,
                max_retries=max_call_retries,
                base_delay=base_delay,
                backoff_factor=backoff_factor,
            )
            if response_text == "__FALLBACK__":
                self.api_calls["failed"] += 1
                return None
            rule_result = self._process_crossover_result(response_text, parent_info)
            if rule_result:
                return rule_result
            else:
                print(
                    f"[Thread {threading.get_ident()} Parse attempt {attempt_parse+1}/{max_parse_retries} failed for crossover, will re-request API..."
                )
        self.api_calls["failed"] += 1
        return None

    # Executes crossover API calls in parallel, with retries and fallback to simulation.
    def batch_api_crossover(self, crossover_tasks):
        if not crossover_tasks:
            return []
        if not self.use_api:
            self.api_calls["fallback"] += len(crossover_tasks)
            return [
                self._simulate_crossover(parent1, parent2)
                for parent1, parent2 in crossover_tasks
            ]
        batch_start_time = time.time()
        self.api_calls["parallel_batches"] += 1
        current_batch_size = len(crossover_tasks)
        print(
            f"Starting parallel crossover batch #{self.api_calls['parallel_batches']} (Tasks: {current_batch_size})"
        )
        tasks_to_submit = []
        for parent1, parent2 in crossover_tasks:
            prompt_text, parent_info_dict = self._prepare_crossover_prompt(
                parent1, parent2
            )
            tasks_to_submit.append((prompt_text, parent_info_dict))
        results = []
        successful_api_calls = 0
        fallback_count_for_batch = 0
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            future_to_task_map = {
                executor.submit(
                    self._retry_api_parse_crossover,
                    prompt,
                    parent_info,
                    10,
                    10,
                    2.0,
                    2.0,
                ): (prompt, parent_info)
                for prompt, parent_info in tasks_to_submit
            }
            for future in concurrent.futures.as_completed(future_to_task_map):
                _original_prompt, original_parent_info = future_to_task_map[future]
                self.api_calls["crossover"] += 1
                try:
                    rule_result = future.result()
                    if rule_result:
                        results.append(rule_result)
                        successful_api_calls += 1
                        self.api_calls["success"] += 1
                    else:
                        parent1_id, parent2_id = original_parent_info["parents"]
                        parent1 = self.all_rules[parent1_id]
                        parent2 = self.all_rules[parent2_id]
                        sim_rule = self._simulate_crossover(parent1, parent2)
                        results.append(sim_rule)
                        fallback_count_for_batch += 1
                        self.api_calls["fallback"] += 1
                except Exception as e:
                    print(
                        f"[Warning] batch_api_crossover() task processing exception: {e}, falling back."
                    )
                    parent1_id, parent2_id = original_parent_info["parents"]
                    parent1 = self.all_rules[parent1_id]
                    parent2 = self.all_rules[parent2_id]
                    sim_rule = self._simulate_crossover(parent1, parent2)
                    results.append(sim_rule)
                    fallback_count_for_batch += 1
                    self.api_calls["fallback"] += 1
                    self.api_calls["failed"] += 1
        batch_time_taken = time.time() - batch_start_time
        self.api_calls["total_time"] += batch_time_taken
        print(
            f"Completed crossover batch #{self.api_calls['parallel_batches']}: "
            f"API success {successful_api_calls}/{current_batch_size}, fallback {fallback_count_for_batch} "
            f"(Time: {batch_time_taken:.2f}s, Avg: {batch_time_taken/max(1,current_batch_size):.2f}s/task)"
        )
        return results

    # Retries API call and parsing for mutation, falling back to None on persistent failure.
    def _retry_api_parse_mutation(
        self,
        prompt,
        rule_info,
        max_call_retries=10,
        max_parse_retries=10,
        base_delay=2.0,
        backoff_factor=2.0,
    ):
        for attempt_parse in range(max_parse_retries):
            response_text = self._execute_api_call_with_exponential_backoff(
                prompt,
                max_retries=max_call_retries,
                base_delay=base_delay,
                backoff_factor=backoff_factor,
            )
            if response_text == "__FALLBACK__":
                self.api_calls["failed"] += 1
                return None
            mutated_rule_result = self._process_mutation_result(
                response_text, rule_info
            )
            if mutated_rule_result:
                return mutated_rule_result
            else:
                print(
                    f"[Warning] Parse attempt {attempt_parse+1}/{max_parse_retries} for mutation tag failed, re-requesting API..."
                )
        self.api_calls["failed"] += 1
        return None

    # Executes mutation API calls in parallel, with retries and fallback to simulation.
    def batch_api_mutation(self, mutation_tasks):
        if not mutation_tasks:
            return []
        if not self.use_api:
            self.api_calls["fallback"] += len(mutation_tasks)
            return [
                self._simulate_mutation(rule_to_mutate)
                for rule_to_mutate in mutation_tasks
            ]
        batch_start_time = time.time()
        current_batch_size = len(mutation_tasks)
        self.api_calls["parallel_batches"] += 1
        print(f"Starting parallel mutation batch (Tasks: {current_batch_size})")
        tasks_to_submit = []
        for idx, rule_to_mutate in enumerate(mutation_tasks):
            prompt_text, rule_info_dict = self._prepare_mutation_prompt(rule_to_mutate)
            tasks_to_submit.append(
                (prompt_text, rule_info_dict, rule_to_mutate)
            )  # Include original rule for fallback

        results = []
        successful_api_calls = 0
        fallback_count_for_batch = 0
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            future_to_task_map = {
                executor.submit(
                    self._retry_api_parse_mutation,
                    prompt,
                    rule_info,
                    10,
                    10,
                    2.0,
                    2.0,
                ): (
                    original_rule_for_fb
                )  # Key by future, value is original rule for fallback
                for prompt, rule_info, original_rule_for_fb in tasks_to_submit
            }
            for future in concurrent.futures.as_completed(future_to_task_map):
                original_input_rule = future_to_task_map[future]
                self.api_calls["mutation"] += 1
                try:
                    mutated_rule_result = future.result()
                    if mutated_rule_result:
                        results.append(mutated_rule_result)
                        successful_api_calls += 1
                        self.api_calls["success"] += 1
                    else:
                        sim_rule = self._simulate_mutation(original_input_rule)
                        results.append(sim_rule)
                        fallback_count_for_batch += 1
                        self.api_calls["fallback"] += 1
                except Exception as e:
                    print(
                        f"[Warning] batch_api_mutation task processing exception: {e}, falling back."
                    )
                    sim_rule = self._simulate_mutation(original_input_rule)
                    results.append(sim_rule)
                    fallback_count_for_batch += 1
                    self.api_calls["fallback"] += 1
                    self.api_calls["failed"] += 1
        batch_time_taken = time.time() - batch_start_time
        self.api_calls["total_time"] += batch_time_taken
        print(
            f"Completed mutation batch #{self.api_calls['parallel_batches']}: "
            f"API success {successful_api_calls}/{current_batch_size}, fallback {fallback_count_for_batch} "
            f"(Time: {batch_time_taken:.2f}s, Avg: {batch_time_taken/max(1, current_batch_size):.2f}s/task)"
        )
        return results

    # Migrates rules between islands to promote diversity.
    def migrate_between_islands(self):
        if self.island_count <= 1:
            return
        reference_island_size = 0
        for island_pop in self.islands:
            if island_pop:
                reference_island_size = len(island_pop)
                break
        if reference_island_size == 0 and self.islands:
            migrants_per_island = 0
        elif not self.islands:
            return
        else:
            migrants_per_island = max(1, int(reference_island_size * 0.05))
        if migrants_per_island == 0:
            return
        island_migrants = {}
        for i, island_pop in enumerate(self.islands):
            if len(island_pop) > migrants_per_island:
                migrants_sample = random.sample(island_pop, migrants_per_island)
                island_migrants[i] = migrants_sample
                for migrant_rule in migrants_sample:
                    island_pop.remove(migrant_rule)
            elif len(island_pop) > 0 and len(island_pop) <= migrants_per_island:
                island_migrants[i] = list(island_pop)
                island_pop.clear()
        migration_map = {}
        island_indices = list(range(self.island_count))
        for source_idx in island_migrants.keys():
            possible_targets = [idx for idx in island_indices if idx != source_idx]
            if possible_targets:
                target_idx = random.choice(possible_targets)
                migration_map[source_idx] = target_idx
        for source_idx, target_idx in migration_map.items():
            if source_idx in island_migrants:
                self.islands[target_idx].extend(island_migrants[source_idx])
                print(
                    f"Migrated {len(island_migrants[source_idx])} rules: Island {source_idx} -> Island {target_idx}"
                )

    # Evolves the population for one generation.
    def evolve_one_generation(self):
        gen_start_time = time.time()
        self.current_generation += 1
        print(f"Starting generation {self.current_generation} evolution...")
        if (
            self.current_generation > 0
            and self.current_generation % self.migration_interval == 0
        ):
            self.migrate_between_islands()
        next_gen_islands = [[] for _ in range(self.island_count)]
        current_sizes = [len(island_pop) for island_pop in self.islands]
        all_crossover_tasks = []
        for island_idx, island_pop in enumerate(self.islands):
            if not island_pop:
                continue
            elites = self.select_elite_rules(island_pop, self.elite_percentage)
            next_gen_islands[island_idx].extend(elites)
            num_elites_current_island = len(elites)
            target_island_size = int(current_sizes[island_idx] * self.growth_factor)
            new_rules_to_create = target_island_size - num_elites_current_island
            new_rules_to_create = max(0, new_rules_to_create)
            if len(island_pop) < 2:
                if island_pop and new_rules_to_create > 0:
                    for _ in range(new_rules_to_create):
                        rule_to_replicate = random.choice(
                            elites if elites else island_pop
                        ).copy()
                        rule_to_replicate["generation"] = self.current_generation
                        rule_to_replicate["id"] = self.next_id
                        self.next_id += 1
                        self.all_rules[rule_to_replicate["id"]] = rule_to_replicate
                        next_gen_islands[island_idx].append(rule_to_replicate)
                continue
            island_crossover_tasks = []
            for _ in range(new_rules_to_create):
                parent1 = self.select_parent(island_pop)
                available_for_parent2 = [
                    r for r in island_pop if r["id"] != parent1["id"]
                ]
                if not available_for_parent2:
                    available_for_parent2 = island_pop
                parent2 = self.select_parent(available_for_parent2)
                island_crossover_tasks.append((parent1, parent2))
            all_crossover_tasks.extend(island_crossover_tasks)
        offspring_rules_from_crossover = []  # Renamed
        for i in range(0, len(all_crossover_tasks), self.batch_size):
            batch_tasks = all_crossover_tasks[i : i + self.batch_size]
            batch_crossover_results = self.batch_api_crossover(batch_tasks)
            offspring_rules_from_crossover.extend(batch_crossover_results)

        current_generation_offspring = (
            []
        )  # Will hold ID'd rules for this gen, before mutation
        for rule_from_crossover in offspring_rules_from_crossover:
            rule_from_crossover["id"] = self.next_id
            self.all_rules[rule_from_crossover["id"]] = rule_from_crossover
            current_generation_offspring.append(rule_from_crossover)
            self.next_id += 1

        if self.current_generation > 0:
            self.save_checkpoint(
                f"checkpoint_gen{self.current_generation}_after_crossover"
            )

        mutation_candidates = []
        for rule_candidate in current_generation_offspring:
            if random.random() < self.mutation_probability:
                mutation_candidates.append(rule_candidate)

        mutated_rules_map = {}
        for i in range(0, len(mutation_candidates), self.batch_size):
            batch_to_mutate = mutation_candidates[i : i + self.batch_size]
            batch_mutation_results = self.batch_api_mutation(batch_to_mutate)
            for original_rule, mutated_version_dict in zip(
                batch_to_mutate, batch_mutation_results
            ):
                mutated_version_dict["id"] = self.next_id
                self.all_rules[mutated_version_dict["id"]] = mutated_version_dict
                self.next_id += 1
                mutated_rules_map[original_rule["id"]] = mutated_version_dict

        finalized_offspring_for_generation = []
        for rule_in_gen_offspring in current_generation_offspring:
            if rule_in_gen_offspring["id"] in mutated_rules_map:
                finalized_offspring_for_generation.append(
                    mutated_rules_map[rule_in_gen_offspring["id"]]
                )
            else:
                finalized_offspring_for_generation.append(rule_in_gen_offspring)

        if self.current_generation > 0:
            self.save_checkpoint(
                f"checkpoint_gen{self.current_generation}_after_mutation"
            )

        for final_offspring_rule in finalized_offspring_for_generation:
            tag_combo = final_offspring_rule.get("tag_combination")
            if tag_combo is None:
                qt = final_offspring_rule.get("question_type", "unknown")
                kp = final_offspring_rule.get("knowledge_point", "unknown")
                tag_combo = (qt, kp)
                final_offspring_rule["tag_combination"] = tag_combo
            target_island_idx = self.tag_to_island.get(tag_combo)
            if target_island_idx is None:
                print(
                    f"Warning: New tag combination {tag_combo} for rule ID {final_offspring_rule.get('id')} not mapped to an island. Assigning to island 0."
                )
                target_island_idx = 0
            if 0 <= target_island_idx < len(next_gen_islands):
                next_gen_islands[target_island_idx].append(final_offspring_rule)
            else:
                print(
                    f"Error: Invalid target_island_idx {target_island_idx} for rule ID {final_offspring_rule.get('id')}. Skipping."
                )
        self.islands = next_gen_islands
        current_total_rules = sum(len(island_pop) for island_pop in self.islands)
        gen_time_taken = time.time() - gen_start_time
        api_stats_for_gen = {
            k: v for k, v in self.api_calls.items() if k not in ["total_time"]
        }
        self.generation_stats[self.current_generation] = {
            "total_rules": current_total_rules,
            "island_sizes": [len(island_pop) for island_pop in self.islands],
            "used_as_parent_count": len(self.used_as_parent),
            "api_calls_summary": api_stats_for_gen,
            "generation_time_seconds": gen_time_taken,
            "new_rules_generated_this_gen": len(finalized_offspring_for_generation),
        }
        self.used_as_parent.clear()
        print(
            f"Generation {self.current_generation} evolution completed: "
            f"{len(finalized_offspring_for_generation)} new rules processed (Total rules: {current_total_rules}, Time: {gen_time_taken:.2f}s)"
        )
        if (
            self.current_generation > 0
            and self.current_generation % self.checkpoint_interval == 0
        ):
            self.save_checkpoint()
        return self.generation_stats[self.current_generation]

    # Evolves the population for multiple generations.
    def evolve_multiple_generations(self, generations=10, show_progress=True):
        all_stats = {}
        total_start_time = time.time()
        starting_gen = self.current_generation
        print(
            f"Planning to evolve {generations} generations (Current: {starting_gen}, Target end gen: {starting_gen + generations})"
        )
        try:
            for i_gen_count in range(1, generations + 1):
                gen_stats = self.evolve_one_generation()
                all_stats[self.current_generation] = gen_stats
                if show_progress:
                    total_rules_now = sum(
                        len(island_pop) for island_pop in self.islands
                    )
                    elapsed_time = time.time() - total_start_time
                    generations_completed_this_run = i_gen_count
                    avg_time_per_completed_gen = (
                        elapsed_time / generations_completed_this_run
                    )
                    remaining_gens_to_run = generations - generations_completed_this_run
                    estimated_remaining_time = (
                        avg_time_per_completed_gen * remaining_gens_to_run
                    )
                    elapsed_str = self._format_time(elapsed_time)
                    remaining_str = self._format_time(estimated_remaining_time)
                    current_api_calls_total = (
                        self.api_calls["crossover"] + self.api_calls["mutation"]
                    )
                    current_api_success_count = self.api_calls["success"]
                    success_rate_str = (
                        f"{current_api_success_count / max(1, current_api_calls_total):.1%}"
                        if current_api_calls_total > 0
                        else "N/A"
                    )
                    print(
                        f"Progress: {generations_completed_this_run}/{generations} generations in this run (Now at Gen {self.current_generation}) "
                        + f"| Current total rules: {total_rules_now:,} "
                        + f"| API calls (total): {self.api_calls['crossover']} C + {self.api_calls['mutation']} M "
                        + f"(Success rate: {success_rate_str})"
                    )
                    print(
                        f"Time elapsed: {elapsed_str} | Estimated time remaining: {remaining_str}"
                    )
                    if (
                        generations_completed_this_run % 5 == 0
                        or generations_completed_this_run == generations
                    ):
                        print(
                            "Island sizes:",
                            " ".join(
                                [
                                    f"{idx}:{len(island_pop)}"
                                    for idx, island_pop in enumerate(self.islands)
                                    if island_pop
                                ]
                            ),
                        )
                    print("-" * 60)
        except KeyboardInterrupt:
            print("\nEvolution process manually interrupted.")
            print("Checkpoint automatically saved. Can resume later.")
            self.save_checkpoint("manual_interrupt")
        print("Evolution process finished for this run. Saving final checkpoint...")
        self.save_checkpoint(f"final_gen{self.current_generation}")
        return all_stats

    # Formats time in seconds to a human-readable string (h:m:s).
    def _format_time(self, seconds_val):
        hours, remainder = divmod(int(seconds_val), 3600)
        minutes, secs_final = divmod(remainder, 60)
        if hours > 0:
            return f"{hours}h{minutes}m{secs_final}s"
        elif minutes > 0:
            return f"{minutes}m{secs_final}s"
        else:
            return f"{secs_final}s"

    # Exports the current system state to JSON file(s).
    def export_to_json(self, output_filename, separate_by_island=False):
        if separate_by_island:
            island_data_export = {}
            for island_idx, island_pop in enumerate(self.islands):
                if not island_pop:
                    continue
                first_rule = island_pop[0]
                tag_info_str = (
                    f"{first_rule['question_type']}_{first_rule['knowledge_point']}"
                )
                rules_list_for_island = []
                for rule_item in island_pop:
                    rule_export_dict = {
                        "id": rule_item["id"],
                        "rule_content": rule_item["rule_content"],
                        "generation": rule_item["generation"],
                        "parents": rule_item["parents"],
                        "mutated": rule_item["mutated"],
                        "question_type": rule_item["question_type"],
                        "knowledge_point": rule_item["knowledge_point"],
                        "creation_method": rule_item.get("creation_method", "unknown"),
                    }
                    rules_list_for_island.append(rule_export_dict)
                base_output_fn, ext = os.path.splitext(output_filename)
                actual_island_filename = (
                    f"{base_output_fn}_island{island_idx}_{tag_info_str}{ext}"
                )
                with open(actual_island_filename, "w", encoding="utf-8") as f:
                    json.dump(rules_list_for_island, f, ensure_ascii=False, indent=2)
                island_data_export[island_idx] = {
                    "filename": actual_island_filename,
                    "rule_count": len(rules_list_for_island),
                    "question_type": first_rule["question_type"],
                    "knowledge_point": first_rule["knowledge_point"],
                }
            index_data_export = {
                "total_rules": len(self.all_rules),
                "total_generations_completed": self.current_generation,
                "islands_export_info": island_data_export,
                "generation_statistics": dict(self.generation_stats),
                "api_call_statistics": self.api_calls,
            }
            with open(output_filename, "w", encoding="utf-8") as f:
                json.dump(index_data_export, f, ensure_ascii=False, indent=2)
            return index_data_export
        else:
            output_data = {
                "total_rules": len(self.all_rules),
                "total_generations_completed": self.current_generation,
                "tag_summary_statistics": {
                    "question_types": sorted(list(self.question_types)),
                    "knowledge_points": sorted(list(self.knowledge_points)),
                    "tag_combinations_observed": sorted(
                        [list(tc) for tc in self.tag_combinations]
                    ),
                },
                "seed_rules": [
                    self._format_rule_for_export(rule_item)
                    for rule_item in self.all_rules.values()
                    if rule_item["generation"] == 0
                ],
                "evolved_rules": [
                    self._format_rule_for_export(rule_item)
                    for rule_item in self.all_rules.values()
                    if rule_item["generation"] > 0
                ],
                "generation_statistics": dict(self.generation_stats),
                "api_call_statistics": self.api_calls,
                "island_summaries": {
                    i: {
                        "size": len(island_pop),
                        "distinct_tags_present": (
                            sorted(
                                list(
                                    set(
                                        (
                                            rule_item["question_type"],
                                            rule_item["knowledge_point"],
                                        )
                                        for rule_item in island_pop
                                    )
                                )
                            )
                            if island_pop
                            else []
                        ),
                    }
                    for i, island_pop in enumerate(self.islands)
                },
            }
            with open(output_filename, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            return output_data

    # Formats a single rule for JSON export.
    def _format_rule_for_export(self, rule):
        return {
            "id": rule["id"],
            "rule_content": rule["rule_content"],
            "generation": rule["generation"],
            "parents": rule["parents"],
            "mutated": rule["mutated"],
            "question_type": rule["question_type"],
            "knowledge_point": rule["knowledge_point"],
            "times_used_as_parent": rule.get("times_used", 0),
            "creation_method": rule.get("creation_method", "unknown"),
            "ancestry_depth": rule.get("ancestry_depth", 0),
        }


# Runs the tag-based evolution algorithm.
def run_tag_based_evolution(
    input_json_file,
    output_filename="evolved_rules.json",
    generations=10,
    separate_output=False,
    api_key=None,
    prompt_dir="Prompts",
    use_api=True,
    max_workers=60,
    batch_size=120,
    checkpoint_interval=1,
    resume_from=None,
    checkpoint_dir_param="checkpoints",  # Added parameter
):
    start_time = time.time()
    evo_system = None
    actual_use_api_flag = (
        use_api and api_key is not None
    )  # Determine final API usage status

    if resume_from:
        print(f"Resuming from checkpoint: {resume_from}")
        evo_system = TagBasedRuleEvolutionSystem.load_checkpoint(
            checkpoint_path=resume_from,
            api_key=api_key,
            prompt_dir=prompt_dir,
            use_api=actual_use_api_flag,  # Pass determined API status
            checkpoint_dir_for_loaded_instance=checkpoint_dir_param,  # Pass the desired checkpoint_dir
        )
        mode_str = f"Resume Mode (API: {'Enabled' if evo_system.use_api else 'Disabled'}, Checkpoints: {evo_system.checkpoint_dir})"
    else:
        print(f"Loading seed rules from {input_json_file}...")
        mode_str = (
            f"API Mode (Workers: {max_workers}, Batch Size: {batch_size}, Checkpoints: {checkpoint_dir_param})"
            if actual_use_api_flag
            else f"Simulation Mode (Checkpoints: {checkpoint_dir_param})"
        )
        evo_system = TagBasedRuleEvolutionSystem(
            input_json_file=input_json_file,
            mutation_probability=0.15,
            migration_interval=3,
            elite_percentage=0.1,
            growth_factor=1.20,
            api_key=api_key,
            prompt_dir=prompt_dir,
            use_api=actual_use_api_flag,
            max_workers=max_workers,
            batch_size=batch_size,
            checkpoint_interval=checkpoint_interval,
            checkpoint_dir=checkpoint_dir_param,  # Use parameter here
        )

    print("-" * 60)
    print(f"Starting evolution for {generations} new generations ({mode_str})...")
    if evo_system:  # Ensure evo_system is initialized
        print(
            f"Will save a checkpoint every {evo_system.checkpoint_interval} generation(s) to '{evo_system.checkpoint_dir}'."
        )
    print("-" * 60)

    if evo_system:
        evo_system.evolve_multiple_generations(generations, show_progress=True)
        print("Evolution run complete. Saving results...")
        results_data = evo_system.export_to_json(
            output_filename, separate_by_island=separate_output
        )
        final_rule_count = len(evo_system.all_rules)
        total_time_elapsed = time.time() - start_time
        print("\n" + "=" * 60)
        print(f"Evolution Run Summary:")
        print(f"Final total rules: {final_rule_count:,}")
        print(
            f"Total islands: {len(evo_system.islands)}, based on {len(evo_system.tag_combinations)} unique tag combinations."
        )
        if evo_system.use_api:
            api_time_total = evo_system.api_calls.get("total_time", 0.0)
            api_calls_total_ops = evo_system.api_calls.get(
                "crossover", 0
            ) + evo_system.api_calls.get("mutation", 0)
            api_batches_total = evo_system.api_calls.get("parallel_batches", 0)
            api_success_count = evo_system.api_calls.get("success", 0)
            print(
                f"API Call Stats (Cumulative): "
                f"{evo_system.api_calls.get('crossover',0)} Crossovers, "
                f"{evo_system.api_calls.get('mutation',0)} Mutations "
                f"(Total Batches: {api_batches_total})"
            )
            if api_calls_total_ops > 0:
                print(
                    f"API Success Rate: {api_success_count}/{api_calls_total_ops} "
                    + f"({api_success_count / api_calls_total_ops * 100:.1f}%)"
                )
                print(
                    f"  Fallbacks to simulation: {evo_system.api_calls.get('fallback',0)}"
                )
            else:
                print("API Success Rate: N/A (0 API operations)")

            if api_calls_total_ops > 0:
                print(
                    f"Total API communication time: {api_time_total:.2f}s "
                    f"(Avg per op: {api_time_total/api_calls_total_ops:.2f}s)"
                )
            else:
                print(
                    f"Total API communication time: {api_time_total:.2f}s (Average: N/A)"
                )

            if (
                api_time_total > 0
                and api_batches_total > 0
                and evo_system.max_workers > 0
            ):
                effective_ops_per_batch_time = api_calls_total_ops / max(
                    1, api_batches_total
                )
                if api_batches_total > 0:
                    print(
                        f"Parallelism ratio (ops/batch): {effective_ops_per_batch_time:.1f} (Theoretical max worker influence: {evo_system.max_workers})"
                    )
        print(f"Total run time: {evo_system._format_time(total_time_elapsed)}")
        if separate_output:
            print(
                f"Results exported separately by island. Index file: {output_filename}"
            )
        else:
            print(f"All results saved to: {output_filename}")
        print("=" * 60)
        return results_data
    else:
        print("Error: Evolution system could not be initialized.")
        return None


# Lists all available checkpoints from the specified directory.
def list_checkpoints(checkpoint_dir="checkpoints"):
    # This standalone function now primarily relies on the classmethod for the actual listing logic
    # but can still be called with a specific directory.
    print(f"Listing checkpoints from directory: {checkpoint_dir}")
    checkpoints_list = TagBasedRuleEvolutionSystem.list_checkpoints(checkpoint_dir)
    if not checkpoints_list:
        # Message already printed by TagBasedRuleEvolutionSystem.list_checkpoints if dir doesn't exist or no files
        return []
    print(
        f"Found {len(checkpoints_list)} available checkpoints in {checkpoint_dir}:"
    )  # Clarified output
    print("-" * 70)
    print(f"{'Index':<6} {'Gen':<5} {'Rules':<8} {'Timestamp':<20} {'Filename'}")
    print("-" * 70)
    for i, cp_info in enumerate(checkpoints_list):
        print(
            f"{i+1:<6} {cp_info['generation']:<5} {cp_info['total_rules']:<8} {cp_info['timestamp']:<20} {os.path.basename(cp_info['checkpoint_file'])}"
        )
    print("-" * 70)
    return checkpoints_list


# Resumes evolution from a specified checkpoint, handling index or path.
def resume_from_checkpoint_action(
    checkpoint_identifier,
    generations_to_run=2,
    output_filename_on_resume="Genetic_algorithm/resumed_rules.json",
    api_key_for_resume=None,
    prompt_dir_for_resume="Prompts",
    checkpoint_dir_to_list_from="checkpoints",  # Added for clarity
    # Pass other args needed by run_tag_based_evolution if they are from CLI
    separate_output_flag=False,
    use_api_on_resume=True,
    max_workers_on_resume=20,
    batch_size_on_resume=120,
    checkpoint_interval_on_resume=1,
):
    checkpoint_file_path = None
    if isinstance(checkpoint_identifier, int):
        available_checkpoints = list_checkpoints(
            checkpoint_dir=checkpoint_dir_to_list_from
        )
        if not available_checkpoints or not (
            0 < checkpoint_identifier <= len(available_checkpoints)
        ):
            print(
                f"Invalid checkpoint index: {checkpoint_identifier}. Max index: {len(available_checkpoints)}"
            )
            return None
        checkpoint_file_path = available_checkpoints[checkpoint_identifier - 1][
            "checkpoint_file"
        ]
    elif isinstance(checkpoint_identifier, str) and os.path.exists(
        checkpoint_identifier
    ):
        checkpoint_file_path = checkpoint_identifier
    else:
        print(
            f"Invalid checkpoint identifier: {checkpoint_identifier}. Not a valid index or existing file path."
        )
        return None

    if checkpoint_file_path:
        print(f"Attempting to resume from: {checkpoint_file_path}")
        return run_tag_based_evolution(
            input_json_file=None,
            output_filename=output_filename_on_resume,
            generations=generations_to_run,
            separate_output=separate_output_flag,
            api_key=api_key_for_resume,
            prompt_dir=prompt_dir_for_resume,
            use_api=use_api_on_resume,
            max_workers=max_workers_on_resume,  # Pass through CLI args
            batch_size=batch_size_on_resume,
            checkpoint_interval=checkpoint_interval_on_resume,
            resume_from=checkpoint_file_path,
            checkpoint_dir_param=checkpoint_dir_to_list_from,  # Ensure resumed session uses consistent checkpoint_dir
        )
    return None


# Main execution block.
if __name__ == "__main__":
    args = parse_args()

    api_key_val = API_KEY
    actual_use_api = args.use_api_flag
    if not api_key_val:
        print("Warning: TOGETHER_API_KEY environment variable not found.")
        if actual_use_api:
            print("API usage will be disabled due to missing key.")
        actual_use_api = False
    else:
        if actual_use_api:
            print("TOGETHER_API_KEY found. API usage is enabled.")
        else:
            print("TOGETHER_API_KEY found, but API usage is disabled by --no_api flag.")

    if args.mode == "resume":
        if args.resume_checkpoint_path:
            print(
                f"Non-interactive resume mode from path: {args.resume_checkpoint_path}"
            )
            run_tag_based_evolution(
                input_json_file=None,
                output_filename=args.resume_output_filename,
                generations=args.resume_generations,
                separate_output=args.separate_output,
                api_key=api_key_val,
                prompt_dir=args.prompt_dir,
                use_api=actual_use_api,
                max_workers=args.max_workers,
                batch_size=args.batch_size,
                checkpoint_interval=args.checkpoint_interval,
                resume_from=args.resume_checkpoint_path,
                checkpoint_dir_param=args.checkpoint_dir,  # Pass checkpoint_dir from args
            )
        else:
            print("Interactive resume mode triggered.")
            checkpoints_found = list_checkpoints(
                checkpoint_dir=args.checkpoint_dir
            )  # Use checkpoint_dir from args
            if not checkpoints_found:
                print(
                    f"No available checkpoints found in '{args.checkpoint_dir}' for interactive resume. Cannot resume."
                )
                print(
                    "To start a new run, execute with 'run' mode or without specifying a mode."
                )
            else:
                try:
                    checkpoint_idx_input_str = input(
                        f"Please enter the index of the checkpoint to resume (1-{len(checkpoints_found)}): "
                    )
                    checkpoint_idx_input = int(checkpoint_idx_input_str)

                    generations_to_run_input_str = input(
                        f"Please enter the number of additional generations to run (default: {args.resume_generations}): "
                    )
                    generations_to_run_input = (
                        int(generations_to_run_input_str.strip())
                        if generations_to_run_input_str.strip()
                        else args.resume_generations
                    )

                    resume_from_checkpoint_action(  # Using the refactored action function
                        checkpoint_identifier=checkpoint_idx_input,
                        generations_to_run=generations_to_run_input,
                        output_filename_on_resume=args.resume_output_filename,
                        api_key_for_resume=api_key_val,
                        prompt_dir_for_resume=args.prompt_dir,
                        checkpoint_dir_to_list_from=args.checkpoint_dir,  # Pass it here
                        separate_output_flag=args.separate_output,
                        use_api_on_resume=actual_use_api,
                        max_workers_on_resume=args.max_workers,
                        batch_size_on_resume=args.batch_size,
                        checkpoint_interval_on_resume=args.checkpoint_interval,
                    )
                except (ValueError, IndexError):
                    print("Invalid input for checkpoint index or generations. Exiting.")
                except Exception as e_resume:
                    print(
                        f"Error during interactive resume setup: {e_resume}. Exiting."
                    )

    elif args.mode == "run":
        effective_output_filename = args.output_filename
        effective_generations = args.generations
        if args.resume_checkpoint_path:  # If "run" mode is used with a resume path
            print(
                f"'run' mode with --resume_checkpoint_path: Resuming from {args.resume_checkpoint_path}"
            )
            effective_output_filename = (
                args.resume_output_filename
            )  # Use resume output for this case
            effective_generations = args.resume_generations  # Use resume generations
        else:
            print("Starting a new evolution run...")

        run_tag_based_evolution(
            input_json_file=(
                args.input_json_file if not args.resume_checkpoint_path else None
            ),
            output_filename=effective_output_filename,
            generations=effective_generations,
            separate_output=args.separate_output,
            api_key=api_key_val,
            prompt_dir=args.prompt_dir,
            use_api=actual_use_api,
            max_workers=args.max_workers,
            batch_size=args.batch_size,
            checkpoint_interval=args.checkpoint_interval,
            resume_from=args.resume_checkpoint_path,
            checkpoint_dir_param=args.checkpoint_dir,  # Pass checkpoint_dir from args
        )
    else:
        print(f"Error: Unknown mode '{args.mode}'. Use 'run' or 'resume'.")
