# VisualSphinx Pipeline

Here, we provide our pipeline for generating VisualSphinx dataset. Please ensure you are in the `pipeline` folder when running the commands.

## Environment Setup

1. Python version `3.12`
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure API Key:
   - Edit `api_config.py` and set `TOGETHER_API_KEY`, `OpenAI_API_KEY`, `XAI_API_KEYS` and `Anthropic_API_KEY`.

## Step 1: Seed Question Collection & Rule Abstraction

### Step 1.1: Pull Seed Questions

Pull all raw seed questions. There are 4K visual logic questions together with their authored explanations from Chinese Civil Service Examination.
```bash
python step1.1_pull_data.py
```
- Output: `./data/step1/1.1_seed_questions.json`

### 1.2 Question Rewriting (Claude)

Translate all explanations into English, rewrite them to eliminate answer leakage, and enrich them with clarifying
details.

```bash
python step1.2_claude_rewrite.py
```
- Input: `1.1_seed_questions.json`
- Output: `1.2_rewritten_questions.json`

### 1.3 Merge Rewriting Results

Merge into a new json.

```bash
python step1.3_merge.py
```
- Input: Multiple rewritten JSON files (from different annotators or models)
- Output: `./data/step1/1.3_merged_rewrites.json`

### 1.4 Verification (Claude)

Perform a consistency check by feeding each image–explanation pair to Claude to verify correct answer retrieval.

```bash
python step1.4_claude_verify.py
```
- Input: `./data/step1/1.3_merged_rewrites.json`
- Output: `./data/step1/1.4_verified_rewrites.json`

### 1.5 Filter Correct Questions

Filter last step questions.

```bash
python step1.5_filter_correct.py
```
- Input: `./data/step1/1.4_verified_rewrites.json`
- Output: `./data/step1/1.5_filtered_questions.json`

### 1.6 Rule Abstraction (Claude)

Each of these remaining questions is then abstracted into a structured seed rule using Claude, comprising five concise bullet points capturing the core visual pattern.

```bash
python step1.6_claude_rule_abstraction.py
```
- Input: `./data/step1/1.5_filtered_questions.json`
- Output: `./data/step1/1.6_abstracted_rules.json`

### 1.7 Rule Extraction

Process last step results.

```bash
python step1.7_rule_extract.py
```
- Input: `./data/step1/1.6_abstracted_rules.json`
- Output: `./data/step1/1.7_extracted_rules.json`

### 1.8 Rule Categorization (Claude)

Categorized into 8 classes with Claude based on their visual patterns and required reasoning skills. 

```bash
python step1.8_claude_rule_category.py
```
- Input: `./data/step1/1.7_extracted_rules.json`
- Output: `./data/step1/1.8_categorized_rules.json`

### 1.9 Merge All Rules

Merge into a new json.

```bash
python step1.9_merge.py
```
- Input: All categorized rule files (e.g., from different categories)
- Output: `./data/step1/1.9_all_rules.json`

### 1.10 Manually Check

Use manually checking tool to ensure there are a total of 8 categories. If there are more categories, refer to Appendix B.1 of paper for manual classification.

```bash
python step1_finally_manually_check_tool.py
```

---

## Step 2: Rule Processing & Deduplication

### 2.1 Rule Grouping

Split into two groups.

```bash
python step2.1_classify_two_groups.py
```
- Input: `./data/step1/1.9_all_rules.json`
- Output: `./data/step2/2.1_grouped_rules.json`

### 2.2 Rule Mutation Generation (DeepSeek V3)

We introduce a *rule-level genetic algorithm*.
Each class of seed rules forms a subpopulation, evolving independently on separate islands through
genetic operations: Mutation rewrites, adds, or deletes individual bullets, while crossover interleaves
bullets from two parent rules. Every three generations, 10% of the rules migrate across islands to
maintain diversity. 

```bash
bash step2.2__bash.sh
# Or run individually
```
- Input: Two grouped rules
- Output: Two Group of mutated rules

### 2.3 Merge Rules

Merge two groups.

```bash
python step2.3_merge.py
```
- Input: Original and mutated rules
- Output: `./data/step2/2.3_rules.json`

### 2.4 Rule Deduplication (Faiss Similarity)

We use the all-mpnet-base-v2 embedding model to project all rules into an embedding space and compute nearest-neighbor distances using FAISS [9]. Rules exceeding a predefined similarity threshold with existing entries are removed to eliminate redundancy and promote diversity.

```bash
python step2.4_dedupulicate.py
```
- Input: `./data/step2/2.3_rules.json`
- Output:
  - Deduplication results: .`/data/step2/deduplication/2.3_rules_duplicates.jsonl`
  - Faiss index: `./data/step2/deduplication/2.3_rules.faiss`

### 2.5 Deduplication Statistics

Check deduplication statistics.

```bash
python step2.5_count_deduplication.py
```
- Input: `./data/step2/deduplication/2.3_rules_duplicates.jsonl`
- Output: `./data/step2/deduplication/2.5_deduplication_stats.json`

### 2.6 Rule Scoring (DeepSeek V3)

 Use DeepSeek V3 to rank the remaining rules based on three criteria: **format**(adherence to the structured five-bullet-point template), **content quality** (clarity and logical coherence of the rule), and **feasibility** for code generation (suitability for generating Python scripts to render images)

```bash
python step2.6_ds-v3_scoring.py
```
- Input: `./data/step2/deduplication/2.3_rules_duplicates.jsonl`
- Output:
  - Scored rules: `./data/step2/2.6_rules_scoring.json`
  - Checkpoint: `./data/scoring_checkpoint/scoring_checkpoint.pkl`

### 2.7 Score Analysis

Check score.

```bash
python step2.7_analyze_score.py
```
- Input: `./data/step2/2.6_rules_scoring.json`
- Output: `./data/step2/2.7_score_analysis.json`

### 2.8 Rule Filtering

Filter rules based on deduplication and scores.

```bash
python step2.8_filter_rules.py
```
- Input: `./data/step2/2.6_rules_scoring.json`
- Output: `./data/step2/2.8_final_rules.json`

---

## Step 3: Puzzle Generation & Filtering

### 3.1 Puzzle Generation (Grok)

Use three styles to generate two Python scripts in a single turn: `correct_script.py` renders five sequential images that adhere to the rule, while `incorrect_script.py` produces three plausible but rule-violating
distractors.

```bash
bash step3.1__bash.sh
# Or run individually
```
- Input: Rule set
- Output: Puzzles in different styles

### 3.2 Puzzle Counting

Count valid puzzles

```bash
bash step3.2__bash.sh
```
- Input: Puzzle sets
- Output: Valid puzzles list

### 3.3 Image Deduplication & Filtering

We use Perceptual Hashing (pHash) to compute hash values for each image within a group, measuring Hamming distances between all pairs. Images with distances below 10 are removed as duplicates. To identify low-quality images (e.g., those whose figures are too small), we compute the Structural Similarity Index (SSIM) against a white reference image, flagging images with scores below 0.1 as blank, and calculate grayscale gradient energy, discarding images below a calibrated threshold.

```bash
bash step3.3__bash.sh
```

- Input: Puzzles and images
- Output: Deduplicated puzzles meta data

Then check the statistics.

```bash
python step3.3.2_count_valid.py
```

---

## Step 4: Option Assembly & Dataset Merging

### 4.1 Assemble Puzzles

Assemble Puzzles according three different methods.

```bash
bash step4.1__bash.sh
# Or run individually
```
- Input: Puzzle groups
- Output: Puzzles

### 4.2 - 4.3 Process Puzzles

Add more information.

```bash
bash step4.2__bash.sh
# Or run individually

bash step4.3__bash.sh
# Or run individually

```
- Input: Puzzles
- Output: Puzzles

### 4.4 Scoring Puzzles (GPT)

Use GPT 4.1 to evaluates visual readability and logical coherence

```bash
bash step4.4__bash.sh
# Or run individually
```
- Input: Puzzles
- Output: Scores

### 4.5 Pass Rate

Use base model to do a pass rate

```bash
bash step4.5__bash.sh
# Or run individually
```
- Input: Puzzles
- Output: Pass rate

### 4.6 - 4.7 Merge Data

Merge scores and pass rate.

```bash
bash step4.6__bash.sh
bash step4.7__bash.sh
```
- Input: Puzzles and last two steps results
- Output: Puzzles


### 4.8 Option Shuffling & 10-Option Merging

Add information to other assemble methods.

```bash
bash step4.8__bash.sh
# Or run individually
```
- Input: 4/10-option puzzles
- Output: puzzles

### 4.9 Final Dataset Merging

For each assemble method, merge all different styles into one whole dataset.

```bash
bash step4.9__bash.sh
# Or run individually
```
- Input: Each style's puzzles and images
- Output: Final `questions.json` and `images/` folder

---

## Notes

- Some steps require GPU or cloud API access; ensure your environment is properly configured.
- Make sure your data directory structure matches the script parameters.
- If you encounter errors, check paths, dependencies, and API Key configuration first.

---