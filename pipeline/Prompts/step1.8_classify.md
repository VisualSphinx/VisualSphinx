## Task Description
You are an expert system designed to analyze and classify complex visual logic puzzles.  
Your task is to examine the given puzzle components and determine **one** question‑type tag and **one** knowledge‑point tag from the predefined lists.

### Question‑type tags
- **Nine-square grid** A 3 × 3 grid with one cell containing a question mark.  
- **Horizontal square** A row of 4 – 6 squares with one containing a question mark.  
- **Two-group** Two groups of three squares; one square of one group has a question mark.  
- **Two set of number** Images with numbers that must be divided into two sets.  
- **Others** Doesn’t fit any category above.

### Knowledge‑point tags
- **Correlated** Each image is directly related to the previous one.  
- **Summarize** All images share an overall rule but adjacent images may not correlate.  
- **Others** Doesn’t fit any category above.

---

## Analysis instructions
1. Examine every puzzle component.  
2. Focus on structure and relationships.  
3. Weigh arguments **for and against** each tag.  
4. Provide detailed reasoning inside `<puzzle_breakdown>` before giving tags.

---

## Input

Here is the puzzle you need to analyze:

<puzzle_image>
<!--PUZZLE_SPLIT-->
</puzzle_image>

<puzzle_question>
{{ prompt }}
</puzzle_question>

<puzzle_options>
{{ options_block }}
</puzzle_options>

<puzzle_explanation>
{{ explanation }}
</puzzle_explanation>

<puzzle_answer>
{{ correct_answer }}
</puzzle_answer>

---

## Output (required structure)

<puzzle_breakdown>
[Your detailed reasoning here]
</puzzle_breakdown>

<question_type>
[Selected question‑type tag]
</question_type>

<knowledge_point>
[Selected knowledge‑point tag]
</knowledge_point>
