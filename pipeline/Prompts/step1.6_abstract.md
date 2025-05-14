## Task Description
You are an expert in analyzing and summarizing complex vision logic puzzles. Your task is to extract and present the key regularities and patterns from a given puzzle and its explanation. The goal is to create a concise list of regularities that captures the essential elements of the puzzle and its solution, which could be used to generate similar puzzles.

Now, follow these steps to analyze the puzzle and extract key regularities:

1. **Analyze** the puzzle and its explanation thoroughly.  
2. **Create** a detailed breakdown of the puzzle inside `<detailed_analysis>` tags …  
3. **Present** your analysis within `<puzzle_breakdown>` tags …  
4. **Create** a list of key regularities within `<key_points>` tags …  
5. **Review & refine** your list …  

---

### Example

Examine the image of the puzzle:  
<image><!--EXAMPLE_SPLIT--></image>

Read the question:  
<puzzle_question>  
From the given four options, select the most suitable one to fill in the question mark to present a certain regularity:  
</puzzle_question>

Review the options:  
<puzzle_options>  
"A": "A",  
"B": "B",  
"C": "C",  
"D": "D"  
</puzzle_options>

Read the explanation:  
<puzzle_explanation>  
The elements have similar compositions … *[omitted for brevity]*  
</puzzle_explanation>

Note the answer:  
<puzzle_answer>  
D  
</puzzle_answer>

The model answer is:

<detailed_analysis></detailed_analysis>
<puzzle_breakdown></puzzle_breakdown>
<key_points>
- Inner shapes become outer shapes in the subsequent panel  
- Shapes alternate between curved and straight-edged forms  
- Each panel contains exactly two nested geometric shapes  
- Multiple pattern layers must be satisfied simultaneously  
- Shape orientation varies while maintaining structural patterns  
</key_points>

---

## Input

Examine the image of the puzzle:  
<image>
<!--PUZZLE_SPLIT-->
</image>

Read the question:  
<puzzle_question>  
{{ prompt }}  
</puzzle_question>

Review the options:  
<puzzle_options>  
{{ options_block }}  
</puzzle_options>

Read the explanation:  
<puzzle_explanation>  
{{ explanation }}  
</puzzle_explanation>

Note the answer:  
<puzzle_answer>  
{{ correct_answer }}  
</puzzle_answer>

---

## Output (your turn)

Your final output **must** follow this exact structure:

<detailed_analysis>
[Your detailed analysis here]
</detailed_analysis>

<puzzle_breakdown>
[Your structured breakdown here]
</puzzle_breakdown>

<key_points>
- [Regularity 1]  
- [Regularity 2]  
- … (max 5 items, each ≤ 30 words)  
</key_points>
