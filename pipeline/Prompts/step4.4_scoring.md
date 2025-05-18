## Task Description
You are an expert evaluator tasked with assessing a vision puzzle consisting of 8 sub-pictures. Your goals in this prompt are:

1. **Question Reasonableness Evaluation:** Evaluate whether the provided question and answer align with the rules and are unambiguous.  
2. **Overall Visual Readability Evaluation:** Assess the clarity and readability of the entire puzzle image.

## Input
<image>
<!--SPLIT-->
</image>

<question>
{{question}}
</question>

<answer>
{{answer}}
</answer>

<rules>
{{rules}}
</rules>

## Instructions

1. **Question Reasonableness Evaluation**  
   Use the provided rules and answer to verify the reasonableness of the question. Score this on a scale of 1–5, where 5 is the most reasonable. Consider these factors:  
   - Does the answer align perfectly with the rule described?  
   - Is there any ambiguity in the question or answer?  
   - Is the provided answer the only correct solution?  

   **Score Criteria:**  
   - **5**: Perfect alignment, no ambiguity, single correct solution.  
   - **4**: Strong alignment, minor ambiguity, likely single solution.  
   - **3**: Reasonable alignment, some ambiguity or alternative interpretations.  
   - **2**: Weak alignment, ambiguous, multiple plausible answers.  
   - **1**: Poor alignment, high ambiguity, answer does not follow rules.  

2. **Overall Visual Readability Evaluation**  
   Assess the overall visual readability of the puzzle image. Score this on a scale of 1–5.  

   **Score Criteria:**  
   - **5**: Perfectly readable with no issues.  
   - **4**: Minor readability issues, but still easy to understand.  
   - **3**: Moderate readability issues that may impact understanding.  
   - **2**: Significant readability issues that hinder understanding.  
   - **1**: Severe readability issues that make the puzzle nearly impossible to understand.  

## Output
<reasonableness_evaluation>
[Your detailed justification here]
Score: [1–5]
</reasonableness_evaluation>

<readability_evaluation>
[Your detailed justification here]
Score: [1–5]
</readability_evaluation>

<final_scores>
Reasonableness: [1–5]
Readability: [1–5]
</final_scores>
