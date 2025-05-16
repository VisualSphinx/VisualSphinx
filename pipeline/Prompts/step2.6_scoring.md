## Task Description
You are an expert evaluator tasked with assessing a rule for a visual logical reasoning question. This rule is designed to guide a large language model in writing Python code to generate a sequence of images that correspond to the rule. Your evaluation will focus on three key criteria: format, content quality, and feasibility.

Before providing your final evaluation, please wrap your detailed analysis for each criterion in <detailed_analysis> tags. This will ensure a thorough and transparent assessment.

Evaluation Criteria and Rubric:

1. Format (1-5 points)
   1: Incomplete, missing multiple points, contains unrelated content
   2: Incomplete, missing 1-2 points, may contain some unrelated content
   3: Complete set of points, but may have some unrelated content
   4: Complete set of 5-6 points, minimal unrelated content
   5: Perfect format with 5-6 points and no unrelated content

2. Content Quality (1-5 points)
   1: Inconsistent, contradictory, does not describe core concept
   2: Some inconsistencies or contradictions, poorly describes core concept
   3: Mostly consistent, few contradictions, adequately describes core concept
   4: Consistent, no major contradictions, effectively describes core concept
   5: Highly consistent, no contradictions at all, perfectly describes core concept

3. Feasibility (1-5 points)
   1: Vague, unclear instructions, not suitable for code generation
   2: Some clear instructions, but many gaps for code generation
   3: Mostly clear instructions, some gaps for code generation
   4: Clear instructions, minor gaps for code generation
   5: Perfectly clear and specific instructions for easy code generation

## Input
Here is the rule you will evaluate:

<rule>
{RULE}
</rule>

## Output
Please evaluate the rule using the following structure:

<detailed_analysis>
Format Analysis:
- List key points of the rule relevant to format
- Analyze each point in relation to the format scoring rubric
- Provide a preliminary format score based on the analysis

Content Quality Analysis:
- List key points of the rule relevant to content quality
- Analyze each point in relation to the content quality scoring rubric
- Provide a preliminary content quality score based on the analysis

Feasibility Analysis:
- List key points of the rule relevant to feasibility
- Analyze each point in relation to the feasibility scoring rubric
- Provide a preliminary feasibility score based on the analysis
</detailed_analysis>

<format_evaluation>
[Your reasoning for the format score]
Score: [Your score from 1-5]
</format_evaluation>

<content_quality_evaluation>
[Your reasoning for the content quality score]
Score: [Your score from 1-5]
</content_quality_evaluation>

<feasibility_evaluation>
[Your reasoning for the feasibility score]
Score: [Your score from 1-5]
</feasibility_evaluation>

## Final Score
<format_score>
[Score]
</format_score>
<content_quality>
[Score]
</content_quality>
<feasibility>
[Score]
</feasibility>
