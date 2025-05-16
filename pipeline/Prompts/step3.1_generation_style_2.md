## Task Description
As a Python imaging expert, create two Python scripts: one generating 5 continuous images that follow a set of rules, and another generating 3 images that intentionally break those rules. Infer a reasonable implementation according to rules (e.g., based on patterns, shapes, or mathematical properties) and document your reasoning.

## Input
Let's review the rules for your images:

<rules>
{{RULES}}
</rules>

## Guidelines
- **Libraries**: Use PIL and numpy.
- **Style**: Black-and-white images, no text.
- **Output Paths**: 
  - Correct images: ./output_correct/
  - Incorrect images: ./output_incorrect/
- **Image Counts**: 5 correct, 3 incorrect.
- **Size**: Adjust the size of elements and images to avoid confusion and unnecessary overlap.
- Optionally, sometimes you can use your continuous output images to express some rules that are related between images instead of expressing all in a single one image.
  
## Deliverables
Provide two complete, runnable Python scripts:
1. **Correct Script**: Generates 5 continuous images complying the rules.
2. **Incorrect Script**: Generates 3 images, each breaking a different rule.

For each script:
- Include all imports.
- Your code should well-documented and add concise comments explaining rule compliance or violation.
- Use relative paths ./output_correct and ./output_incorrect.

## Output
Generate two Python scripts as requested. Present them *exactly* in the following format, with only the raw code inside the tags:

<correct_script>
*Your complete Python script for generating 5 correct images*
</correct_script>

<incorrect_script>
*Your complete Python script for generating 3 incorrect images*
</incorrect_script>

Remember, two scripts should be complete and directly runnable.
