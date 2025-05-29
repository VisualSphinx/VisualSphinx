# 🦁 VisualSphinx: Large-Scale Synthetic Vision Logic Puzzles for RL

This is the official repository for paper "[VisualSphinx: Large-Scale Synthetic Vision Logic Puzzles for RL]". 


VisualSphinx is the largest fully-synthetic open-source dataset providing vision logic puzzles. It consists of over 660K automatically generated logical visual puzzles. Each logical puzzle is grounded with an interpretable rule and accompanied by both correct answers and plausible distractors.

- 🌐 [Project Website](https://visualsphinx.github.io/) - To discover more about VisualSphinx
- 📖 [Technical Report]() - Discover the methodology and technical details behind VisualSphinx
- 🔧 [Github Repo](https://github.com/VisualSphinx/VisualSphinx) - Access the complete pipeline used to produce VisualSphinx-V1
- 🤗 HF Datasets:
  - [VisualSphinx-V1 (Raw)](https://huggingface.co/datasets/VisualSphinx/VisualSphinx-V1-Raw);
  - [VisualSphinx-V1 (For RL)](https://huggingface.co/datasets/VisualSphinx/VisualSphinx-V1-RL-20K); 
  - [VisualSphinx-V1 (Benchmark)](https://huggingface.co/datasets/VisualSphinx/VisualSphinx-V1-Benchmark);
  - [VisualSphinx (Seeds)](https://huggingface.co/datasets/VisualSphinx/VisualSphinx-Seeds); 
  - [VisualSphinx (Rules)](https://huggingface.co/datasets/VisualSphinx/VisualSphinx-V1-Rules). 

## Overview
![VisualSphinx](https://visualsphinx.github.io/static/images/pipeline.jpg)
![performance](https://visualsphinx.github.io/static/images/performance.png)

## Installation

**Build environment**
```
git clone https://github.com/VisualSphinx/VisualSphinx-Generator.git
cd VisualSphinx-Generator
conda create -n VisualSphinx python=3.12 -y
conda activate VisualSphinx
pip install -r requirements.txt
```

## Generate Data
Please go into [pipeline](/pipeline) for reproduce VisualSphinx. Please do not forget to define your API-Keys in [api_config.py](pipeline/api_config.py).


<!-- ## Features
KodCode is a comprehensive pipeline designed to generate diverse, challenging, and verifiable synthetic datasets for coding tasks. Key features include:
- **Diverse Sources:** Generate high-quality coding questions from multiple sources including zero-shot generation, human-written assessment questions, code snippets, and technical documentation - **all unified in a single framework!**
- **Self-Verification:** Generate verifiable solutions and tests for each coding question. Support pytest and parallel execution.
- **Style Converter:** Easy to convert between different styles of coding questions. -->


## Training

Please refer to [verl](https://github.com/volcengine/verl) for RL training using VisualSphinx datasets, which is based on .

## 🧐 Other Information

**License**: Please follow [MIT](https://choosealicense.com/licenses/mit/).

**Contact**: For questions, suggestions, or feedback, please reach out to [Yichen](mailto:yfeng42@uw.edu), or [raise an issue](https://github.com/VisualSphinx/VisualSphinx/issues/new). We welcome your input and are committed to continuously improving VisualSphinx to better serve the community.

## 📚 Citation

If you find the model, data, or code useful, please cite:
```
@article{
}
```