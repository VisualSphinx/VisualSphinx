# 🦁 VisualSphinx: Large-Scale Synthetic Vision Logic Puzzles for RL

This is the official repository for paper "[VisualSphinx: Large-Scale Synthetic Vision Logic Puzzles for RL]". 


VisualSphinx is the largest fully-synthetic open-source dataset providing vision logic puzzles. It consists of over 660K automatically generated logical visual puzzles. Each logical puzzle is grounded with an interpretable rule and accompanied by both correct answers and plausible distractors.

- 🌐 [Project Website](https://visualsphinx.github.io/) - Discover more about VisualSphinx
- 📖 [Technical Report]() - Discover the methodology and technical details behind VisualSphinx
- 🔧 [Github Repo](https://github.com/VisualSphinx/VisualSphinx) - Access the complete pipeline used to produce VisualSphinx-V1
- 🤗 HF Datasets - Find all VisualSphinx-V1 datasets
  - [VisualSphinx-V1 (Raw)](https://huggingface.co/datasets/VisualSphinx/VisualSphinx-V1-Raw);
  - [VisualSphinx-V1 (For RL)](https://huggingface.co/datasets/VisualSphinx/VisualSphinx-V1-RL-20K); 
  - [VisualSphinx-V1 (Benchmark)](https://huggingface.co/datasets/VisualSphinx/VisualSphinx-V1-Benchmark);
  - [VisualSphinx (Seeds)](https://huggingface.co/datasets/VisualSphinx/VisualSphinx-Seeds); 
  - [VisualSphinx (Rules)](https://huggingface.co/datasets/VisualSphinx/VisualSphinx-V1-Rules). 

## Overview
![VisualSphinx](https://visualsphinx.github.io/static/images/pipeline-mini.jpg)
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


## Features
VisualSphinx is a comprehensive pipeline designed to generate large-scale, diverse, and verifiable synthetic datasets for vision logic puzzles. Key features include:
- **Diverse Generation**: Automatically produces high-quality visual logic puzzles from a variety of sources and rule templates, supporting multiple puzzle styles and formats.
- **Self-Verification**: Each puzzle is accompanied by correct answers and plausible distractors, with automated verification and scoring to ensure quality.
- **Open & Reproducible**: All code, prompts, and data processing steps are open-source and fully documented for reproducibility and community extension.


## Training

Please refer to [verl](https://github.com/volcengine/verl) for RL training using VisualSphinx datasets, which is based on .

## Other Information

**License**: Please follow [MIT](https://choosealicense.com/licenses/mit/).

**Contact**: For questions, suggestions, or feedback, please reach out to [Yichen](mailto:yfeng42@uw.edu), or [raise an issue](https://github.com/VisualSphinx/VisualSphinx/issues/new). We welcome your input and are committed to continuously improving VisualSphinx to better serve the community.

## Citation

If you find the model, data, or code useful, please cite:
```
@article{
}
```