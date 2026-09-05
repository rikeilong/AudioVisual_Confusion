# AVConfusion
[AAAI’26] Official Implementation for When Eyes and Ears Disagree: Can MLLMs Discern Audio-Visual Confusion?

[![中关村学院 GitHub 组织](https://github.com/bjzgcai)](https://github.com/bjzgcai)

## 1. Overview

This project investigates whether multimodal large language models can correctly interpret audio when visual and acoustic evidence conflict, rather than hallucinating sounds associated with visible objects.

The paper introduces **AV-ConfuseBench**, which contains two evaluation settings:

- **Audio-Muted (Task 1):** The sound of a visible object or instrument is removed. The model must determine whether that sound is still present.
- **Audio-Modified (Task 2):** The original soundtrack is replaced with audio that is inconsistent with the visual content. The model must separately describe what it sees and what it hears.

The paper also proposes **RL-CoMM**, a reinforcement-learning framework built on Qwen2.5-Omni to reduce visually induced audio hallucinations. It contains two main components:

- **Step-wise Reasoning Reward (Step-RR):** Evaluates audio reasoning, visual reasoning, and the final answer at separate stages.
- **Answer-centered Confidence Optimization (Ans-CO):** Increases confidence in correct answers while suppressing visually biased incorrect answers.

## 2. Repository Structure

```text
AudioVisual_Confusion/
├── AVConfuseBench/
│   ├── avconfusebench_test_m1.json     # Task 1 annotations
│   ├── avconfusebench_test_m2.json     # Task 2 annotations
│   ├── test_script/                    # Task 1 inference with Qwen/Gemini
│   └── test_script_m2/                 # Task 2 inference and GPT evaluation
├── src/
│   ├── RL-CoMM/
│   │   ├── prepocess_datast.py         # Feature preprocessing (original filename)
│   │   ├── train_st1.py                # Stage-1 Step-RR training entry point
│   │   └── trainer/                    # Custom GRPO trainer and reward logic
│   └── llamafactory/                   # LLaMA-Factory-related code
├── warm-up-phase/
│   └── data.json                       # Warm-up training data example
├── demo/                               # Demonstration assets
├── requirements.txt
└── setup.py
```

> The current repository mainly provides annotation files, evaluation scripts, and the first-stage RL-CoMM training code. The full AV-ConfuseBench videos, model weights, and training datasets are not committed to the repository and must be obtained separately.

## 3. Environment Setup

The recommended environment is Linux, Python 3.10, an NVIDIA GPU, and CUDA 11.8 or later. A GPU with BF16 support is recommended. If GPU memory is limited, consider a smaller model, quantization, shorter inputs, or lower batch sizes.

```bash
git clone https://github.com/rikeilong/AudioVisual_Confusion.git
cd AudioVisual_Confusion

conda create -n avconfusion python=3.10 -y
conda activate avconfusion

python -m pip install --upgrade pip
pip install -e ".[torch]"
pip install qwen-omni-utils qwen-vl-utils
```

The provided inference scripts enable FlashAttention 2. If your hardware and CUDA environment are compatible, install it with:

```bash
pip install flash-attn --no-build-isolation
```

If the installation fails, remove the following argument from the inference scripts:

```python
attn_implementation="flash_attention_2"
```

Transformers will then use its default attention implementation.

## 4. Models and Data

The code references the following Hugging Face models:

- Benchmark evaluation: `Qwen/Qwen2.5-Omni-7B`
- RL-CoMM policy model: `Qwen/Qwen2.5-Omni-3B`
- Audio reference model: `Qwen/Qwen2.5-Audio-7B-Instruct`

A suggested directory layout is shown below:

```text
AudioVisual_Confusion/
├── datasets/
│   └── AVConfuseBench/
│       ├── avconfusebench_test_m1.json
│       ├── avconfusebench_test_m2.json
│       ├── confused_video/             # Task 1 videos
│       └── confused_video_m2/          # Task 2 videos
└── checkpoints/
    ├── Qwen2.5-Omni-7B/
    ├── Qwen2.5-Omni-3B/
    └── Qwen2.5-Audio-7B-Instruct/
```

Several scripts contain absolute paths or relative paths that differ from this layout. Before running the code, locate and update all model, dataset, video, feature, and output paths:

```bash
grep -R "model_dir\|dataset_path\|video_folder\|output" AVConfuseBench src/RL-CoMM
```

On Windows PowerShell, use:

```powershell
Get-ChildItem AVConfuseBench,src/RL-CoMM -Recurse -File |
  Select-String 'model_dir|dataset_path|video_folder|output'
```

## 5. Running AV-ConfuseBench

### 5.1 Task 1: Audio-Muted

Open `AVConfuseBench/test_script/qwen_omni_7b.py` and configure:

- `model_dir`: the path to the Qwen2.5-Omni-7B checkpoint;
- the test annotation file: `avconfusebench_test_m1.json`;
- the video directory: `confused_video`;
- the output result path.

Run the script from the repository root:

```bash
python AVConfuseBench/test_script/qwen_omni_7b.py
```

Task 1 annotations use fields similar to:

```json
{
  "video_id": "video identifier",
  "question": "question text",
  "answer": "yes/no",
  "instruments": ["target sound source"],
  "instruments_len": 1
}
```

### 5.2 Task 2: Audio-Modified

Configure the model, annotation, video, and output paths in `AVConfuseBench/test_script_m2/qwen_omni_7b.py`, then run:

```bash
python AVConfuseBench/test_script_m2/qwen_omni_7b.py
```

Task 2 annotations use fields similar to:

```json
{
  "video_id": "video identifier",
  "filename": "video filename",
  "question": "question text",
  "sound_type": "sound category",
  "annotation": "human annotation"
}
```

### 5.3 Gemini Evaluation (Optional)

`AVConfuseBench/test_script/gemini.py` contains entry points for both Task 1 and Task 2. Before running it, provide a compatible `API_KEY` and `BASE_URL`, and verify the annotation and video paths near the bottom of the file:

```bash
python AVConfuseBench/test_script/gemini.py
```

To avoid committing credentials, modify the script to read them from environment variables:

```python
import os

API_KEY = os.environ["GEMINI_API_KEY"]
BASE_URL = os.environ.get("GEMINI_BASE_URL", "")
```

### 5.4 Automated Evaluation of Task 2 Outputs

The evaluation entry point is `AVConfuseBench/test_script_m2/gpt_eval.py`. This file imports `config.QUESTION_PROMPT`, but the corresponding configuration file is not included in the current repository snapshot.

Before running the evaluator, create a local `config.py`, define the evaluation prompt, and configure a compatible API endpoint and key. Do not commit private credentials to a public repository.

## 6. RL-CoMM Data Preprocessing

Open `src/RL-CoMM/prepocess_datast.py` and update at least the following values:

- the Qwen model path;
- the Music-AVQA training annotation path;
- the source video directory;
- the audio and video feature output directories;
- the placeholder in `process_feats_omni('...')` at the end of the file.

Run preprocessing with:

```bash
python src/RL-CoMM/prepocess_datast.py
```

After successful preprocessing, each sample directory should contain:

```text
sample_id/
├── audio_inputs.pt
└── video_inputs.pt
```

## 7. Citation

```bibtex
@misc{ye2025eyesearsdisagreemllms,
  title         = {When Eyes and Ears Disagree: Can MLLMs Discern Audio-Visual Confusion?},
  author        = {Qilang Ye and Wei Zeng and Meng Liu and Jie Zhang and Yupeng Hu and Zitong Yu and Yu Zhou},
  year          = {2025},
  eprint        = {2511.10059},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2511.10059}
}
```

## 8. Links

- Repository: https://github.com/rikeilong/AudioVisual_Confusion
- AAAI paper: https://ojs.aaai.org/index.php/AAAI/article/view/38183
- arXiv: https://arxiv.org/abs/2511.10059
