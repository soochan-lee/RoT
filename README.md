# Recursion of Thought

Official PyTorch implementation of ACL 2023 (short, findings) paper: *Recursion of Thought: A Divide and Conquer Approach to Multi-Context Reasoning with Language Models*.

- [Paper](https://arxiv.org/abs/2306.06891)
- [Poster](https://soochanlee.com/img/rot/rot_poster.pdf)


## Installation

We recommend using Anaconda.
The following command will create a new conda environment `rot` with all the dependencies.
```bash
conda env create -f environment.yml
```

To activate the environment:
```bash
conda activate rot
```

## Directory structure
```
├── configs
├── data
├── episodes
├── models
├── eval.py
├── gpt_eval.py
├── gpt_fine_tune.py
├── train_cmds
└── train.py
```
- `configs`: Model & training configurations
- `data`: Codes related to problem generation
- `episodes`: Problem configurations
- `models`: Codes for Transformer and LSTM
- `eval.py`: Evaluation logic
- `gpt_eval.py`: GPT-3 evaluation script
- `gpt_fine_tune.py`: GPT-3 fine-tuning script
- `train_cmds`: The commands used to train the models
- `train.py`: The main training logic

## GPT-3 experiments

Run the `gpt_fine_tune.py` script to create data for GPT-3 and request fine-tuning.
The API key should be present in the environment variable `OPENAI_API_KEY`.
```bash
python gpt_fine_tune.py
```

The following script will evaluate the fine-tuned models.
```bash
python gpt_eval.py
```

## Experiments with the tiny models

Use the following command to train a model.
```bash
python train.py -c [config_path] -e [episode_path] -p [paradigm] -l [log_directory]
```
`config_path` should be one of the files in `configs` directory,
and `episode_path` should be one of the files in `episodes`.
`paradigm` should be one of `wt`, `cot`, and `rot`.
In the `train_cmds` file, we provide all commands that we used to for our main experiments.
The evaluation results will be recorded in the TensorBoard summary.

