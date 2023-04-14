# Fine-tune LLaMA with DeepSpeed

> **WARNING**: This code was not tested **at all**. This was adapted from a private repo blindly without ever having been run. Expect it to not run from the first few attempts and to have numerous bugs.

This directory contains code to fine-tune a LLaMA model with DeepSpeed on a compute cluster. 

It assumes that you have access to a compute cluster with a SLURM scheduler and access to the LLaMA model weights. In particular, the path to the model is currently hardcoded.

## Installation

```bash
git clone git@github.com:nikebless/deepspeed_llama.git
cd deepspeed_llama
pip install -e .
```

## Fine-tuning

1. First, add your W&B API key to to the environment:

```bash
export WANDB_API_KEY=your_api_key
```

2. To fine-tune a 13B model, run the following command:

```bash
python run/sweep.py --experiment_name "testing" --config_file experiments/example_sweeps/13b.yaml
```

This will run a sweep of experiments defined in `experiments/example_sweeps/13b.yaml` and log the learning curves and results to W&B.

## Requirements

Only Linux.

The hardware requirements aren't clear for now. This was only run on 80GB A100 GPUs. Only one GPU should be necessary for the finetuning, but more GPUs will speed up the training significantly. 

**NB:** RAM usage scales with the number of GPUs. E.g. LLaMA-13B loaded in BF16 takes up ~26GB of RAM per GPU before being transferred to the GPU. This way, fine-tuning a 30B model on 8xA100 requires at least 480GB of RAM, with some overhead (to be safe, I'd say you should have 600GB.)

## Training Speed

The following table shows the training speed of LLaMA on 8xA100 on our cluster.

| Model | Batch Size (total) | Dataset Size | Time per Epoch |
| --- | --- | --- | --- |
| 13B | 32 | ~13K documents (0.5M tokens) | 8min (+3min initialization) |
| 30B | 32 | ~13K documents (0.5M tokens) | 18min (+3min initialization) |
