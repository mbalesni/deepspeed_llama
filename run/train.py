import wandb
import os
import argparse
import json
import deepspeed  # type: ignore
from argparse import Namespace
from typing import Dict
from deepspeed_llama.common import attach_debugger, project_dir
from deepspeed_llama.models.llama import get_llama_hf_model
from deepspeed_llama.huggingface import get_compute_metrics_fn, get_datasets, train


def main(project: str, name: str, config: Dict, args: Namespace):

    wandb.init(project=project, name=name, config=config, group=name) # NOTE: this will log `n_gpus` W&B runs at once, to see them nicely, group the runs on W&B by group

    train_path = wandb.config.train_path
    validation_path = wandb.config.validation_path
    data_dir = os.path.join(project_dir, wandb.config.data_dir)
    deepspeed_config = os.path.join(project_dir, wandb.config.deepspeed_config)

    wandb.config.update({
        "train_path": train_path,
        "validation_path": validation_path,
        "data_dir": data_dir,
        "deepspeed_config": deepspeed_config
    }, allow_val_change=True)
    model, tokenizer = get_llama_hf_model(wandb.config.model_name)

    datasets, tokenizer = get_datasets(tokenizer=tokenizer, verbose=args.logging, num_retries=args.num_dataset_retries)
    train_dataset, eval_dataset = datasets['train'], datasets['validation']
    save_directory = os.path.join(os.path.dirname(args.file), f"{args.job_id}_{args.task_id}_results")
    print(f"Saving metrics and model output to {save_directory}")
    compute_metrics = get_compute_metrics_fn(tokenizer, eval_dataset, save_directory)

    
    train(model, train_dataset, eval_dataset, compute_metrics, tokenizer, 
            verbose=args.logging, save_model_dir=save_directory)

    wandb.finish()


if __name__ == "__main__":
    # TODO: This should be a self-contained script, such that it can be ran independently of the rest of the codebase (and in particular, independently of SLURM).
    # This would mean moving everything to args which can be passed in and having a separate script for calling it from SLURM.

    parser = argparse.ArgumentParser()

    parser.add_argument("--project", type=str, required=True)  # TODO: Add descriptions to all of the arguments
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument('--local_rank', type=int, default=0, help='local rank passed from distributed launcher')
    parser.add_argument("--job_id", type=int, required=True)
    parser.add_argument("--task_id", type=int, required=True)
    parser.add_argument("--logging", type=str, default=True)
    parser.add_argument("--num_dataset_retries", type=int, default=3)
    parser.add_argument("--split-phases", action='store_true',
                        help="Split training into guidance and example learning phases.")
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--debug_port", type=int, default=5678)

    deepspeed.add_config_arguments(parser)  # TODO: is this needed?

    args = parser.parse_args()

    if args.debug:
        # check if main process via env
        if os.environ.get('LOCAL_RANK', '0') == '0':
            attach_debugger(args.debug_port)

    config = json.load(open(args.file, 'r'))[args.task_id]

    main(project=args.project,
         name=f"{config['experiment_name']} ({args.job_id}_{args.task_id})", config=config, args=args)
