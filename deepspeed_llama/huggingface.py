import pandas as pd
import re
import json
import os
import torch
import wandb
import time
import deepspeed  # type: ignore
import random
import numpy as np
from argparse import Namespace
from typing import Dict, Union, Tuple, Callable, Optional, Literal, List

from transformers import (Seq2SeqTrainer, Trainer,
                          Seq2SeqTrainingArguments, EvalPrediction, PreTrainedTokenizer,
                          PreTrainedTokenizerFast, PreTrainedModel, DataCollatorWithPadding)
from datasets.arrow_dataset import Dataset
from deepspeed_llama.evaluation import evaluate_completions_exact_match
from deepspeed_llama.dataset import get_hugface_datasets
import math
import os

freeze_types = ["decoder", "mlp", "final_layers", "all", "none"]
FREEZE_TYPE = Literal["decoder", "mlp", "final_layers", "all", "none"]
TTokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]


def safe_save_model_for_hf_trainer(trainer: Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def get_compute_metrics_fn(tokenizer: TTokenizer, eval_dataset: Dataset, directory_path: str):

    def find_latest_file_version(directory_path, file_prefix):
        file_regex = re.compile(f"{file_prefix}_(\\d+)")
        max_version = -1

        for filename in os.listdir(directory_path):
            match = file_regex.match(filename)
            if match:
                version = int(match.group(1))
                if version > max_version:
                    max_version = version
        return max_version

    def save_files(df, metrics):

        # Create the directory if it doesn't exist
        if not os.path.exists(directory_path):
            os.makedirs(directory_path, exist_ok=True)

        # Save the DataFrame as a CSV file in the created directory
        step = find_latest_file_version(directory_path, f"df") + 1
        csv_file_path = os.path.join(directory_path, f"df_{step}.csv")
        df.to_csv(csv_file_path, index=False)

        # Save the dictionary as a JSON file in the created directory
        step = find_latest_file_version(directory_path, f"metrics") + 1
        json_file_path = os.path.join(directory_path, f"metrics_{step}.json")
        with open(json_file_path, "w") as json_file:
            json.dump(metrics, json_file)

    def compute_metrics(eval_preds: EvalPrediction) -> Dict:
        predictions = eval_preds.predictions
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        pred_tokens = torch.argmax(torch.tensor(predictions), dim=-1)
        label_tokens = eval_preds.label_ids
        assert isinstance(label_tokens, np.ndarray), "Typing screams if it's a tuple"

        label_tokens[label_tokens == -100] = 0

        prompts = [x["prompt"] for x in eval_dataset] # type: ignore
        completions = [x["completion"] for x in eval_dataset] # type: ignore


        # Select the tokens that are are completion from the model predictions
        outs_decoded = tokenizer.batch_decode(pred_tokens)
        preds = [pred.replace(tokenizer.pad_token, "").replace(prompt, "") for pred, prompt in zip(outs_decoded, prompts)]
        prompts = [x.replace(tokenizer.pad_token, "") for x in prompts]
        labels = completions

        eval_results = evaluate_completions_exact_match(Namespace(verbose=False), preds, labels)
        is_correct_list = eval_results["is_correct_list"]

        df = pd.DataFrame({'prompt': prompts, 'labels': labels, 'preds': preds, 'correct': is_correct_list, 'model_outs': outs_decoded})

        metrics = {}

        wandb.log({"validation_examples": wandb.Table(dataframe=df)})
        
        accuracy = eval_results["accuracy"]
        metrics["accuracy"] = accuracy
        wandb.log({"validation_accuracy": accuracy})
        rank = int(os.environ["RANK"])
        if rank == 0:
            save_files(df, metrics)
        return metrics

    return compute_metrics


def get_datasets(tokenizer, num_retries: int, verbose: bool) -> Tuple[Dict[str, Dataset], TTokenizer]:

    if verbose:
        print("Loading tokenizer and generating datasets")

    train_dataset = None
    eval_dataset = None
    for i in range(num_retries):
        try:
            train_path = os.path.join(wandb.config.data_dir, wandb.config.train_path)
            validation_path = os.path.join(wandb.config.data_dir, wandb.config.validation_path)
            train_dataset, eval_dataset = get_hugface_datasets(train_path, validation_path, tokenizer)
            break
        except Exception as e:
            print("Failed to generate datasets, retrying")
            print(e.args)
            time.sleep(random.randint(1, 10))
            if i == num_retries - 1:
                raise e

    if not train_dataset or not eval_dataset:
        raise ValueError("Failed to generate datasets")

    print("Generated dataset")

    if wandb.config.randomise_data_order:
        train_dataset = train_dataset.shuffle()

    datasets = {}
    datasets["train"] = train_dataset
    datasets["validation"] = eval_dataset

    return datasets, tokenizer


def log(string, verbose):
    if verbose:
        print(string)


def get_deepspeed_config(use_deepspeed: bool, verbose: bool) -> Optional[str]:
    if use_deepspeed:
        deepspeed_config = wandb.config.deepspeed_config
        if verbose:
            print("Using deepspeed")
    else:
        deepspeed_config = None

    return deepspeed_config


def train(model: PreTrainedModel, train_dataset: Dataset, eval_dataset: Dataset, compute_metrics: Callable, tokenizer: TTokenizer, verbose: bool, save_model_dir: Optional[str]):

    deepspeed_config = get_deepspeed_config(wandb.config.deepspeed, verbose)

    training_args = Seq2SeqTrainingArguments(
        output_dir=wandb.config.output_dir,
        per_device_train_batch_size=wandb.config.batch_size // wandb.config.num_gpus,
        per_device_eval_batch_size=wandb.config.batch_size // wandb.config.num_gpus,
        learning_rate=wandb.config.lr,
        num_train_epochs=wandb.config.num_epochs,
        logging_steps=math.ceil(len(train_dataset) / (wandb.config.batch_size * wandb.config.num_logs_per_epoch)),
        save_strategy="no",  # TODO: Make this a parameter
        logging_first_step=True,
        evaluation_strategy="steps",
        # lr_scheduler_type='constant' if wandb.config.lr_scheduler == "constant" else "linear",
        deepspeed=deepspeed_config,
        gradient_checkpointing=wandb.config.gradient_checkpointing,
        bf16=wandb.config.bf16,
        fp16=False,  # TODO: Do I really need to set this?
        auto_find_batch_size=False,
        predict_with_generate=False, # NOTE: when False, the model will use teacher-forcing during evaluation (argmax one-shot prediction at all time steps at once)
        generation_max_length=192,  # TODO Should probably be a parameter
        include_inputs_for_metrics=True,
        eval_accumulation_steps=wandb.config.eval_accumulation_steps_config,
        dataloader_num_workers=wandb.config.num_gpus*4  # TODO: Make this a parameter
    )

    def custom_collator(inputs, model=model):
        # We want the labels to have -100 in the padding positions, so that they are ignored in the loss computation.
        # We also want padding to be done base don the longest inputs within the batch.

        labels = [i["labels"] for i in inputs]
        for i in inputs:
            del i["labels"]

        # Have to delete labels from inputs because DataCollatorsWith padding will try to turn them directory to tensors, and error out

        collator_with_padding = DataCollatorWithPadding(tokenizer, padding='longest', return_tensors='pt')
        collated_inputs = collator_with_padding(inputs)

        labels_max_length = max([len(x) for x in labels])
        labels = [x + [-100] * (labels_max_length - len(x)) for x in labels]

        collated_inputs["labels"] = torch.tensor(labels)  # TODO: Why do I not need to send this to a device?

        return collated_inputs

    log("Creating trainer", verbose)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset, # type: ignore
        eval_dataset=eval_dataset, # type: ignore
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=custom_collator
    )

    log("Training", verbose)
    trainer.train()
    if save_model_dir:
        trainer.save_state()
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=save_model_dir)

    log("Finished", verbose)
    wandb.finish()
