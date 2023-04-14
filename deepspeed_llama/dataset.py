from __future__ import annotations
from datasets.dataset_dict import DatasetDict
from datasets.arrow_dataset import Dataset
from datasets.load import load_dataset
import os
import wandb
import copy


def get_hugface_dataset(path: str, tokenizer, eval: bool = False) -> Dataset:

    dataset = dataset = load_dataset(
        'json', data_files=path,
        cache_dir="./cache",
    )
    tokenized_dataset = tokenize_dataset(dataset, tokenizer, eval=eval)

    assert isinstance(tokenized_dataset, Dataset)

    return tokenized_dataset


def preprocess_function_dec(examples, tokenizer):
    inputs = [doc + ex for doc, ex in zip(examples["prompt"], examples["completion"])]

    # Need to leave padding='max_length' otherwise there's an error creating tensor
    model_inputs = tokenizer(inputs)
    assert "attention_mask" in model_inputs
    model_inputs["labels"] = copy.deepcopy(model_inputs["input_ids"])

    if wandb.config.ignore_loss_on_prompt_tokens:
        prompts = [tokenizer.encode(doc) for doc in examples["prompt"]]
        prompt_lengths = [len(prompt) for prompt in prompts]
        for j, label in enumerate(model_inputs["labels"]):
            for i in range(0, prompt_lengths[j]):
                label[i] = -100

    return model_inputs


def max_pad_evaluate(examples, tokenizer, max_pad_length, keys_to_pad=["input_ids", "attention_mask", "labels"]):
    # Due to the way that tensors are concatenated during evaluation, we need to pad the inputs to the max length of the batch

    for key in keys_to_pad:
        examples_key_batch = [e for e in examples[key]]
        padding_value = None
        if key == "labels":
            padding_value = -100
        elif key == "attention_mask":
            padding_value = 0
        else:
            padding_value = tokenizer.pad_token_id
        examples_key_batch_padded = [e + [padding_value]*(max_pad_length-len(e)) for e in examples_key_batch]
        examples[key] = examples_key_batch_padded

    return examples


def tokenize_dataset(dataset, tokenizer, num_proc: int = 16, eval: bool = False) -> Dataset:

    def preprocess_function(examples): return preprocess_function_dec(examples, tokenizer=tokenizer)
    def max_pad_function_curried(max_length): return (
        lambda examples: max_pad_evaluate(examples, tokenizer, max_length))

    preprocessed_dataset = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )

    max_length_labels = max([len(x) for x in preprocessed_dataset["labels"]])
    max_pad_function = max_pad_function_curried(max_length_labels)

    if eval:
        preprocessed_dataset = preprocessed_dataset.map(max_pad_function, batched=True, num_proc=num_proc,
                                        load_from_cache_file=False, desc="Padding validation dataset")

    return preprocessed_dataset
