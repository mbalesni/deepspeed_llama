import debugpy
import os
from typing import List
import torch
import psutil
from typing import List, Dict
import string
import pathlib

project_dir = pathlib.Path(__file__).parent.parent


def attach_debugger(port=5678):
    debugpy.listen(port)
    print('Waiting for debugger!')

    debugpy.wait_for_client()
    print('Debugger attached!')


def memory_usage():
    main_process = psutil.Process(os.getpid())
    children_processes = main_process.children(recursive=True)

    cpu_percent = main_process.cpu_percent()
    mem_info = main_process.memory_info()
    ram_usage = mem_info.rss / (1024 ** 2)

    # Add memory usage of DataLoader worker processes
    for child_process in children_processes:
        ram_usage += child_process.memory_info().rss / (1024 ** 2)

    print("CPU Usage: {:.2f}%".format(cpu_percent))
    print("RAM Usage (including DataLoader workers): {:.2f} MB".format(ram_usage))

    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_mem_alloc = torch.cuda.memory_allocated(device) / (1024 ** 2)
        gpu_mem_cached = torch.cuda.memory_reserved(device) / (1024 ** 2)

        print("GPU Memory Allocated: {:.2f} MB".format(gpu_mem_alloc))
        print("GPU Memory Cached: {:.2f} MB".format(gpu_mem_cached))
    else:
        print("CUDA is not available")


def flatten(list_of_lists: List[List]):
    return [item for sublist in list_of_lists for item in sublist]


def apply_replacements(list: List, replacements: Dict) -> List:
    return [apply_replacements_to_str(string, replacements) for string in list]


def apply_replacements_to_str(string: str, replacements: Dict) -> str:
    for before, after in replacements.items():
        string = string.replace(before, after)
    return string

def normalize_answer(s):
    """Lower text and remove punctuation, and extra whitespace."""

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


def log_memory(args):
    if args.logging:
        memory_usage()


def log(string, args):
    if args.logging:
        print(string)

