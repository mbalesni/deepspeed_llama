"""
runs `agent_deepspeed.sh` which runs `train.py` according to a .yaml sweep config file
"""

import subprocess
import yaml
from itertools import product
import json
import argparse
import os
from deepspeed_llama.common import attach_debugger, project_dir
from datetime import datetime


def sweep(args):

    config_yaml = args.config_file

    with open(config_yaml) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    config_dir = os.path.dirname(config_yaml)
    param_combinations = product(*config['hyperparameters'].values())
    sweeps = [dict(zip(config['hyperparameters'].keys(), values)) for values in param_combinations]

    for sweep in sweeps:
        sweep.update(config['fixed_parameters'])
        sweep["experiment_name"] = args.experiment_name

    # Check that all data files exist, this has errored me out enough times that I think it's worth an assert
    for sweep in sweeps:
        train_file = os.path.join(project_dir, sweep["data_dir"], sweep["train_path"])
        valid_file = os.path.join(project_dir, sweep["data_dir"], sweep["validation_path"])
        data_files = [train_file, valid_file]
        assert any([os.path.isfile(data_file) for data_file in data_files]
                   ), f"Data file {data_files[0]} or {data_files[1]} does not exist"

    sweep_file_dir = os.path.join(config_dir, 'sweep_configs')
    if not os.path.exists(sweep_file_dir):
        os.makedirs(sweep_file_dir)
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    sweep_file = os.path.join(sweep_file_dir, f'{current_time}.json')

    if os.path.isfile(sweep_file):
        os.remove(sweep_file)

    i = 0
    while os.path.isfile(sweep_file):
        i += 1
        sweep_file = os.path.join(sweep_file_dir, f'{current_time}_{i}.json')

    json.dump(sweeps, open(sweep_file, 'w'))

    partition = 'compute' if not args.run_interactive else 'interactive'

    slurm_script = os.path.join(os.path.dirname(__file__), 'agent_deepspeed.sh')

    log_dir = os.path.join(os.path.dirname(os.path.dirname(sweep_file)), 'logs')
    os.makedirs(log_dir, exist_ok=True)

    if args.node_list is None:
        command = [
            'sbatch',
            '--nodes=1',
            f'--gpus={config["fixed_parameters"]["num_gpus"]}',
            '--array',
            f'0-{len(sweeps) - 1}',
            f'--cpus-per-gpu',
            f'{config["fixed_parameters"]["cpus_per_gpu"]}',
            f'--mem={config["fixed_parameters"]["ram_limit_gb"]}G',
            '--partition',
            partition,
            '--output',
            os.path.join(log_dir, '%A_%a.log'),
            slurm_script,
            config['project_name'],
            sweep_file,
            os.environ['WANDB_API_KEY']]

        subprocess.run(command)
    else:
        # NOTE: only necessary if you launch jobs that use less than one full node, and you want to spread them across pre-selected partially occupied nodes that have enough free RAM
        # an alternative where there're enough free nodes is to use the `--exclusive` flag with the above if statement
        job_num = 0
        while job_num < len(sweeps):
            command = ['sbatch',
                        '--nodes=1', # TODO: this potentially shouldn't be here, but should be in the initial `if` statement
                        f'--gpus={config["fixed_parameters"]["num_gpus"]}',
                        '--array',
                        f'{job_num}-{job_num}',
                        '--cpus-per-gpu',
                        f'{config["fixed_parameters"]["cpus_per_gpu"]}',
                        f'--mem={config["fixed_parameters"]["ram_limit_gb"]}G',
                        f'-w',
                        f'compute-permanent-node-{args.node_list[job_num % len(args.node_list)]}',
                        '--partition',
                        partition,
                        '--output',
                        os.path.join(log_dir, '%A_%a.log'),
                        slurm_script,
                        config['project_name'],
                        sweep_file,
                        os.environ['WANDB_API_KEY']]
            print(command)
            job_num += 1

            subprocess.run(command)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_port", type=int, default=5678)
    parser.add_argument("--run_interactive", action="store_true", default=False)
    parser.add_argument("--node_list", type=str, required=False, default=None)

    args = parser.parse_args()

    if args.debug:
        attach_debugger(port=args.debug_port)

    args.node_list = args.node_list.split(",") if args.node_list is not None else None

    sweep(args)
