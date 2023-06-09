#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time 0-16:00:00

# NOTE: set the environment in your shell before running this script
date;hostname;id;pwd
export WANDB_API_KEY=$3
source /opt/rh/devtoolset-10/enable

train_script=~/deepspeed_llama/run/train.py

random_number=$(( ($RANDOM  % 32000 )  + 1 ))
deepspeed --master_port $((random_number + 1024)) $train_script --project $1 --file $2 --job_id $SLURM_ARRAY_JOB_ID --task_id $SLURM_ARRAY_TASK_ID 

if grep -q "The server socket has failed to listen on any local network address" ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log; then
    echo "Restarting job with different tcp port"
    sbatch --array=$SLURM_ARRAY_TASK_ID $0 $1 $2 $3
fi