#!/bin/bash
#PBS -N mvi-sp
#PBS -q all
#PBS -l select=1:ncpus=4:ngpus=3:mem=16gb
#PBS -l walltime=500:00:00

# load software modules if needed
# see /etc/modulefiles for a list of available modules
module add /home/opt/lich/modulefiles/lich/cuda-11.7 # load CUDA for GPU calculations

#activate environment (conda)
ENVS_ROOT="/home/$USER/miniconda3/envs"
eval "$(command /home/$USER/miniconda3/bin/conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate "$ENVS_ROOT/mvi"

# check if CUDA is available
python -c "import torch; 
if torch.cuda.is_available(): 
    exit()" || { echo >&2 'CUDA UNAVAILABLE'; exit 6; }

# Run the job
python job.py || { echo >&2 "Error during job execution!"; exit 3; }