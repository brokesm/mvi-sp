#!/bin/bash
#PBS -N mvi-sp
#PBS -q all
#PBS -l select=1:ncpus=4:ngpus=3:mem=16gb
#PBS -l walltime=500:00:00

# define a DATADIR variable: directory where the input files are taken from and where output will be copied to
ENV_NAME="mvi"
DATADIR=/home/$USER/mvi-sp/

# use the local scratch directory on the worker to store data for the calculation
# this is important because your home directory is only a network mount on the worker which could cause loss of data and other performance issues
SCRATCHDIR=/home/$USER/scratch/job
mkdir -p /home/$USER/scratch/job

# copy files we will need for the calculation to our scratch
# if the copy operation fails, issue an error message and exit
cp $DATADIR/job.py $SCRATCHDIR || { echo >&2 "Error while copying input file(s)!"; exit 2; } 
cp -r $DATADIR/papyrus_datasets $SCRATCHDIR || { echo >&2 "Error while copying input file(s)!"; exit 2; } 

# move to the scratch directory (the job's working directory)
cd $SCRATCHDIR || { echo >&2 "Error while copying input file(s)! Cannot enter directory..."; exit 2; }

# load software modules if needed
# see /etc/modulefiles for a list of available modules
module add /home/opt/lich/modulefiles/lich/cuda-11.7 # load CUDA for GPU calculations

#activate environment (conda)
ENVS_ROOT="/home/$USER/miniconda3/envs" # where your conda/mamba environments live
eval "$(command /home/$USER/miniconda3/bin/conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate "$ENVS_ROOT/mvi"

# check if CUDA is available
python -c "import torch; 
if torch.cuda.is_available(): 
    exit()" || { echo >&2 'CUDA UNAVAILABLE'; exit 6; }

# Run the job
python job.py || { echo >&2 "Error during job execution!"; exit 3; }

# copy everything important to $DATADIR in our home for persistent storage
RESULTSDIR=/home/$USER/results/
mkdir -p /home/$USER/results/
cp -r $SCRATCHDIR/output $RESULTSDIR || { echo >&2 "Result file(s) copying failed (with a code $?)!!! You can retrieve your files from `hostname -f`:`pwd`"; exit 4; } 

# clean the SCRATCH directory
cd /home/$USER
rm -rf /home/$USER/scratch/job