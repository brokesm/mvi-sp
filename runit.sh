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

# not crucial, but convenient
# append a line to a file (i.e. "jobs_info.txt") in your $DATADIR containing the ID of the job and the hostname of the node it is running on
# this information helps you to debug the jobs if they fail and find the files in $SCRATCHDIR more easily (i.e. if you need to delete them after a failed attempt)
# echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR" >> $DATADIR/jobs_info.txt

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

# # run all combinations of below
# declare -a datasets=(P00918 P03372 P04637 P08684 P14416 P22303 P42336 Q9Y468 Q12809 Q16637)
# declare -a splits=(random cluster aggregate_cluster)

# # Calculate the index for each array item combination
# total_combinations=$((${#datasets[@]} * ${#splits[@]}))

# # Validate the job array index to ensure it is within the valid range (1 to 240)
# if [[ $PBS_ARRAY_INDEX -lt 1 || $PBS_ARRAY_INDEX -gt $total_combinations ]]; then
#   echo "Invalid PBS_ARRAY_INDEX: $PBS_ARRAY_INDEX"
#   exit 1
# fi

# # Map the PBS_ARRAY_INDEX to a specific combination
# index=$((PBS_ARRAY_INDEX - 1)) # Adjust index for zero-based array
# dataset_index=$((index % ${#datasets[@]}))   # Get dataset index
# index=$((index / ${#datasets[@]}))  # Divide
# split_index=$((index % ${#splits[@]}))  # Get  split index


# # Get the values for the current job
# dataset=${datasets[$dataset_index]}
# split=${splits[$split_index]}

# # Print which combination will be run
# echo "Running with dataset=$dataset, split=$split"

# Run our command with the appropriate arguments
python job.py || { echo >&2 "Error during job execution!"; exit 3; }

# copy everything important to $DATADIR in our home for persistent storage
RESULTSDIR=/home/$USER/results/
mkdir -p /home/$USER/results/
cp -r $SCRATCHDIR/output $RESULTSDIR || { echo >&2 "Result file(s) copying failed (with a code $?)!!! You can retrieve your files from `hostname -f`:`pwd`"; exit 4; } 
cp -r $SCRATCHDIR/output/training.log $RESULTSDIR || { echo >&2 "Result file(s) copying failed (with a code $?)!!! You can retrieve your files from `hostname -f`:`pwd`"; exit 5; } 

# clean the SCRATCH directory
cd /home/$USER
rm -rf /home/$USER/scratch/job