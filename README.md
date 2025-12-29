## Evaluation and enhancement of an existing GNN model for drug discovery

#### Project definition: 

The primary goal of this semestral work is to benchmark an existing GGNN[3] (gated graph 
neural network) and a Chemprop[7] model (message-passing neural network) against a base 
model – an SKLearn implementation of XGBoost for a total of 10 target datasets extracted 
from Papyrus database8 v.05.6. Each dataset is divided into training, validation and test set 
using one of the following algorithms: 
- random split algorithm 
- cluster split algorithm 
- aggregate cluster split algorithm. 

Each split for each dataset was run 21 times for random seeds 0 (used for hyperparameter 
optimization) to 20 (used for test set evaluation). Each dataset contains SMILES of 
compounds tested against the target and a target variable for binary classification derived 
from median – 6.5 pCHEMBL value – a threshold above which the compound-target 
interaction is classified as active, otherwise inactive[2]. The benchmark will compare the 
algorithms’ ability to make reasonable predictions for compounds very different from training 
set. The secondary goal is to finalize and test the GGNN implementation[5] to be compatible with 
the open-source QSPRpred API[4][1]. 

#### Large files:

The abovementioned splits are stored as json files using git LFS. To clone or pull the repository with 
the actual large files run:

`git lfs clone <repo>`

#### Run the project:

The benchmarking workflow has been run on UCT (VŠCHT) HPC cluster after home directory creation (/home/brokesm) 
following the instructions found within private lich-compute (https://github.com/lich-uct/lich-compute) repo.

Conda has been installed and a new environment created.
Three packages had to be installed manually using the following commands:

`conda install -c dglteam/label/th24_cu121 dgl`

`pip install qsprpred[chemprop]`

`pip install git+https://github.com/brokesm/QSPRpred.git@add-graph-dl`

Afterwards, the remaining dependencies can be installed from `gnn_project_env.yml`.

After conda installation and initalization, `runit.sh` and `job.py` were copied to the head node and run using `qsub`.