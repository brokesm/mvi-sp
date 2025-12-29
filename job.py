import sys
import optuna.trial
import optuna
import getopt, sys
import os
import importlib
import pandas as pd 
import json
import traceback
import numpy as np

from qsprpred.data import QSPRDataset
from qsprpred.models import QSPRModel
from qsprpred.data.descriptors.fingerprints import MorganFP
from qsprpred.data.descriptors.sets import SmilesDesc
from qsprpred.models import OptunaOptimization, TestSetAssessor, CrossValAssessor, SklearnModel
from qsprpred.data.sampling.splits import DataSplit
from qsprpred.models import EarlyStoppingMode
from sklearn.ensemble import GradientBoostingClassifier
from qsprpred.extra.gpu.models.chemprop import ChempropModel
from qsprpred.models.assessment.methods import ModelAssessor
from qsprpred.logs import logger
from qsprpred.models.monitors import BaseMonitor, HyperparameterOptimizationMonitor
from qsprpred.models.hyperparam_optimization import HyperparameterOptimization
from datetime import datetime

from typing import Callable, Iterable, List, Tuple, Literal


modname = 'qsprpred.extra.gpu.models.gdnn'
if modname in sys.modules:
    del sys.modules[modname]

import qsprpred.extra.gpu.models.gdnn as gdnn_module
from qsprpred.extra.gpu.models.gdnn import GGNN
importlib.reload(gdnn_module)


BASE_DIR = "/home/brokesm/scratch/job"
LOG_DIR = os.path.join(BASE_DIR, "output/logs")

#Define models and search spaces here
model_ggnn = gdnn_module.DNNModel(
    base_dir=os.path.join(BASE_DIR,'output/models/GGNN'),
    name='GGNNModel',
    parameters={'in_feats': 74},
    tol=0.01,
    random_state=42,
    patience=10
)

search_space_ggnn = {
    "n_hidden_layers": ["int", 1, 6],
    "dropout_rate": ["float", 0.05, 0.5],
    "steps": ["int", 1, 5],
    "batch_size": ["categorical", [32,64,128,256]],
    'out_feats': ["int",74,256],
    'optim':["categorical", ["adam","adamw","rmsprop","sgd"]],
    "optim_lr":["float", 1e-6, 1e-2],
    "activation_in":["categorical", ["relu","selu","tanh","leaky_relu"]],
    "n_epochs":["int",1,2],
    "optim_weight_decay":["float",0.0,1e-2],
    "optim_momentum":["float", 0.1, 0.9],
    "scheduler":["categorical", ["step","exp","reduce_on_plateau","cosine"]],
    "scheduler_gamma":["float",0.2,0.9],
    "scheduler_step_size":["int",1,2]
}


model_chemprop = ChempropModel(
    base_dir=os.path.join(BASE_DIR,'output/models/Chemprop'),
    name='ChempropModel',
    parameters={
        "loss_function":'binary_cross_entropy',
        "metric":"f1"
        },
    quiet_logger=False,
    random_state=42,
)

search_space_chemprop = {
    "epochs": ["int", 1,2],
    "batch_size": ["categorical", [32,64,128,256]],
    "hidden_size": ["int",100,500],
    "depth":["int",1,5],
    "dropout":["float",0.05,0.5],
    "activation":["categorical", ['ReLU', 'LeakyReLU', 'tanh', 'SELU']],
    "ffn_num_layers":["int",1,10],
    "init_lr":["float",1e-6,1e-2],
    "evidential_regularization":["float",0.05,0.5],
    "class_balance":["categorical", [True,False]]
}

model_xgb = SklearnModel(
    name="XGBModel",
    alg=GradientBoostingClassifier,
    base_dir=os.path.join(BASE_DIR, "output/models/XGB"),
    parameters={
        "random_state":42
    }
)

search_space_xgb = {
    "max_depth": ["int", 2, 10],
    "n_estimators": ["int", 5,500],
    "loss":["categorical", ["log_loss","exponential"]],
    "learning_rate":["float",1e-4,1e-1],
    "subsample":["float",0,1]
}



def write_log(target, split, path: str, err: str | BaseException) -> None:
    """
    Append error message / traceback to a per-target+split log file.
    Uses 'None' if target/split aren't parsed yet.
    """
    os.makedirs(path, exist_ok=True)
    safe_target = str(target) if target is not None else "None"
    safe_split = str(split) if split is not None else "None"
    name = f"log_{safe_target}_{safe_split}.txt"
    message = err if isinstance(err, str) else "".join(
        traceback.format_exception(type(err), err, err.__traceback__)
    )
    try:
        with open(os.path.join(path, name), mode="a") as f:
            f.write(message + "\n")
    except Exception as e:  # last resort: don't let logging crash the script
        # If even logging fails, print to stderr.
        sys.stderr.write(f"[LOGGING ERROR] {e}\nOriginal error:\n{message}\n")


#create a folder structure
def ensure_output_dirs() -> None:
    os.makedirs(os.path.join(BASE_DIR,"output/models"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR,"output/benchmarking/data"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR,"output/optimization/data"), exist_ok=True)


# define a customsplit class
# inherits from datasplit
# input - QSPRDataset ids
# output - (train,test) splits

class CustomSplit(DataSplit):

    def __init__(self, test_ids: list[list[str]]):
        super().__init__()
        self.test_ids = test_ids

    def split(
        self,
        X: np.ndarray | pd.DataFrame, 
        y: np.ndarray | pd.DataFrame | pd.Series
    ) -> Iterable[tuple[list[int], list[int]]]:
        """
        Uses only the specified IDs from the data set as test set
        """
        splits = []
        for test_ids in self.test_ids:
            test = np.where(X.index.isin(test_ids))[0]
            train = np.where(~X.index.isin(test_ids))[0]
            splits.append((train, test))
        return splits

# obtained from source code, modified so that it returns a search space to be accepted by next optuna iteration    
class OptunaOptimization(HyperparameterOptimization):
    """Class for hyperparameter optimization of QSPRModels using Optuna.

    Attributes:
        nTrials (int):
            number of trials for bayes optimization
        nJobs (int):
            number of jobs to run in parallel. At the moment only n_jobs=1 is supported.
        bestScore (float):
            best score found during optimization
        bestParams (dict):
            best parameters found during optimization

    Example of OptunaOptimization for scikit-learn's MLPClassifier:
        >>> model = SklearnModel(base_dir=".",
        >>>                     alg = MLPClassifier(), alg_name="MLP")
        >>> search_space = {
        >>>    "learning_rate_init": ["float", 1e-5, 1e-3,],
        >>>    "power_t" : ["discrete_uniform", 0.2, 0.8, 0.1],
        >>>    "momentum": ["float", 0.0, 1.0],
        >>> }
        >>> optimizer = OptunaOptimization(
        >>>     scoring="average_precision",
        >>>     param_grid=search_space,
        >>>     n_trials=10
        >>> )
        >>> best_params = optimizer.optimize(model, dataset) # dataset is a QSPRDataset

    Available suggestion types:
        ["categorical", "discrete_uniform", "float", "int", "loguniform", "uniform"]
    """

    def __init__(
        self,
        param_grid: dict,
        model_assessor: ModelAssessor,
        score_aggregation: Callable[[Iterable], float] = np.mean,
        monitor: HyperparameterOptimizationMonitor | None = None,
        n_trials: int = 100,
        n_jobs: int = 1,
    ):
        """Initialize the class for hyperparameter optimization
        of QSPRModels using Optuna.

        Args:
            param_grid (dict):
                search space for bayesian optimization, keys are the parameter names,
                values are lists with first element the type of the parameter and the
                following elements the parameter bounds or values.
            model_assessor (ModelAssessor):
                assessment method to use for the optimization
                (default: CrossValAssessor)
            score_aggregation (Callable):
                function to aggregate the scores of different folds if the assessment
                method returns multiple predictions
            monitor (HyperparameterOptimizationMonitor):
                monitor for the optimization, if None, a BaseMonitor is used
            n_trials (int):
                number of trials for bayes optimization
            n_jobs (int):
                number of jobs to run in parallel.
                At the moment only n_jobs=1 is supported.
        """
        super().__init__(param_grid, model_assessor, score_aggregation, monitor)
        if monitor is None:
            self.monitor = BaseMonitor()
        search_space_types = [
            "categorical",
            "discrete_uniform",
            "float",
            "int",
            "loguniform",
            "uniform",
        ]
        if not all(v[0] in search_space_types for v in param_grid.values()):
            logger.error(
                f"Search space {param_grid} is missing or has invalid search type(s), "
                "see OptunaOptimization docstring for example."
            )
            raise ValueError(
                "Search space for optuna optimization is missing or "
                "has invalid search type(s)."
            )

        self.nTrials = n_trials
        self.nJobs = n_jobs
        if self.nJobs > 1:
            logger.warning(
                "At the moment n_jobs>1 not available for bayes optimization, "
                "n_jobs set to 1."
            )
            self.nJobs = 1
        self.bestScore = -np.inf
        self.bestParams = None
        self.config.update(
            {
                "n_trials": n_trials,
                "n_jobs": n_jobs,
            }
        )

    def optimize(
        self,
        model: QSPRModel,
        ds: QSPRDataset,
        save_params: bool = True,
        refit_optimal: bool = False,
        **kwargs,
    ) -> dict:
        """Bayesian optimization of hyperparameters using optuna.

        Args:
            model (QSPRModel): the model to optimize
            ds (QSPRDataset): dataset to use for the optimization
            save_params (bool):
                whether to set and save the best parameters to the model
                after optimization
            refit_optimal (bool):
                Whether to refit the model with the optimal parameters on the
                entire training set after optimization. This implies 'save_params=True'.
            **kwargs: additional arguments for the assessment method

        Returns:
            dict: best parameters found during optimization
        """

        self.monitor.onOptimizationStart(
            model, ds, self.config, self.__class__.__name__
        )

        logger.info(
            "Bayesian optimization can take a while "
            "for some hyperparameter combinations"
        )
        # create optuna study
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=model.randomState),
        )
        logger.info(
            "Bayesian optimization started: %s"
            % datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        study.optimize(
            lambda t: self.objective(t, model, ds), self.nTrials, n_jobs=self.nJobs
        )
        logger.info(
            "Bayesian optimization ended: %s"
            % datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        # save the best study
        trial = study.best_trial
        trials = study.get_trials()
        top_n = sorted(trials, key=lambda x:x.values)[:10]
        
        aggr_values = {k:[] for k in top_n[0].params.keys()}
        for tr in top_n:
            params = tr.params
            for k,v in params.items():
                aggr_values[k].append(v)

        next_iter = {
            k:[self.config["param_grid"][k][0], np.min(v), np.max(v)] if self.config["param_grid"][k][0] != "categorical" \
            else ["categorical", np.unique(v).tolist()] \
            for k,v in aggr_values.items()
        }

        # log the best study
        logger.info("Bayesian optimization best params: %s" % trial.params)
        # save the best score and parameters, return the best parameters
        self.bestScore = trial.value
        self.bestParams = trial.params

        self.monitor.onOptimizationEnd(self.bestScore, self.bestParams)
        # save the best parameters to the model if requested
        self.saveResults(model, ds, save_params, refit_optimal)
        # return self.bestParams
        return next_iter
    
    def objective(
        self, trial: optuna.trial.Trial, model: QSPRModel, ds: QSPRDataset, **kwargs
    ) -> float:
        """Objective for bayesian optimization.

        Arguments:
            trial (optuna.trial.Trial): trial object for the optimization
            model (QSPRModel): the model to optimize
            ds (QSPRDataset): dataset to use for the optimization
            **kwargs: additional arguments for the assessment method

        Returns:
            float: score of the model with the current parameters
        """
        bayesian_params = {}
        # get the suggested parameters for the current trial
        for key, value in self.paramGrid.items():
            if value[0] == "categorical":
                bayesian_params[key] = trial.suggest_categorical(key, value[1])
            elif value[0] == "discrete_uniform":
                bayesian_params[key] = trial.suggest_float(
                    key, value[1], value[2], step=value[3]
                )
            elif value[0] == "float":
                bayesian_params[key] = trial.suggest_float(key, value[1], value[2])
            elif value[0] == "int":
                bayesian_params[key] = trial.suggest_int(key, value[1], value[2])
            elif value[0] == "loguniform":
                bayesian_params[key] = trial.suggest_float(
                    key, value[1], value[2], log=True
                )
            elif value[0] == "uniform":
                bayesian_params[key] = trial.suggest_float(key, value[1], value[2])
        self.monitor.onIterationStart(bayesian_params)
        # assess the model with the current parameters and return the score
        scores = self.runAssessment(
            model,
            ds=ds,
            save=False,
            parameters=bayesian_params,
            monitor=self.monitor,
            **kwargs,
        )
        score = self.scoreAggregation(scores)
        logger.info(bayesian_params)
        logger.info(f"Score: {score}, std: {np.std(scores)}")
        self.monitor.onIterationEnd(score, list(scores))
        return score


def select_ids(dataset_name, keep_ids):
    return [f"{dataset_name}_{'0' * (4 - len(str(id)))}{id}" for id in keep_ids]


def data_loading(
    target:Literal["P00918","P03372","P04637","P08684","P14416","P22303","P42336","Q12809","Q16637","Q9Y468"], 
    purpose:Literal["ForOptimization","ForBenchmarking"],
    model:QSPRModel | None = None,
    save = True
) -> Tuple[QSPRDataset, List, List, List]:

    dataset_name = f"{purpose}_{target}"
    store_dir = os.path.join(BASE_DIR,f"output/{purpose[3:].lower()}/data")

    dataset = QSPRDataset.fromTableFile(
        filename=os.path.join(BASE_DIR,f"papyrus_datasets/{target}.csv"),
        sep=",",
        store_dir=store_dir,
        name=dataset_name,
        target_props=[{"name": "Y", "task": "SINGLECLASS", "th":"precomputed"}],
        random_state=42
    )

    if model is not None:
        if model.supportsEarlyStopping:
            # In case of GNNs (both support early stopping) add SmilesDesc as descriptors
            dataset.addDescriptors([SmilesDesc()])
        else:
            # In case of XGB (doesn't support early stopping) add MorganFP with default parameters as descriptors
            dataset.addDescriptors([MorganFP()])

    if save:
        dataset.save()
    
    return dataset


def hyperparameter_optimization(
    model:QSPRModel, 
    dataset:QSPRDataset, 
    search_space:dict, 
    scoring:str, 
    val_ids:List,
    test_ids:List,
    n_trials:List = [80,80,160]
):
    # opravit, prvy split CVA je rozdelenie povodneho datasetu na train/test
    # nasledny dataset split bude na train/val (val set je iba na early stopping v ramci CVA)
    # Uz asi hotovo

    dataset.prepareDataset(
        split=CustomSplit([val_ids])
    )

    for n in n_trials:
        gridsearcher = OptunaOptimization(
            n_trials=n,
            param_grid=search_space,
            model_assessor=CrossValAssessor(scoring=scoring, split=CustomSplit([test_ids])),
        )

        search_space = gridsearcher.optimize(model, dataset)



def set_loader(
    target:Literal["P00918","P03372","P04637","P08684","P14416","P22303","P42336","Q12809","Q16637","Q9Y468"], 
    split_type:Literal["random", "cluster", "aggregate_cluster"],
    seed,
    purpose:Literal["ForBenchmarking","ForOptimization"]
):
    seed = str(seed)

    with open(os.path.join(BASE_DIR, f"papyrus_datasets/{split_type}_split.json")) as file:
        json_file = file.read()
    split = json.loads(json_file)
    
    train_ids = split[split_type][target][seed]["train"]
    val_ids = split[split_type][target][seed]["valid"]
    test_ids = split[split_type][target][seed]["test"]

    train_ids = select_ids(f"{purpose}_{target}",list(train_ids))
    val_ids = select_ids(f"{purpose}_{target}",list(val_ids))
    test_ids = select_ids(f"{purpose}_{target}",list(test_ids))

    return train_ids, val_ids, test_ids


def run_optimization(
    target:Literal["P00918","P03372","P04637","P08684","P14416","P22303","P42336","Q12809","Q16637","Q9Y468"],
    split:Literal["random","cluster","aggregate_cluster"],
    model:QSPRModel,
    search_space:dict,
    seed = 0
):
    dataset = data_loading(target,model=model, purpose="ForOptimization")
    train_ids, val_ids, test_ids = set_loader(target,split,seed,purpose="ForOptimization")
    
    hyperparameter_optimization(model=model, dataset=dataset, search_space=search_space, scoring="f1", val_ids=val_ids, test_ids=test_ids)
    

def get_model_params(
        target:Literal["P00918","P03372","P04637","P08684","P14416","P22303","P42336","Q12809","Q16637","Q9Y468"], 
        split_type:Literal["random","cluster","aggregate_cluster"],
        model:Literal["XGB","GGNN","Chemprop"]
    ):
    with open(os.path.join(BASE_DIR, f"output/models/{model}/{model}Model_{target}_{split_type}/{model}Model_{target}_{split_type}_meta.json")) as f:
        params = f.read()

    params = json.loads(params)
    return params["py/state"]["parameters"]


def prepare_for_benchmarking(dataset:QSPRDataset,descriptors, chemprop=False):
    dataset.addDescriptors([descriptors])
    if chemprop:
        # binary cross entropy loss cannot deal with target variable being of type int
        dataset.transformProperties(["Y","Y_original"],transformer=np.float32)


def benchmark(
    target:Literal["P00918","P03372","P04637","P08684","P14416","P22303","P42336","Q12809","Q16637","Q9Y468"],
    split_type:Literal["random","cluster","aggregate_cluster"]
):
    # save the dataset corresponding to a given target
    data_loading(target,purpose="ForBenchmarking")

    for metric in ["matthews_corrcoef","f1","recall","precision","roc_auc"]:

        for seed in range(1,21):
            # get the ids for training, validation and test sets for a given combination of target + split + seed
            _, val_ids, test_ids = set_loader(target,split_type,seed=seed, purpose="ForBenchmarking")
            dataset_path = os.path.join(BASE_DIR, f"output/benchmarking/data/ForBenchmarking_{target}/ForBenchmarking_{target}_meta.json")
            
            dataset_xgb = QSPRDataset.fromFile(dataset_path)
            dataset_ggnn = QSPRDataset.fromFile(dataset_path)
            dataset_chemprop = QSPRDataset.fromFile(dataset_path)

            prepare_for_benchmarking(dataset_xgb,MorganFP())
            prepare_for_benchmarking(dataset_ggnn,SmilesDesc())
            prepare_for_benchmarking(dataset_chemprop,SmilesDesc(), chemprop=True)

            model_xgb.parameters = get_model_params(target,split_type,"XGB")
            model_ggnn.parameters = get_model_params(target,split_type,"GGNN")
            model_chemprop.parameters = get_model_params(target,split_type,"Chemprop")

            proba = True
            if metric == "matthews_corrcoef":
                proba = False

            dataset_xgb.prepareDataset(split = CustomSplit([test_ids]))
            xgb_score = TestSetAssessor(scoring=metric, use_proba=proba)(model_xgb, dataset_xgb)

            # Tu mozno pouzit iba CVA, kde na val mnozine najdem best epoch pomocou early stopping
            # Na test mnozine v ramci toho isteho CVA vypocitam skore
            # Uz opravene
            dataset_ggnn.prepareDataset(split=CustomSplit([val_ids]))
            ggnn_score = CrossValAssessor(
                scoring=metric,
                use_proba=proba,
                mode=EarlyStoppingMode.RECORDING,
                split=CustomSplit([test_ids]))(model_ggnn, dataset_ggnn)

            dataset_chemprop.prepareDataset(split=CustomSplit([val_ids]))
            chemprop_score = CrossValAssessor(
                scoring=metric,
                use_proba=proba,
                mode=EarlyStoppingMode.RECORDING,
                split=CustomSplit([test_ids]))(model_chemprop, dataset_chemprop)
        
            with open(os.path.join(BASE_DIR, f"output/benchmarking/{target}/{metric}/results.txt"), mode="a") as f:
                f.write(f"XGB\t{split_type}\t{xgb_score.item()}\n")
                f.write(f"GGNN\t{split_type}\t{ggnn_score.item()}\n") 
                f.write(f"Chemprop\t{split_type}\t{chemprop_score.item()}\n")
                f.close()


def main(target,split) -> None:
    ensure_output_dirs()

    # Hyperparameter optimization for all three models
    models = [model_xgb, model_ggnn, model_chemprop]
    search_spaces = [search_space_xgb, search_space_ggnn, search_space_chemprop]
    for model, search_space in zip(models, search_spaces):
        # preserve existing naming logic
        model.name += f"_{target}_{split}"
        run_optimization(
            target=target,
            split=split,
            model=model,
            search_space=search_space
        )
        model.name = model.name.split("_")[0] #restore base name
    # Benchmarking after optimization
    benchmark(target, split)


if __name__ == "__main__":
    targets = ["P00918","P03372","P04637","P08684","P14416","P22303","P42336","Q12809","Q16637","Q9Y468"]
    splits = ["random","cluster","aggregate_cluster"]
    for target in targets:

        for metric in ["matthews_corrcoef","f1","recall","precision","roc_auc"]:
            os.makedirs(os.path.join(BASE_DIR, f"output/benchmarking/{target}/{metric}"), exist_ok=True)
            with open(os.path.join(BASE_DIR, f"output/benchmarking/{target}/{metric}/results.txt"), mode='w') as file:
                file.write(f"model\tsplit_type\tscore\n")
                file.close()

        for split in splits:
            try:
                main(target,split)
            except Exception as e:
                # catch any unhandled exceptions and log them
                write_log(target, split, LOG_DIR, e)
                sys.exit(1)




















