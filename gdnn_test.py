# KOD PRO INSPIRACI - CO MUSI!!! BYT IMPLEMENTOVANO!!!


"""This module holds the base class for DNN models
as well as fully connected NN subclass.
"""

import inspect
from collections import defaultdict

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as f
from torch.utils.data import DataLoader, TensorDataset

from ....logs import logger
from ....models.monitors import BaseMonitor, FitMonitor

import os
from typing import Any, Type

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import ShuffleSplit
import torch.nn as nn
import torch.nn.functional as F

from qsprpred.tasks import ModelTasks
from .base_torch import QSPRModelPyTorchGPU, DEFAULT_TORCH_GPUS
from ....data.sampling.splits import DataSplit
from ....data.tables.qspr import QSPRDataset
#from ....extra.gpu.models.graph_neural_network import GGNN
#STFullyConnected, Base
from ....models.early_stopping import EarlyStoppingMode, early_stopping
from ....models.monitors import BaseMonitor, FitMonitor


import dgl
from dgl.nn import GatedGraphConv, GlobalAttentionPooling
from dgl.dataloading import GraphDataLoader
from dgllife.utils import mol_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
import math
from rdkit import Chem
from sklearn.metrics import matthews_corrcoef

class GGNN(nn.Module):
    """Single task DNN classification/regression model.

    It contains four fully connected layers between which are
    dropout layers for robustness.

    Attributes:
        n_dim (int): the No. of columns (features) for input tensor
        n_class (int): the No. of columns (classes) for output tensor.
        device (torch.cude): device to run the model on
        gpus (list): list of gpu ids to run the model on
        n_epochs (int): max number of epochs
        lr (float): neural net learning rate
        batch_size (int): batch size for training
        patience (int): early stopping patience
        tol (float): early stopping tolerance
        is_reg (bool): whether the model is for regression or classification
        neurons_h1 (int): No. of neurons in the first hidden layer
        neurons_hx (int): No. of neurons in the second hidden layer
        extra_layer (bool): whether to add an extra hidden layer
        dropout_frac (float): dropout fraction
        criterion (torch.nn.Module): the loss function
        dropout (torch.nn.Module): the dropout layer
        fc0 (torch.nn.Module): the first fully connected layer
        fc1 (torch.nn.Module): the second fully connected layer
        fc2 (torch.nn.Module): the third fully connected layer
        fc3 (torch.nn.Module): the fourth fully connected layer
        activation (torch.nn.Module): the activation function
    """

    def __init__(
        self,
        n_dim,
        n_class,
        device,
        gpus,
        activation_in,
        criterion,
        optim,
        n_epochs=100,
        lr=None,
        batch_size=256,
        patience=50,
        tol=0,
        is_reg=True,
        in_feats=256,
        out_feats=128,
        n_hidden_layers=2,
        dropout_rate=0.25,
        gamma=0.8,
        steps=3,
        etypes=1
    ):
        """Initialize the STFullyConnected model.

        Args:
            n_dim (int):
                the No. of columns (features) for input tensor
            n_class (int):
                the No. of columns (classes) for output tensor.
            device (torch.cude):
                device to run the model on
            gpus (list):
                list of gpu ids to run the model on
            n_epochs (int):
                max number of epochs
            lr (float):
                neural net learning rate
            batch_size (int):
                batch size
            patience (int):
                number of epochs to wait before early stop if no progress on
                validation set score, if patience = -1, always train to n_epochs
            tol (float):
                minimum absolute improvement of loss necessary to
                count as progress on best validation score
            is_reg (bool, optional):
                Regression model (True) or Classification model (False)
            neurons_h1 (int):
                number of neurons in first hidden layer
            neurons_hx (int):
                number of neurons in other hidden layers
            extra_layer (bool):
                add third hidden layer
            dropout_frac (float):
                dropout fraction
        """

        super().__init__()
        self.device = torch.device(device)
        self.gpus = gpus
        if len(self.gpus) > 1:
            logger.warning(
                f"At the moment multiple gpus is not possible: "
                f"running DNN on gpu: {gpus[0]}."
            )
        self.n_dim = n_dim
        self.is_reg = is_reg
        self.n_class = n_class if not self.is_reg else 1
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.n_hidden_layers = n_hidden_layers
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.steps = steps
        self.etypes = etypes
        self.patience = patience
        self.tol = tol
        self.lr = lr if not None else 1e-4
        self.dropout = None
        self.activation_out = None
        self.activation_in = activation_in
        self.criterion = criterion
        self.optim = optim
        self.layers = nn.ModuleList()
        self.initModel()
        

    def initModel(self):
        """Define the layers of the model."""
        self.dropout = nn.Dropout(self.dropout_rate)
        for i in range(self.n_hidden_layers):
            if i == 0:
                set_in_feats = self.in_feats
            else:
                set_in_feats = self.out_feats

            layer = GatedGraphConv(
                in_feats=set_in_feats,
                out_feats=self.out_feats,
                n_steps=self.steps,
                n_etypes=self.etypes
            )
            self.layers.append(layer)  
        pooling_gate_nn = nn.Linear(self.out_feats, 1)
        self.pooling = GlobalAttentionPooling(pooling_gate_nn) 
        self.output_layer = nn.Linear(self.out_feats, self.n_class)

        if self.is_reg:
            # loss function for regression
            self.criterion = nn.MSELoss()
        elif self.n_class == 1:
            # loss and activation function of output layer for binary classification
            self.criterion = nn.BCELoss()
            self.activation = nn.Sigmoid()
        else:
            # loss and activation function of output layer for multiple classification
            self.criterion = nn.CrossEntropyLoss()
            self.activation = nn.Softmax(dim=1)

    @classmethod
    def _get_param_names(cls) -> list:
        """Get the class parameter names.

        Function copied from sklearn.base_estimator!

        Returns:
            parameter names (list): list of the class parameter names.
        """
        init_signature = inspect.signature(cls.__init__)
        parameters = [
            p
            for p in init_signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]
        return sorted([p.name for p in parameters])

    def get_params(self, deep=True) -> dict:
        """Get parameters for this estimator.

        Function copied from sklearn.base_estimator!

        Args:
            deep (bool): If True, will return the parameters for this estimator

        Returns:
            params (dict): Parameter names mapped to their values.
        """
        out = {}
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, "get_params"):
                deep_items = value.get_params().items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        """Set the parameters of this estimator.

        Function copied from sklearn.base_estimator!
        The method works on simple estimators as well as on nested objects
        (such as :class:`~sklearn.pipeline.Pipeline`). The latter have
        parameters of the form ``<component>__<parameter>`` so that it's
        possible to update each component of a nested object.

        Args:
            **params : dict Estimator parameters.

        Returns:
            self : estimator instance
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)
        # grouped by prefix
        nested_params = defaultdict(dict)
        for key, value in params.items():
            key, delim, sub_key = key.partition("__")
            if key not in valid_params:
                local_valid_params = self._get_param_names()
                raise ValueError(
                    f"Invalid parameter {key!r} for estimator {self}. "
                    f"Valid parameters are: {local_valid_params!r}."
                )
            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value
        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)
        self.initModel()
        return self

    def forward(self, graph, features, is_train=False) -> torch.Tensor:
        """Invoke the class directly as a function.

        Args:
            X (FloatTensor):
                m X n FloatTensor, m is the No. of samples, n is
                the No. of features.
            is_train (bool, optional):
                is it invoked during training process (True) or
                just for prediction (False)
        Returns:
            y (FloatTensor): m X n FloatTensor, m is the No. of samples,
                n is the No. of classes
        """

        features = features.to(graph.device)
        y = self.activation(self.layers[0].forward(graph, features))
        if is_train:    
            y = self.dropout(y)
        for i in range(self.n_hidden_layers):
            if i == 0:
                continue
            else:
                y = self.activation(self.layers[i].forward(graph, y))  
                if i < self.n_hidden_layers - 1 and is_train:
                    y = self.dropout(y)
        y = self.pooling(graph, y)
        if self.is_reg:
            return self.output_layer(y)
        return self.activation_out(self.output_layer(y))

    def collate(self, samples): 
        if len(samples[0]) == 2:
            graphs = [s[0] for s in samples]
            labels = [s[1].item() for s in samples]
            batched_graph = dgl.batch(graphs)
            if self.is_reg:
                labels = torch.tensor(labels,dtype=torch.float32)
            else:
                labels = torch.tensor(labels,dtype=torch.int64)
            return batched_graph,labels
        else:
            graphs = [s[0] for s in samples]
            batched_graph = dgl.batch(graphs)
            return batched_graph


    def getLoader(self, X, y, batch_size, schuffle=True, include_labels=False):
        graphs, labels = [], []
        for i in range(len(X)):

            smiles = X[i][0]
            mol = Chem.MolFromSmiles(smiles)

            if mol is None:
                print(f"[WARNING] Skipping invalid SMILES at index {i}: {smiles}")
                continue
            graph = mol_to_bigraph( 
                mol,              
                node_featurizer=CanonicalAtomFeaturizer(),
                edge_featurizer=CanonicalBondFeaturizer(),
                explicit_hydrogens=False
            )
            graph = dgl.add_self_loop(graph)
            graphs.append(graph)
            if include_labels:
                label = y[i]
                labels.append(label)
        if include_labels:
            labels = torch.tensor(labels).unsqueeze(1)

            loader = GraphDataLoader(
                list(zip(graphs, labels)),
                batch_size=batch_size,
                shuffle=True if schuffle else False,
                collate_fn=self.collate,
                num_workers=0)
        else:
            loader = GraphDataLoader(
                list(zip(graphs)),
                batch_size=batch_size,
                shuffle=True if schuffle else False,
                collate_fn=self.collate,
                num_workers=0)
        return loader
    

    def fit(self, X, y, Xval=None, yval=None, monitor=None, optimizer=None):
        print("Fitting...")
        self.to(self.device)
        monitor = BaseMonitor() if monitor is None else monitor

        train_loader = self.getLoader(X, y, batch_size=self.batch_size, schuffle=True, include_labels=True)
        val_loader = None
        if (Xval is not None and yval is not None):
            val_loader = self.getLoader(Xval, yval, batch_size=self.batch_size, schuffle=False, include_labels=True)
            patience = self.patience
        else:
            patience = -1
        if "optim" in self.__dict__:
            optimizer = self.optim
        else:
            optimizer = optim.Adam(self.parameters(), lr=self.lr)
        
        best_loss = np.inf
        best_weights = self.state_dict()
        last_save = 0  # record the epoch when optimal model is saved.
        for epoch in range(self.n_epochs):
            monitor.onEpochStart(epoch)
            loss = None
            self.train()
            # decrease learning rate over the epochs
            for param_group in optimizer.param_groups:
                param_group["lr"] = self.lr * (1 - 1 / self.n_epochs) ** (epoch * 10)
            for batch_idx, (batched_graph, target) in enumerate(train_loader):
                monitor.onBatchStart(batch_idx)
                # Batch of target tenor and label tensor
                batched_graph, target = batched_graph.to(self.device), target.to(self.device)
                batched_graph.ndata['y'] = batched_graph.ndata['y'].float().to(self.device)
                optimizer.zero_grad()
                # predicted probability tensor
                logits = self.forward(batched_graph, batched_graph.ndata['y'], is_train=True)

                if self.n_class > 1:
                    loss = self.criterion(logits, target.long())
                else:
                    loss = self.criterion(logits, target)
                loss.backward()
                optimizer.step()
                monitor.onBatchEnd(batch_idx, float(loss))
            if patience == -1:
                monitor.onEpochEnd(epoch, loss.item())
            else:
                # loss value on validation set based on which optimal model is saved.
                loss_valid = self.evaluate(val_loader)
                if loss_valid + self.tol < best_loss:
                    best_weights = self.state_dict()
                    best_loss = loss_valid
                    last_save = epoch
                elif epoch - last_save > patience:  # early stop
                    break
                monitor.onEpochEnd(epoch, loss.item(), loss_valid)
        if patience == -1:
            best_weights = self.state_dict()
        self.load_state_dict(best_weights)
        return self, last_save
    
    def predict(self, X):
        test_loader = self.getLoader(X, y=None, batch_size=self.batch_size, schuffle=False, include_labels=False)
        self.eval()
        score = []
        with torch.no_grad():      
            for batched_graph in test_loader:       
                batched_graph = batched_graph.to(self.device)   
                node_features = batched_graph.ndata['y'].to(self.device).float()
                logits = self.forward(batched_graph, node_features)

                score.append(logits.detach().cpu())
        score = torch.cat(score, dim=0).numpy()
        return score
    
    def evaluate(self, loader) -> float:
        """Evaluate the performance of the DNN model.

        Args:
            loader (torch.util.data.DataLoader):
                data loader for test set,
                including m X n target FloatTensor and l X n label FloatTensor
                (m is the No. of sample, n is the No. of features, l is the
                No. of classes or tasks)

        Return:
            loss (float):
                the average loss value based on the calculation of loss
                function with given test set.
        """
        self.to(self.device)
        self.eval()
        loss = 0
        with torch.no_grad():
            for batched_graph, target in loader:
                batched_graph, target = batched_graph.to(self.device), target.to(self.device)
                batched_graph.ndata["y"] = batched_graph.ndata["y"].float().to(self.device)
                logits = self.forward(batched_graph, batched_graph.ndata['h'], is_train=False)

                if self.n_class > 1:
                    loss += self.criterion(logits, target.long()).item()
                else:
                    loss += self.criterion(logits, target).item()
        loss = loss / len(loader)
        return loss


class DNNModel(QSPRModelPyTorchGPU):
    """This class holds the methods for training and fitting a
    Deep Neural Net QSPR model initialization.

    Here the model instance is created and parameters can be defined.

    Attributes:
        name (str): name of the model
        alg (estimator): estimator instance or class
        parameters (dict): dictionary of algorithm specific parameters
        estimator (object):
            the underlying estimator instance, if `fit` or optimization is performed,
            this model instance gets updated accordingly
        featureCalculators (MoleculeDescriptorsCalculator):
            feature calculator instance taken from the data set
            or deserialized from file if the model is loaded without data
        featureStandardizer (SKLearnStandardizer):
            feature standardizer instance taken from the data set
            or deserialized from file if the model is loaded without data
        baseDir (str):
            base directory of the model, the model files
            are stored in a subdirectory `{baseDir}/{outDir}/`
        patience (int):
            number of epochs to wait before early stop if no progress
            on validation set score
        tol (float):
            minimum absolute improvement of loss necessary to count as
            progress on best validation score
        nClass (int): number of classes
        nDim (int): number of features
        patience (int):
            number of epochs to wait before early stop
            if no progress on validation set score
    """

    def getGPUs(self):
        return self.gpus

    def setGPUs(self, gpus: list[int]):
        self.gpus = gpus
        if not isinstance(self.estimator, str):
            self.estimator.gpus = gpus
        if torch.cuda.is_available() and gpus:
            self.setDevice(f"cuda:{gpus[0]}")
        else:
            self.setDevice("cpu")

    def getDevice(self) -> torch.device:
        return self.device

    def setDevice(self, device: str):
        self.device = torch.device(device)
        if isinstance(self.estimator, nn.Module):
            self.estimator.device = self.device

    def __init__(
            self,
            base_dir: str,
            alg: Type = GGNN,
            name: str | None = None,
            parameters: dict | None = None,
            random_state: int | None = None,
            autoload: bool = True,
            gpus: list[int] = DEFAULT_TORCH_GPUS,
            patience: int = 50,
            tol: float = 0,
            optimizer = torch.optim.Adam,
            activation = F.relu
    ):
        """Initialize a DNNModel model.

        Args:
            base_dir (str):
                base directory of the model, the model files are stored in
                a subdirectory `{baseDir}/{outDir}/`
            alg (Type, optional):
                model class or instance. Defaults to STFullyConnected.
            name (str, optional):
                name of the model. Defaults to None.
            parameters (dict, optional):
                dictionary of algorithm specific parameters. Defaults to None.
            autoload (bool, optional):
                whether to load the model from file or not. Defaults to True.
            device (torch.device, optional):
                The cuda device. Defaults to `DEFAULT_TORCH_DEVICE`.
            gpus (list[int], optional):
                gpu number(s) to use for model fitting. Defaults to `DEFAULT_TORCH_GPUS`.
            patience (int, optional):
                number of epochs to wait before early stop if no progress
                on validation set score. Defaults to 50.
            tol (float, optional):
                minimum absolute improvement of loss necessary to count as progress
                on best validation score. Defaults to 0.
        """

        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = None
        self.gpus = None
        self.patience = patience
        self.tol = tol
        self.optimizer = optimizer
        self.activation = activation
        self.nClass = None
        self.nDim = None
                
        super().__init__(
            base_dir,
            alg,
            name,
            parameters,
            autoload=autoload,
            random_state=random_state,
        )
        self.setGPUs(gpus)


    def initRandomState(self, random_state):
        """Set random state if applicable.
        Defaults to random state of dataset if no random state is provided by the constructor.

        Args:
            random_state (int): Random state to use for shuffling and other random operations.
        """
        super().initRandomState(random_state)
        if random_state is not None:
            torch.manual_seed(random_state)

    @property
    def supportsEarlyStopping(self) -> bool:
        """Whether the model supports early stopping or not."""
        return True

    def initFromDataset(self, data: QSPRDataset | None):
        super().initFromDataset(data)
        if self.targetProperties[0].task.isRegression():
            self.nClass = 1
        elif data is not None:
            self.nClass = self.targetProperties[0].nClasses
        if data is not None:
            self.nDim = data.getFeatures()[0].shape[1]

    def loadEstimator(self, params: dict | None = None) -> object:
        """Load model from file or initialize new model.

        Args:
            params (dict, optional): model parameters. Defaults to None.

        Returns:
            model (object): model instance
        """
        if self.nClass is None or self.nDim is None:
            return "Uninitialized model."
        # initialize model - GGNN here
        
        estimator = self.alg(
            n_dim=self.nDim,
            n_class=self.nClass,
            device=str(self.device),
            gpus=self.gpus,
            is_reg=self.task == ModelTasks.REGRESSION,
            patience=self.patience,
            tol=self.tol,
            #parameters=params,
            activation_in=self.activation
        ).to(self.device)
        # set parameters if available and return
        #FIXED load parameters
        new_parameters = self.getParameters(params)
        if new_parameters is not None:
           estimator.set_params(**new_parameters)
        return estimator

    def loadEstimatorFromFile(
            self, params: dict | None = None, fallback_load: bool = True
    ) -> object:
        """Load estimator from file.

        Args:
            params (dict): parameters
            fallback_load (bool):
                if `True`, init estimator from `alg` and `params` if no estimator
                found at path

        Returns:
            estimator (object): estimator instance
        """
        path = f"{self.outPrefix}_weights.pkg"
        estimator = self.loadEstimator(params)
        if estimator == "Uninitialized model.":
            return estimator
        # load states if available
        if os.path.exists(path):
            estimator.load_state_dict(torch.load(path))
        elif not fallback_load:
            raise FileNotFoundError(
                f"No estimator found at {path}, "
                f"loading estimator weights from file failed."
            )
        return estimator

    def saveEstimator(self) -> str:
        """Save the DNNModel model.

        Returns:
            str: path to the saved model
        """
        path = f"{self.outPrefix}_weights.pkg"
        if not isinstance(self.estimator, str):
            torch.save(self.estimator.state_dict(), path)
        else:
            # just save the estimator message
            with open(path, "w") as f:
                f.write(self.estimator)
        return path

    @early_stopping
    def fit(
            self,
            X: pd.DataFrame | np.ndarray,
            y: pd.DataFrame | np.ndarray,
            estimator: Any | None = None,
            mode: EarlyStoppingMode = EarlyStoppingMode.NOT_RECORDING,
            split: DataSplit | None = None,
            monitor: FitMonitor | None = None,
            **kwargs,
    ):
        """Fit the model to the given data matrix or `QSPRDataset`.

        Args:
            X (pd.DataFrame, np.ndarray): data matrix to fit
            y (pd.DataFrame, np.ndarray): target matrix to fit
            estimator (Any): estimator instance to use for fitting
            mode (EarlyStoppingMode): early stopping mode
            split (DataSplit): data split to use for early stopping,
                if None, a ShuffleSplit with 10% validation set size is used
            monitor (FitMonitor): fit monitor instance, if None, a BaseMonitor is used
            kwargs (dict): additional keyword arguments for the estimator's fit method

        Returns:
            Any: fitted estimator instance
            int, optional: in case of early stopping, the number of iterations
                after which the model stopped training
        """
        if self.task.isMultiTask():
            raise NotImplementedError(
                "Multitask modelling is not implemented for this model."
            )
        monitor = BaseMonitor() if monitor is None else monitor
        estimator = self.estimator if estimator is None else estimator
        estimator.device = self.device
        estimator.gpus = self.gpus
        split = split or ShuffleSplit(
            n_splits=1, test_size=0.1, random_state=self.randomState
        )
        X, y = self.convertToNumpy(X, y)

        # fit with early stopping
        if self.earlyStopping:
            # split cross validation fold train set into train
            # and validation set for early stopping
            train_index, val_index = next(split.split(X, y))
            monitor.onFitStart(
                self, X[train_index, :], y[train_index], X[val_index, :], y[val_index]
            )
            
            estimator_fit = estimator.fit(
                X[train_index, :],
                y[train_index],
                X[val_index, :],
                y[val_index],
                monitor=monitor,
                optimizer=self.optimizer,
                **kwargs,
            )
            monitor.onFitEnd(estimator_fit[0], estimator_fit[1])
            return estimator_fit
        monitor.onFitStart(self, X, y)
        # set fixed number of epochs if early stopping is not used
        estimator.n_epochs = self.earlyStopping.getEpochs()
        estimator_fit = estimator.fit(
            X, 
            y, 
            monitor=monitor,
            optimizer=self.optimizer,
            **kwargs
        )
        monitor.onFitEnd(estimator_fit[0])
        return estimator_fit

    def predict(
            self, X: pd.DataFrame | np.ndarray | QSPRDataset, estimator: Any = None
    ) -> np.ndarray:
        """See `QSPRModel.predict`."""
        estimator = self.estimator if estimator is None else estimator
        estimator.device = self.device
        estimator.gpus = self.gpus
        scores = self.predictProba(X, estimator)
        # return class labels for classification
        if self.task.isClassification():
            return np.argmax(scores[0], axis=1, keepdims=True)
        else:
            return scores[0]

    def predictProba(
            self, X: pd.DataFrame | np.ndarray | QSPRDataset, estimator: Any = None
    ) -> np.ndarray:
        """See `QSPRModel.predictProba`."""
        estimator = self.estimator if estimator is None else estimator
        estimator.device = self.device
        estimator.gpus = self.gpus
        X = self.convertToNumpy(X)

        return [estimator.predict(X)]