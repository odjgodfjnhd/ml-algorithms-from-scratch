import pandas as pd
import numpy as np
from typing import Callable
import random
from utils.metrics import mae, mse, rmse, mape, r2
from utils.regularizers import lasso, ridge, elasticnet

class MyLineReg:
    """
    Linear Regression model with support for SGD, regularization, and custom metrics.

    Attributes:
        _METRICS: Dictionary of available metric functions.
        _REGS: Dictionary of available regularization functions.
    """

    _METRICS = {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "r2": r2,
    }
    _REGS = {"l1": lasso, "l2": ridge, "elasticnet": elasticnet}

    def __init__(
        self,
        n_iter: int = 100,
        learning_rate: float | Callable = 0.1,
        weights: np.ndarray | None = None,
        metric=None,
        reg=None,
        l1_coef=0,
        l2_coef=0,
        sgd_sample=None,
        random_state=42,
    ):
        """
        Initialize Linear Regression model.

        Args:
            n_iter: Number of training iterations.
            learning_rate: Learning rate (float or callable for dynamic rate).
            weights: Initial weights (optional).
            metric: Metric name for evaluation ('mse', 'mae', 'rmse', 'mape', 'r2').
            reg: Regularization type ('l1', 'l2', 'elasticnet').
            l1_coef: L1 regularization coefficient.
            l2_coef: L2 regularization coefficient.
            sgd_sample: Sample size for SGD (int, float for fraction, or None for batch).
            random_state: Random seed for reproducibility.
        """
        self.n_iter = n_iter
        if isinstance(learning_rate, (int, float)):
            self.learning_rate = lambda x: learning_rate
        else:
            self.learning_rate = learning_rate
        self.weights = weights
        self.metric = metric
        self.metric_f = self._METRICS[metric] if metric is not None else None
        self.reg = reg
        self.reg_f = self._REGS[reg] if reg is not None else None
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose:bool | int = False):
        """
        Train the linear regression model.

        Args:
            X: Feature matrix (DataFrame).
            y: Target values (Series).
            verbose: If True or int, print training progress every N steps.
        """
        random.seed(self.random_state)
        X_np, y_np = X.to_numpy(), y.to_numpy()
        X_np = np.hstack((np.ones((X.shape[0], 1)), X_np))
        samples_n = X_np.shape[0]
        feature_n = X_np.shape[1]
        self.weights = np.ones(feature_n)
        if self.sgd_sample is None:
            batch_size = samples_n
        elif isinstance(self.sgd_sample, float):
            batch_size = int(samples_n * self.sgd_sample)
        else:
            batch_size = self.sgd_sample
        for step in range(self.n_iter):
            self._train_step(X_np, y_np, batch_size, step)
            if verbose and (step == 0 or (step + 1) % verbose == 0):
                y_predicted = X_np @ self.weights
                MSE = np.mean((y_predicted - y_np) ** 2)
                if self.metric_f is not None:
                    metric = self.metric_f(y_np, y_predicted)
                else:
                    metric = "+inf"
                self._log(step, MSE, metric)

    def get_coef(self) -> np.ndarray:
        """Returns models coefficients without bias"""
            

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.

        Args:
            X: Feature matrix (DataFrame).

        Returns:
            Array of predicted values.
        """
        X_np = X.to_numpy()
        X_np = np.hstack((np.ones((X_np.shape[0], 1)), X_np))
        y_predict = X_np @ self.weights
        return y_predict

    def _train_step(
        self, X: np.ndarray, y: np.ndarray, batch_size: int, step_number: int
    ) -> None:
        """
        Perform one training step (gradient descent update).

        Args:
            X: Feature matrix (numpy array with bias column).
            y: Target values (numpy array).
            batch_size: Number of samples in mini-batch.
            step_number: Current step number (for dynamic learning rate).
        """
        sample_rows_idx = random.sample(range(X.shape[0]), batch_size)
        X_batch = X[sample_rows_idx]
        y_batch = y[sample_rows_idx]
        y_predicted = X_batch @ self.weights
        reg_offset = (
            self.reg_f(self.weights, self.l1_coef, self.l2_coef)
            if self.reg_f is not None
            else (0, 0)
        )
        grad = (2 / batch_size) * ((y_predicted - y_batch) @ X_batch) + reg_offset[1]
        self.weights -= self.learning_rate(step_number + 1) * grad

    def _log(self, step_number: int, MSE: float, metric: float|str) -> None:
        """
        Print training progress.

        Args:
            step_number: Current step number.
            MSE: Mean squared error on full dataset.
            metric: Value of the selected metric.
        """
        print(f"step={step_number} | MSE = {MSE} | metric {self.metric} = {metric}")