import numpy as np

def mae(
    vector1: np.ndarray,
    vector2: np.ndarray,
):
    """
    Calculate MAE.

    Args:
        vector1 (np.ndarray): True values.
        vector2 (np.ndarray): Predicted values.
    Returns:
        float: MAE
    """
    return np.mean(np.abs(vector1 - vector2))


def mse(vector1: np.ndarray, vector2: np.ndarray):
    """
    Calculate Mean Squared Error (MSE).

    Args:
        vector1: True values.
        vector2: Predicted values.

    Returns:
        Mean squared error.
    """
    return np.mean(((vector1 - vector2) ** 2))


def rmse(vector1: np.ndarray, vector2: np.ndarray):
    """
    Calculate Root Mean Squared Error (RMSE).

    Args:
        vector1: True values.
        vector2: Predicted values.

    Returns:
        Root mean squared error.
    """
    return np.sqrt(mse(vector1, vector2))


def mape(vector1: np.ndarray, vector2: np.ndarray):
    """
    Calculate Mean Absolute Percentage Error (MAPE).

    Args:
        vector1: True values.
        vector2: Predicted values.

    Returns:
        Mean absolute percentage error.
    """
    return np.mean(np.abs((vector1 - vector2) / vector1)) * 100


def r2(vector1: np.ndarray, vector2: np.ndarray):
    """
    Calculate R-squared (coefficient of determination).

    Args:
        vector1: True values.
        vector2: Predicted values.

    Returns:
        R-squared value.
    """
    return 1 - (
        np.sum((vector1 - vector2) ** 2) / np.sum((vector1 - vector1.mean()) ** 2)
    )