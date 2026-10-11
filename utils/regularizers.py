import numpy as np

def lasso(weights: np.ndarray, lasso_coef, ridge_coef):
    """
    Calculate L1 regularization (Lasso) penalty and gradient.

    Args:
        weights: Model weights.
        lasso_coef: L1 regularization coefficient.
        ridge_coef: L2 regularization coefficient (unused in Lasso).

    Returns:
        Tuple of (regularization penalty, gradient of penalty).
    """
    return lasso_coef * np.sum(np.abs(weights)), lasso_coef * np.sign(weights)


def ridge(weights: np.ndarray, lasso_coef, ridge_coef):
    """
    Calculate L2 regularization (Ridge) penalty and gradient.

    Args:
        weights: Model weights.
        lasso_coef: L1 regularization coefficient (unused in Ridge).
        ridge_coef: L2 regularization coefficient.

    Returns:
        Tuple of (regularization penalty, gradient of penalty).
    """
    return ridge_coef * np.sum(weights**2), ridge_coef * (2 * weights)


def elasticnet(weights: np.ndarray, l1, l2):
    """
    Calculate Elastic Net regularization penalty and gradient.

    Args:
        weights: Model weights.
        l1: L1 regularization coefficient.
        l2: L2 regularization coefficient.

    Returns:
        Tuple of (regularization penalty, gradient of penalty).
    """
    offset_lasso = lasso(weights, l1, 0)
    offset_ridge = ridge(weights, 0, l2)
    return offset_lasso[0] + offset_ridge[0], offset_lasso[1] + offset_ridge[1]