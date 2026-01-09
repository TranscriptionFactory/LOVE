"""
Estimation of non-pure rows of the loading matrix A.
Translated from R/EstNonpure.R
"""

import numpy as np
from scipy.optimize import linprog
from typing import List, Optional


def EstY(Sigma: np.ndarray, AI: np.ndarray, pureVec: np.ndarray) -> np.ndarray:
    """
    Estimate the K by |J| submatrix of Sigma.

    Parameters
    ----------
    Sigma : np.ndarray
        The p by p covariance matrix.
    AI : np.ndarray
        The p by K loading matrix.
    pureVec : np.ndarray
        Array of pure variable indices.

    Returns
    -------
    np.ndarray
        A K by |J| matrix (where J is the set of non-pure variables).
    """
    pureVec = list(pureVec)
    p = Sigma.shape[0]

    # Get non-pure indices
    nonPureVec = [i for i in range(p) if i not in pureVec]

    AI_sub = AI[pureVec, :]

    # solve(crossprod(AI_sub), t(AI_sub) @ Sigma[pureVec, -pureVec])
    # = inv(AI_sub.T @ AI_sub) @ AI_sub.T @ Sigma[pureVec, nonPureVec]
    cross_AI = AI_sub.T @ AI_sub
    Sigma_IJ = Sigma[np.ix_(pureVec, nonPureVec)]

    try:
        Y = np.linalg.solve(cross_AI, AI_sub.T @ Sigma_IJ)
    except np.linalg.LinAlgError:
        Y = np.linalg.pinv(cross_AI) @ AI_sub.T @ Sigma_IJ

    return Y


def EstAJInv(Omega: np.ndarray, Y: np.ndarray, lbd: float) -> np.ndarray:
    """
    Estimate non-pure rows via soft-thresholding.

    Estimates the |J| by K submatrix A_J by using soft thresholding.

    Parameters
    ----------
    Omega : np.ndarray
        The estimated precision matrix of Z.
    Y : np.ndarray
        A K by |J| response matrix.
    lbd : float
        Tuning parameter for soft-thresholding.

    Returns
    -------
    np.ndarray
        A |J| by K matrix.
    """
    n_J = Y.shape[1]  # Number of non-pure variables
    K = Y.shape[0]    # Number of factors
    AJ = np.zeros((n_J, K))

    for i in range(n_J):
        Atilde = Omega @ Y[:, i]
        AJ[i, :] = LP(Atilde, lbd)
        # Normalize if L1 norm > 1
        if np.sum(np.abs(AJ[i, :])) > 1:
            AJ[i, :] = AJ[i, :] / np.sum(np.abs(AJ[i, :]))

    return AJ


def LP(y: np.ndarray, lbd: float) -> np.ndarray:
    """
    Soft-thresholding via linear program.

    Solves:
        min sum(beta_pos + beta_neg)
        s.t. beta_pos - beta_neg <= lbd + y
             -beta_pos + beta_neg <= lbd - y
             beta_pos >= 0, beta_neg >= 0

    Parameters
    ----------
    y : np.ndarray
        A vector of length K.
    lbd : float
        Threshold parameter.

    Returns
    -------
    np.ndarray
        A vector of length K (beta = beta_pos - beta_neg).
    """
    K = len(y)

    # Variables layout matches R: [beta_1_pos, beta_2_pos, ..., beta_K_pos,
    #                              beta_1_neg, beta_2_neg, ..., beta_K_neg]
    # Total: 2*K variables

    # Objective: minimize sum of all variables
    c = np.ones(2 * K)

    # Build constraint matrix C where C[k, k] = 1, C[k, k+K] = -1
    # So C @ x = beta_pos - beta_neg
    # Constraints: C @ x <= lbd + y, -C @ x <= lbd - y
    C = np.zeros((K, 2 * K))
    for k in range(K):
        C[k, k] = 1        # beta_k_pos
        C[k, k + K] = -1   # beta_k_neg

    # Stack constraints: [C; -C] @ x <= [lbd + y; lbd - y]
    A_ub = np.vstack([C, -C])
    b_ub = np.concatenate([lbd + y, lbd - y])

    # Bounds: all variables >= 0
    bounds = [(0, None)] * (2 * K)

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    if result.success:
        solution = result.x
        # beta = beta_pos - beta_neg (R: LPsol[1:K] - LPsol[(K+1):(2*K)])
        beta = solution[:K] - solution[K:]
        return beta
    else:
        # Return soft-thresholded y as fallback
        return np.sign(y) * np.maximum(np.abs(y) - lbd, 0)


def EstAJDant(C_hat: np.ndarray, Y: np.ndarray, lbd: float,
              se_est_J: np.ndarray) -> np.ndarray:
    """
    Estimate non-pure rows via the Dantzig approach.

    Parameters
    ----------
    C_hat : np.ndarray
        The estimated covariance matrix of Z.
    Y : np.ndarray
        A K by |J| response matrix.
    lbd : float
        Base tuning parameter.
    se_est_J : np.ndarray
        Estimated standard errors of the non-pure variables.

    Returns
    -------
    np.ndarray
        A |J| by K matrix.
    """
    n_J = Y.shape[1]
    K = Y.shape[0]
    AJ = np.zeros((n_J, K))

    for i in range(n_J):
        AJ[i, :] = Dantzig(C_hat, Y[:, i], lbd * se_est_J[i])
        # Normalize if L1 norm > 1
        if np.sum(np.abs(AJ[i, :])) > 1:
            AJ[i, :] = AJ[i, :] / np.sum(np.abs(AJ[i, :]))

    return AJ


def Dantzig(C_hat: np.ndarray, y: np.ndarray, lbd: float) -> np.ndarray:
    """
    The Dantzig approach for solving one non-pure row.

    Solves:
        min sum(beta_pos + beta_neg)
        s.t. C_hat @ (beta_pos - beta_neg) - y <= lbd  (element-wise)
             -C_hat @ (beta_pos - beta_neg) + y <= lbd  (element-wise)
             beta_pos >= 0, beta_neg >= 0

    Parameters
    ----------
    C_hat : np.ndarray
        The covariance matrix estimate.
    y : np.ndarray
        Response vector.
    lbd : float
        Threshold parameter.

    Returns
    -------
    np.ndarray
        A vector of length K.
    """
    K = len(y)

    # Variables layout matches R: [beta_1_pos, beta_2_pos, ..., beta_K_pos,
    #                              beta_1_neg, beta_2_neg, ..., beta_K_neg]
    # Total: 2*K variables

    # Objective: minimize sum of all variables
    c = np.ones(2 * K)

    # Build constraint matrix matching R:
    # new_C_hat[k, :] = [C_hat[k, :], -C_hat[k, :]]
    # So new_C_hat @ x = C_hat @ beta_pos - C_hat @ beta_neg = C_hat @ beta
    new_C_hat = np.zeros((K, 2 * K))
    for k in range(K):
        new_C_hat[k, :K] = C_hat[k, :]      # beta_pos coefficients
        new_C_hat[k, K:] = -C_hat[k, :]     # beta_neg coefficients

    # Stack constraints: [new_C_hat; -new_C_hat] @ x <= [lbd + y; lbd - y]
    A_ub = np.vstack([new_C_hat, -new_C_hat])
    b_ub = np.concatenate([lbd + y, lbd - y])

    # Bounds: all variables >= 0
    bounds = [(0, None)] * (2 * K)

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    if result.success:
        solution = result.x
        # beta = beta_pos - beta_neg (R: LPsol[1:K] - LPsol[(K+1):(2*K)])
        beta = solution[:K] - solution[K:]
        return beta
    else:
        # Return zeros as fallback
        return np.zeros(K)
