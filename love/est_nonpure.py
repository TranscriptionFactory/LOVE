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

    # Variables: [beta_1_pos, beta_1_neg, beta_2_pos, beta_2_neg, ...]
    # Total: 2*K variables

    # Objective: minimize sum of all variables
    c = np.ones(2 * K)

    # Constraints:
    # beta_pos - beta_neg <= lbd + y  for each k
    # -beta_pos + beta_neg <= lbd - y  for each k
    # beta_pos >= 0, beta_neg >= 0 (handled by bounds)

    A_ub_list = []
    b_ub_list = []

    for k in range(K):
        # beta_k_pos - beta_k_neg <= lbd + y[k]
        row_pos = np.zeros(2 * K)
        row_pos[2 * k] = 1      # beta_k_pos
        row_pos[2 * k + 1] = -1  # -beta_k_neg
        A_ub_list.append(row_pos)
        b_ub_list.append(lbd + y[k])

        # -beta_k_pos + beta_k_neg <= lbd - y[k]
        row_neg = np.zeros(2 * K)
        row_neg[2 * k] = -1     # -beta_k_pos
        row_neg[2 * k + 1] = 1   # beta_k_neg
        A_ub_list.append(row_neg)
        b_ub_list.append(lbd - y[k])

    A_ub = np.array(A_ub_list)
    b_ub = np.array(b_ub_list)

    # Bounds: all variables >= 0
    bounds = [(0, None)] * (2 * K)

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    if result.success:
        solution = result.x
        # beta = beta_pos - beta_neg
        beta = np.zeros(K)
        for k in range(K):
            beta[k] = solution[2 * k] - solution[2 * k + 1]
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

    # Variables: [beta_1_pos, beta_1_neg, beta_2_pos, beta_2_neg, ...]
    # Total: 2*K variables

    # Objective: minimize sum of all variables
    c = np.ones(2 * K)

    # Build constraint matrix
    # C_hat @ (beta_pos - beta_neg) <= lbd + y
    # -C_hat @ (beta_pos - beta_neg) <= lbd - y

    A_ub_list = []
    b_ub_list = []

    # Build the coefficient matrix for C_hat @ beta
    # new_C_hat[k, :] = [C_hat[k, 0], -C_hat[k, 0], C_hat[k, 1], -C_hat[k, 1], ...]
    new_C_hat = np.zeros((K, 2 * K))
    for k in range(K):
        for j in range(K):
            new_C_hat[k, 2 * j] = C_hat[k, j]      # beta_j_pos coefficient
            new_C_hat[k, 2 * j + 1] = -C_hat[k, j]  # beta_j_neg coefficient

    # C_hat @ beta <= lbd + y
    for k in range(K):
        A_ub_list.append(new_C_hat[k, :])
        b_ub_list.append(lbd + y[k])

    # -C_hat @ beta <= lbd - y
    for k in range(K):
        A_ub_list.append(-new_C_hat[k, :])
        b_ub_list.append(lbd - y[k])

    A_ub = np.array(A_ub_list)
    b_ub = np.array(b_ub_list)

    # Bounds: all variables >= 0
    bounds = [(0, None)] * (2 * K)

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    if result.success:
        solution = result.x
        # beta = beta_pos - beta_neg
        beta = np.zeros(K)
        for k in range(K):
            beta[k] = solution[2 * k] - solution[2 * k + 1]
        return beta
    else:
        # Return zeros as fallback
        return np.zeros(K)
