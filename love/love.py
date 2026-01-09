"""
LOVE: Latent-model based OVErlapping clustering.
Translated from R/LOVE.R

Main entry point for the LOVE algorithm.
"""

import numpy as np
from typing import Dict, Optional, Union, List
import warnings

from .utilities import recoverGroup, threshA
from .cv import CV_delta, KfoldCV_delta, CV_lbd
from .est_pure_homo import EstAI, EstC, FindSignPureNode
from .est_pure_hetero import Est_BI_C, Re_Est_Pure
from .est_nonpure import EstY, EstAJInv, EstAJDant
from .est_omega import estOmega


def LOVE(X: np.ndarray, lbd: float = 0.5, mu: float = 0.5,
         est_non_pure_row: str = "HT", verbose: bool = False,
         pure_homo: bool = False, diagonal: bool = False,
         delta: Optional[np.ndarray] = None, merge: bool = False,
         rep_CV: int = 50, ndelta: int = 50, q: int = 2,
         exact: bool = False, max_pure: Optional[float] = None,
         nfolds: int = 10) -> Dict:
    """
    LOVE: Latent-model based OVErlapping clustering.

    Perform overlapping (variable) clustering of a p-dimensional feature
    generated from the latent factor model:
        X = A @ Z + E
    with identifiability conditions on A and Cov(Z).

    Parameters
    ----------
    X : np.ndarray
        A n by p data matrix.
    lbd : float, optional
        Leading constant of lambda. Default is 0.5.
    mu : float, optional
        Leading constant used for thresholding the loading matrix. Default is 0.5.
    est_non_pure_row : str, optional
        Procedure used for estimating non-pure rows.
        One of {"HT", "ST", "Dantzig"}. Default is "HT".
    verbose : bool, optional
        If True, print progress. Default is False.
    pure_homo : bool, optional
        If True, pure loadings have the same magnitude. Default is False.
    diagonal : bool, optional
        If True, covariance matrix of Z is diagonal. Default is False.
    delta : np.ndarray, optional
        Grid of leading constant of delta. Default is None.
    merge : bool, optional
        If True, take union of candidate pure variables; else intersection.
        Default is False.
    rep_CV : int, optional
        Number of repetitions for cross validation. Default is 50.
    ndelta : int, optional
        Length of delta grid. Default is 50.
    q : int, optional
        Type of score (2 or np.inf). Default is 2.
    exact : bool, optional
        If True, compute Inf score exactly via LP. Default is False.
    max_pure : float, optional
        Max proportion of pure variables. Default is None.
    nfolds : int, optional
        Number of folds for cross validation. Default is 10.

    Returns
    -------
    Dict
        A dict containing:
        - 'K': Estimated number of clusters
        - 'pureVec': Estimated set of pure variables
        - 'pureInd': Estimated partition of pure variables
        - 'group': Estimated clusters (indices of each cluster)
        - 'A': Estimated p by K assignment matrix
        - 'C': Covariance matrix of Z
        - 'Omega': Precision matrix of Z
        - 'Gamma': Diagonal of covariance matrix of E
        - 'optDelta': Selected value of delta

    Examples
    --------
    >>> import numpy as np
    >>> p, n, K = 6, 100, 2
    >>> A = np.array([[1, 0], [-1, 0], [0, 1], [0, 1], [1/3, 2/3], [1/2, -1/2]])
    >>> Z = np.random.randn(n, K) * np.sqrt(2)
    >>> E = np.random.randn(n, p)
    >>> X = Z @ A.T + E
    >>> res_LOVE = LOVE(X, pure_homo=False, delta=None)
    >>> res_LOVE = LOVE(X, pure_homo=True, delta=np.arange(0.1, 1.2, 0.1))

    References
    ----------
    Bing, X., Bunea, F., Yang N and Wegkamp, M. (2020)
    Adaptive estimation in structured factor models with applications to
    overlapping clustering, Annals of Statistics, Vol.48(4) 2055-2081.

    Bing, X., Bunea, F. and Wegkamp, M. (2021)
    Detecting approximate replicate components of a high-dimensional random
    vector with latent structure.
    """
    n, p = X.shape

    # Centering
    X = X - np.mean(X, axis=0)

    if pure_homo:
        # Estimate the pure rows using homogeneous approach
        se_est = np.std(X, axis=0, ddof=1)  # Sample standard errors

        if delta is not None:
            deltaGrids = delta * np.sqrt(np.log(max(p, n)) / n)
        else:
            deltaGrids = np.array([0.5 * np.sqrt(np.log(max(p, n)) / n)])

        if verbose:
            print("Select delta by using data splitting...")

        if len(deltaGrids) > 1:
            # Multiple CV runs to select delta
            delta_selections = []
            for _ in range(rep_CV):
                selected = CV_delta(X, deltaGrids, diagonal, se_est, merge)
                delta_selections.append(selected)
            optDelta = np.median(delta_selections)
        else:
            optDelta = deltaGrids[0]

        if verbose:
            print("Finish selecting delta and start estimating the pure loadings...")

        Sigma = np.cov(X, rowvar=False)
        resultAI = EstAI(Sigma, optDelta, se_est, merge)

        # Check if there is any group with ONLY ONE pure variable
        pure_numb = [len(x['pos']) + len(x['neg']) for x in resultAI['pureSignInd']]
        if any(n == 1 for n in pure_numb):
            print("Change 'merge' to 'union' and reselecting delta...")
            if len(deltaGrids) > 1:
                delta_selections = []
                for _ in range(rep_CV):
                    selected = CV_delta(X, deltaGrids, diagonal, se_est, merge=False)
                    delta_selections.append(selected)
                optDelta = np.median(delta_selections)
            resultAI = EstAI(Sigma, optDelta, se_est, merge=False)

        A_hat = resultAI['AI']
        I_hat = resultAI['pureVec']
        I_hat_part = resultAI['pureSignInd']

        if I_hat is None or len(I_hat) == 0:
            print("Algorithm fails due to the non-existence of any pure variable.")
            raise RuntimeError("No pure variables found")

        C_hat = EstC(Sigma, A_hat, diagonal)

        # Estimate the covariance matrix of error for non-pure variables
        Gamma_hat = np.zeros(p)
        I_hat_list = list(I_hat)
        diag_Sigma_I = np.diag(Sigma[np.ix_(I_hat_list, I_hat_list)])
        A_I = A_hat[I_hat_list, :]
        diag_ACA = np.diag(A_I @ C_hat @ A_I.T)
        Gamma_hat[I_hat_list] = diag_Sigma_I - diag_ACA

    else:
        # Estimate the pure rows via heterogeneous approach
        R_hat = np.corrcoef(X, rowvar=False)
        Sigma = np.cov(X, rowvar=False)

        if verbose:
            print(f"Select delta by using {nfolds} fold cross-validation...")

        # Find parallel rows and its partition
        CV_res = KfoldCV_delta(X, delta, ndelta, q, exact, nfolds, max_pure)
        pure_res = CV_res['est_pure']
        est_I = pure_res['I_part']
        est_I_set = pure_res['I']
        # Store original partition for FindSignPureNode (R uses pure_res$I_part, not updated est_I)
        original_I_part = pure_res['I_part']

        optDelta = CV_res['delta_min']

        if verbose:
            print("Finish selecting delta and start estimating the pure loadings...")

        # Post-select pure variables
        if len(est_I) >= 2:
            BI_C_res = Est_BI_C(CV_res['moments'], R_hat, est_I, est_I_set)
            est_I = Re_Est_Pure(X, Sigma, CV_res['moments'], est_I, BI_C_res['Gamma'])
            est_I_set = np.array([idx for part in est_I for idx in part])
        elif len(est_I) == 0:
            print("Algorithm fails due to the non-existence of any pure variable.")
            raise RuntimeError("No pure variables found")

        # Re-estimate B and C with potentially updated est_I
        BI_C_res = Est_BI_C(CV_res['moments'], R_hat, est_I, est_I_set)

        D_Sigma = np.diag(Sigma)
        B_hat = BI_C_res['B']
        R_Z = BI_C_res['C']

        # Estimate the loading matrix A and C
        B_hat = np.sqrt(D_Sigma)[:, np.newaxis] * B_hat
        D_B = np.max(np.abs(B_hat), axis=0)
        A_hat = B_hat / D_B
        # R: D_B * R_Z uses column-wise recycled multiplication, not outer product
        C_hat = D_B[:, np.newaxis] * R_Z

        if diagonal:
            C_hat = np.diag(np.diag(C_hat))

        I_hat = est_I_set
        # Use original partition from CV, matching R's: FindSignPureNode(pure_res$I_part, Sigma)
        I_hat_part = FindSignPureNode(original_I_part, Sigma)

        Gamma_hat = BI_C_res['Gamma'] * D_Sigma

    # Ensure non-negative Gamma
    Gamma_hat[Gamma_hat < 0] = 0

    if verbose:
        print("Finish estimating the pure loadings...")

    I_hat_list = list(I_hat)

    if len(I_hat) == p:
        # All variables are pure
        group = I_hat_part
        Omega = None
    else:
        if verbose:
            method_name = {
                "HT": "Hard Thresholding",
                "ST": "Soft Thresholding",
                "Dantzig": "Dantzig"
            }.get(est_non_pure_row, est_non_pure_row)
            print(f"Estimate the non-pure loadings by {method_name}...")

        if est_non_pure_row == "Dantzig":
            AI = np.abs(A_hat[I_hat_list, :])
            cross_AI_inv = np.linalg.solve(AI.T @ AI, AI.T)
            sigma_bar_sup = np.max(cross_AI_inv @ se_est[I_hat_list])
            J_list = [i for i in range(p) if i not in I_hat_list]
            AJ = EstAJDant(C_hat, EstY(Sigma, A_hat, I_hat),
                           mu * optDelta * sigma_bar_sup,
                           sigma_bar_sup + se_est[J_list])
            Omega = None
        else:
            if diagonal:
                Omega = np.diag(1.0 / np.diag(C_hat))
            else:
                if verbose:
                    print("Select lambda for estimating the precision of Z...")

                lbdGrids = np.array([lbd]) * optDelta if np.isscalar(lbd) else lbd * optDelta

                if len(lbdGrids) > 1:
                    lbd_selections = []
                    for _ in range(rep_CV):
                        selected = CV_lbd(X, lbdGrids, A_hat, I_hat, diagonal)
                        lbd_selections.append(selected)
                    optLbd = np.median(lbd_selections)
                else:
                    optLbd = lbdGrids[0]

                if verbose:
                    idx = np.argmin(np.abs(lbdGrids - optLbd))
                    print(f"Select lambda = {optLbd} with leading constant {idx + 1}...")
                    print("Start estimating the precision of Z...")

                Omega = estOmega(optLbd, C_hat)

            Y = EstY(Sigma, A_hat, I_hat)
            # R's norm(Omega, "I") is infinity norm = max absolute row sum
            threshold = mu * optDelta * np.max(np.sum(np.abs(Omega), axis=1))

            if est_non_pure_row == "HT":
                AJ = threshA((Omega @ Y).T, threshold)
            elif est_non_pure_row == "ST":
                AJ = EstAJInv(Omega, Y, threshold)
            else:
                warnings.warn(f"Unknown method '{est_non_pure_row}' for estimating non-pure rows.")
                AJ = threshA((Omega @ Y).T, threshold)

        # Fill in non-pure rows
        J_list = [i for i in range(p) if i not in I_hat_list]
        A_hat[J_list, :] = AJ
        group = recoverGroup(A_hat)

    return {
        'K': A_hat.shape[1],
        'pureVec': I_hat,
        'pureInd': I_hat_part,
        'group': group,
        'A': A_hat,
        'C': C_hat,
        'Omega': Omega,
        'Gamma': Gamma_hat,
        'optDelta': optDelta
    }
