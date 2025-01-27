import numpy as np
from scipy.stats import rankdata


def vectorized_rankdata(a):
    """
    Vectorized ranking function using NumPy.

    Parameters:
    -----------
    a : np.array
        Input array to be ranked.

    Returns:
    --------
    ranks : np.array
        Ranked array.
    """
    a = a.flatten()
    ranks = np.empty_like(a, dtype=float)
    ranks[np.argsort(a)] = np.arange(len(a)) + 1
    return ranks

def calculate_spearman_r_map(niftis, indep_var, verbose=False):
    """
    Calculate the Spearman rank-order correlation coefficient for each voxel
    in a fully vectorized manner.

    Parameters:
    -----------
    niftis : np.array
        2D array where each row represents a subject and each column represents a voxel.
    indep_var : np.array
        1D array representing the independent variable for each subject.

    Returns:
    --------
    rho : np.array
        1D array of Spearman's rank correlation coefficients for each voxel.
    """
    # Rank the data using scipy.stats.rankdata to handle ties
    ranked_niftis = np.apply_along_axis(rankdata, 0, niftis)
    ranked_indep_var = rankdata(indep_var)

    # Calculate the Pearson correlation coefficient on the ranked data
    # This generates identical results to Scipy.stats.spearmanr
    X = ranked_indep_var[:, np.newaxis]
    Y = ranked_niftis
    X_BAR = X.mean(axis=0)[:, np.newaxis]
    Y_BAR = Y.mean(axis=0)[np.newaxis, :]
    X_C = X - X_BAR
    Y_C = Y - Y_BAR
    NUMERATOR = np.dot(X_C.T, Y_C)
    SST_X = np.sum((X - X_BAR)**2, axis=0)
    SST_Y = np.sum((Y - Y_BAR)**2, axis=0)
    DENOMINATOR = np.sqrt(SST_X * SST_Y)
    rho = NUMERATOR / DENOMINATOR
    
    if verbose:
        print(f"Shape of X: {X.shape}")
        print(f"Shape of Y: {Y.shape}")
        print(f"Shape of X_BAR: {X_BAR.shape}")
        print(f"Shape of Y_BAR: {Y_BAR.shape}")
        print(f"Shape of X_C: {X_C.shape}")
        print(f"Shape of Y_C: {Y_C.shape}")
        print(f"Shape of NUMERATOR: {NUMERATOR.shape}")
        print(f"Shape of DENOMINATOR: {DENOMINATOR.shape}")
    return rho