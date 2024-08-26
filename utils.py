import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, FactorAnalysis


def extract_spike_trial_data(spike_data, event_time, bin_size, bins_before_after):
    event_bin = event_time // bin_size  # Convert event time to bin index
    start_bin = event_bin - bins_before_after
    end_bin = event_bin + bins_before_after + 1  # Include the event bin
    
    # Ensure the indices are within bounds
    if start_bin >= 0 and end_bin <= spike_data.shape[1]:
        trial_data = spike_data[:,start_bin:end_bin]
        return trial_data.flatten()
    else:
        return None

def extract_keypoints_trial_data(spike_data, event_time, bin_size, bins_before_after):
    event_bin = event_time // bin_size  # Convert event time to bin index
    start_bin = event_bin - bins_before_after
    end_bin = event_bin + bins_before_after + 1  # Include the event bin
    
    # Ensure the indices are within bounds
    if start_bin >= 0 and end_bin <= spike_data.shape[0]:
        print(start_bin)
        print(end_bin)
        trial_data = spike_data[start_bin:end_bin]
        return trial_data.flatten()
    else:
        return None
    
def cum_variance_pca(arr):
    pca = PCA().fit(arr)
    return np.cumsum(pca.explained_variance_ratio_)

def participation_ratio(explained_variances):
    """
    Estimate the number of "important" components based on explained variances

    Parameters
    ----------
    explained_variances : 1D np.ndarray
        explained variance per dimension

    Returns
    -------
    dimensionality estimated using participation ratio formula
    """
    return np.sum(explained_variances) ** 2 / np.sum(explained_variances ** 2)

def pca_pr(arr):
    """
    Estimate the data's dimensionality using PCA and participation ratio
    
    Parameters
    ----------
    arr : 2D array
        n_samples x n_features data
    
    Returns
    -------
    estimated dimensionality
    """
    pca = PCA().fit(arr)
    return participation_ratio(pca.explained_variance_)