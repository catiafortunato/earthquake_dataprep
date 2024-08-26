import numpy as np
import pandas as pd
import sys
import pyaldata as pyd
import os
from sklearn.decomposition import PCA, FactorAnalysis


def bin_spikes(spike_times, neuron_ids, bin_size, start_time=None, end_time=None):

    """

    Bin the spike times into a 2D array.

    Parameters:

    - spike_times: array-like, times at which spikes occurred.

    - neuron_ids: array-like, neuron identifiers corresponding to each spike time.

    - bin_size: float, the size of each time bin.

    - start_time: float, the start time for binning. If None, use the minimum spike time.

    - end_time: float, the end time for binning. If None, use the maximum spike time.

    Returns:

    - binned_spikes: 2D numpy array of shape (number_of_neurons, number_of_bins).

    """

    if start_time is None:

        start_time = np.min(spike_times)

    if end_time is None:

        end_time = np.max(spike_times)

    number_of_neurons = int(np.max(neuron_ids)) + 1

    number_of_bins = int(np.ceil((end_time - start_time) / bin_size))

    # Initialize the binned spikes array

    binned_spikes = np.zeros((number_of_neurons, number_of_bins), dtype=int)

    # Populate the binned spikes array

    for spike_time, neuron_id in zip(spike_times, neuron_ids):

        if start_time <= spike_time < end_time:

            bin_index = int((spike_time - start_time) / bin_size)

            binned_spikes[neuron_id, bin_index] += 1

    return binned_spikes

def load_spike_data (mouse_id, dataset, probe_nb, break_rec): 

    data_dir = '/data/mouse_data/processed/'+mouse_id+'/'+dataset+'/'+dataset+'_ephys/'+dataset+'_g0/'+dataset+'_g0_imec'+probe_nb+'/sorter_output/'
    sys.path.append(data_dir)
    import params

    camps = pd.read_csv(data_dir+'cluster_Amplitude.tsv', sep='\t')['Amplitude'].values
    contam_pct = pd.read_csv(data_dir+'cluster_ContamPct.tsv', sep='\t')['ContamPct'].values
    chan_map =  np.load(data_dir+'channel_map.npy')
    templates =  np.load(data_dir+'templates.npy')
    chan_best = (templates**2).sum(axis=1).argmax(axis=-1) 
    labels = pd.read_csv(data_dir+'cluster_KSLabel.tsv', sep='\t') # label for each cluster (can be good, or multiunit activity --mua)
    amplitudes = np.load(data_dir+'amplitudes.npy')
    st = np.load(data_dir+'spike_times.npy')# time when spike happened. to transform into seconds 
    clu = np.load(data_dir+'spike_clusters.npy') # cluster identity of each spike
    firing_rates = np.unique(clu, return_counts=True)[1] * 30000 / st.max()
    sys.path.append(data_dir)

    sample_rate = params.sample_rate

    # transform spike times to miliseconds

    st_milsec = st/30000.*1000
    bin_size = 10 #in milliseconds
    start_time = 0

    binned_spikes = bin_spikes(st_milsec, clu, bin_size, start_time=start_time, end_time=None)# transform spike times to miliseconds

    # select neurons for each region

    #neurons_str = (chan_best<break_rec) & ((labels['KSLabel']=='good').values)
    neurons_str = chan_best<break_rec
    #neurons_m1 = (chan_best>=break_rec) & ((labels['KSLabel']=='good').values)
    neurons_m1 = chan_best>=break_rec

    binned_str = binned_spikes[neurons_str,:]
    binned_m1 = binned_spikes[neurons_m1,:]

    return binned_str, binned_m1

# Function to compute velocity from position
def compute_velocity(positions, dt):
    # Calculate differences between consecutive positions
    velocity = np.diff(positions, axis=0) / dt
    # Append a zero velocity for the last position to keep the array shape consistent
    velocity = np.vstack([velocity, np.zeros((1, 3))])
    return velocity

# Function to create lagged data
def create_lagged_data(X, n_lags):
    n_samples, n_features = X.shape
    X_lagged = np.zeros((n_samples, n_features * n_lags))
    for lag in range(n_lags):
        X_lagged[lag:, lag*n_features:(lag+1)*n_features] = X[:n_samples-lag, :]
    return X_lagged

def process_data(dataset, mouse_id):
    data_dir = '/data/mouse_data/processed/'+mouse_id+'/'+dataset+'/'
    fname = os.path.join(data_dir, dataset+'_pyaldata.mat')

    df = pyd.mat2dataframe(fname, shift_idx_fields=False, td_name='df')

    df['bin_size'] = 0.01

    df = pyd.remove_low_firing_neurons(df, "m1_spikes",  1)
    df = pyd.remove_low_firing_neurons(df, "s1_spikes", 1)
    df = pyd.remove_low_firing_neurons(df, "str_motor_spikes",  1)
    df = pyd.remove_low_firing_neurons(df, "str_sensor_spikes", 1)

    df = pyd.transform_signal(df, "m1_spikes",  'sqrt')
    df = pyd.transform_signal(df, "s1_spikes", 'sqrt')
    df = pyd.transform_signal(df, "str_motor_spikes",  'sqrt')
    df = pyd.transform_signal(df, "str_sensor_spikes", 'sqrt')

    df = pyd.merge_signals(df, ['m1_spikes', 's1_spikes'], 'cortical_spikes')
    df = pyd.merge_signals(df, ['str_motor_spikes', 'str_sensor_spikes'], 'striatal_spikes')
    df = pyd.merge_signals(df, ['m1_spikes', 's1_spikes','str_motor_spikes', 'str_sensor_spikes'], 'all_spikes')


    df = pyd.add_firing_rates(df,'smooth')

    list_of_keypoints = [col for col in df.columns if '_pos' in col]

    dt = 0.01 # 10ms bins

    for keypoint in list_of_keypoints:

        string = keypoint
        keypoint = string.rsplit('_', 1)[0]

        df[keypoint+'_vel'] = df[keypoint+'_pos'].apply(lambda pos: compute_velocity(pos, dt))

    pca_dims = 15

    df = pyd.dim_reduce(df, PCA(pca_dims), "m1_rates", "m1_pca")
    df = pyd.dim_reduce(df, PCA(pca_dims), "s1_rates", "s1_pca")
    df = pyd.dim_reduce(df, PCA(pca_dims), "str_motor_rates", "str_motor_pca")
    df = pyd.dim_reduce(df, PCA(pca_dims), "str_sensor_rates", "str_sensor_pca")

    return df


