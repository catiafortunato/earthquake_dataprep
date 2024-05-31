import numpy as np
import pandas as pd
import sys

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

    st_milsec = st/sample_rate*1000
    bin_size = 10 #in milliseconds
    start_time = 0

    binned_spikes = bin_spikes(st_milsec, clu, bin_size, start_time=start_time, end_time=None)# transform spike times to miliseconds

    st_milsec = st/sample_rate*1000
    bin_size = 10 #in milliseconds
    start_time = 0

    binned_spikes = bin_spikes(st_milsec, clu, bin_size, start_time=start_time, end_time=None)

    # select neurons for each region

    neurons_str = (chan_best<break_rec) & ((labels['KSLabel']=='good').values)
    neurons_m1 = (chan_best>=break_rec) & ((labels['KSLabel']=='good').values)

    binned_str = binned_spikes[neurons_str,:]
    binned_m1 = binned_spikes[neurons_m1,:]

    return binned_str, binned_m1

