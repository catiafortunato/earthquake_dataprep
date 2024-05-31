import numpy as np
import pandas as pd

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