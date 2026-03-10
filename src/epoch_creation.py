# ------------------------------------------------ #
# Function: Make epochs from tic events 
# Author: Martyna
# Goal: Based on the extracted start times of tics make epochs around it 
# ------------------------------------------------ #

"""
Extract epochs around the start of the tic. 
For the comparison we extract the random epochs, epochs with no tic. 
"""
import numpy as np
import mne

def extract_random_epochs_in_phase(raw_cropped, start_time, end_time, n_epochs=50, epoch_duration=2.5, event_id=1, seed=None):

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # calculate the max possible start times to avoid exceeding end_time
    max_start = end_time - epoch_duration
    if max_start <= start_time:
        raise ValueError("Interval too short for the requested epoch duration.")

    # randomly select the start times
    random_starts = np.random.uniform(low=start_time, high=max_start, size=n_epochs)

    # build an events array
    events = []
    for t in random_starts:
        sample_idx = np.round(t * raw_cropped.info['sfreq']).astype(int)
        events.append([sample_idx, 0, event_id])
    events = np.array(events, dtype=int)

    # create the epochs
    epochs = mne.Epochs(
        raw_cropped,
        events,
        event_id={f"random_{event_id}": event_id},
        tmin=0,
        tmax=epoch_duration,
        baseline=None,
        preload=True
    )

    return epochs
    
def extract_pre_tic_epochs(raw_cropped, urge_times, pre_seconds=2.0, post_seconds=0.5):

    events = []

    # convert urge_times → events array
    for t in urge_times:
        sample_idx = np.round(t * raw_cropped.info['sfreq']).astype(int)
        events.append([sample_idx, 0, 1])  # event_id=1 for urge

    events = np.array(events, dtype=int)

    # create epochs
    epochs = mne.Epochs(
        raw_cropped,
        events,
        event_id={"urge": 1},
        tmin=-pre_seconds,
        tmax=post_seconds,
        baseline=None,
        preload=True
    )

    return epochs