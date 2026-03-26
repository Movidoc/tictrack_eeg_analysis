# ---------------------------------------------- #
# Function: Time-frequency Analysis
# Author: Martyna
# ---------------------------------------------- #


import numpy as np
import mne 
import matplotlib.pyplot as plt

def trf_analysis_normalized(epochs, random_epochs, freqs=np.arange(1, 40, 1), epoch_type='pre_tic'):
    """
    Time frequency analysis with morlet technique. 
    Paramrters:
    frequency: 1-40Hz
    n_cycles: freqs/2
    normalization: z-score
    """
    
    n_cycles = freqs / 2.

    # -- 1. compute baseline mean/std from random epochs                
    power_random = random_epochs.compute_tfr(
        method="morlet",
        freqs=freqs,
        n_cycles=n_cycles,
        return_itc=False,
        average=False
    )

    # Crop to a stable window within the random epochs
    power_random_baseline = power_random.copy().crop(tmin=0.5, tmax=2.0)

    # Shape: (1, n_channels, n_freqs, 1) — broadcasts over any n_times 
    baseline_mean = power_random_baseline.data.mean(axis=(0, 3), keepdims=True) # check with the closed eyes phase as well 
    baseline_std  = power_random_baseline.data.std(axis=(0, 3), keepdims=True)
    baseline_std  = np.where(baseline_std == 0, 1e-10, baseline_std)

    # --- 2. compute TFR on epochs, normalize per epoch then average     
    power = epochs.compute_tfr(
        method="morlet",
        freqs=freqs,
        n_cycles=n_cycles,
        return_itc=False,
        average=False  # (n_epochs, n_channels, n_freqs, n_times)
    )

    # Z-score each epoch against the random baseline
    power_normalized = (power.data - baseline_mean) / baseline_std

    # Average across epochs
    power_avg = power_normalized.mean(axis=0)  # (n_channels, n_freqs, n_times)

    # --- 3. put data back into MNE object to keep metadata              
    power_mne = power.average()
    power_mne.data = power_avg

    return power_mne


def tfr_per_ROI_normalized(patient, pre_tic_epochs, random_epochs, epoch_type='pre_tic', freqs=np.arange(1.0, 40.0, 2.0), bad_channels=[], normalization = 'zscore'):
    """
    epoch_type : 'pre_tic' or 'random'
    Normalizes pre_tic epochs against random epochs baseline to avoid
    one strong epoch polluting the average.
    """
    channels_to_use_32 = ["Cz",  "Pz", "Fp1", "Fp2","C3","C4"] 
    channels_to_use_64 = ["Cz", "FCz",  "FC3", "CP3", "Pz", "Fp1", "AF3", "AF4", "FC4","Fp2","C3","C4","AFz","CPz","CP4"] 

    roi_lists_32 = {
        "midline_premotor":   ["Cz"],
        "left_sensorimotor":  ["C3"],
        "right_sensorimotor": ["C4"],
        "midline_posterior":  ["Pz"],
        "midline_prefrontal": ["Fp1", "Fp2"]
    }
    roi_lists_64 = {
        "midline_premotor":   ["Cz", "FCz"],
        "left_sensorimotor":  ["FC3", "CP3", "C3"],
        "right_sensorimotor": ["FC4", "C4", "CP4"],
        "midline_posterior":  ["Pz", "CPz"],
        "midline_prefrontal": ["Fp1", "AF3", "AF4", "Fp2", "AFz"]
    }

    # Choose ROI and channels based on montage
    if patient["montage"] == "standard_1020":
        channels_to_use = channels_to_use_32
        roi_lists = roi_lists_32
    elif patient["montage"] == "standard_1005":
        channels_to_use = channels_to_use_64
        roi_lists = roi_lists_64
    else:
        raise ValueError(f"Montage inconnu pour le patient: {patient['montage']}")


    # Pick relevant channels from both epoch sets
    epochs_roi        = pre_tic_epochs.copy().pick(channels_to_use)
    random_epochs_roi = random_epochs.copy().pick(channels_to_use)

    n_pre_tic_epochs = len(pre_tic_epochs)
    n_cycles = freqs / 2.0

    # compute TFR on random epochs and extract baseline mean/std  

    power_random = random_epochs_roi.compute_tfr(
        method="morlet",
        freqs=freqs,
        n_cycles=n_cycles,
        return_itc=False,
        average=False
    )
    if normalization == 'zscore' :
        # log-transform to stabilize variance, add small value to avoid log(0)
        power_random.data = 10 * np.log(power_random.data + 1e-12)  # log-transform to stabilize variance, add small value to avoid log(0)
    power_random_baseline = power_random.copy().crop(tmin=0.5, tmax=1.5) # check with the closed eyes phase as well

    # Average over epochs (axis=0) and time (axis=3)
    baseline_mean = power_random_baseline.data.mean(axis=(0, 3), keepdims=True)
    baseline_std  = power_random_baseline.data.std(axis=(0, 3), keepdims=True)

    # Avoid division by zero
    baseline_std = np.where(baseline_std == 0, 1e-10, baseline_std)

    # compute TFR on pre-tic epochs, normalize per epoch          
    power = epochs_roi.compute_tfr(
        method="morlet",
        freqs=freqs,
        n_cycles=n_cycles,
        return_itc=False,
        average=False
    )
    if normalization == 'zscore' :
        power.data = 10 * np.log(power.data + 1e-12)  # log-transform to stabilize variance, add small value to avoid log(0)
    power = power.crop(tmin=-2.0, tmax=1.5) #extend to see the entire tic time window 
    # Z-score normalize each epoch against the random baseline
    # (n_epochs, n_channels, n_freqs, n_times) - (1, n_channels, n_freqs, 1)
    power_normalized = (power.data - baseline_mean) / baseline_std
    # try other log ratio normalization 
    if normalization == 'logratio' :
        power_normalized = np.log((power.data + 1e-12) / (baseline_mean + 1e-12))
    if normalization == 'percent':
        power_normalized = 100 * (power.data - baseline_mean) / baseline_mean
    if normalization == 'substraction':
        power_normalized = power.data - baseline_mean
    if normalization == 'division':
        power_normalized = power.data / baseline_mean

    # Average across epochs → shape (n_channels, n_freqs, n_times)
    power_avg = power_normalized.mean(axis=0)

    # put normalized data back into MNE object to use .crop() etc 
    power_mne = power.average()   # AverageTFR with correct metadata
    power_mne.data = power_avg    # replace with normalized data

    # Print min and max power after normalization for debugging
    print(f"Power after normalization: min={power_mne.data.min():.2f}, max={power_mne.data.max():.2f}")

    # Crop to time window of interest
    # if epoch_type == 'pre_tic':
    #     power_mne = power_mne.crop(tmin=-1.0, tmax=0.0)
    # else:
    #     power_mne = power_mne.crop(tmin=0.6, tmax=2.0)

    # average over channels within each ROI                       
    roi_tfr = {}
    for roi_name, roi_channels in roi_lists.items():
        picks = mne.pick_channels(power_mne.ch_names, roi_channels, ordered=False)
        if len(picks) == 0:
            print(f"Warning: no valid channels for ROI '{roi_name}', skipping.")
            continue
        roi_power = power_mne.data[picks].mean(axis=0)
        roi_tfr[roi_name] = roi_power

    return roi_tfr, power_mne.freqs, power_mne.times, n_pre_tic_epochs



def plot_trf_roi(tfr_results, freqs, times, n, epoch_type='pre_tic', vmin=-2, vmax=2):

    # if vmin is None or vmax is None:

    #     all_data = np.concatenate([data.flatten() for data in tfr_results.values()])
    #     vmin, vmax = np.percentile(all_data, [1, 99])
    #     print(f"Auto vmin/vmax based on 1th/99th percentiles: vmin={vmin:.2f}, vmax={vmax:.2f}")


    roi_names = list(tfr_results.keys())
    n_rois = len(roi_names)

    # Create a 2x3 grid
    fig, axs = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    fig.suptitle(f"Pre-tic TRF (ROI-averaged) n = {n}",fontsize=18,fontweight="bold")

    for i, roi_name in enumerate(roi_names):
        row, col = divmod(i, 3)  # row = i//3, col = i%3
        ax = axs[row, col]

        data = tfr_results[roi_name]
        if epoch_type == 'pre_tic':
            extent = [-2, 1.5, 1, 40]
        else:
            extent = [0.5, 1,5, 1, 40]

        im = ax.imshow(
            data,
            aspect="auto",
            origin="lower",
            extent=extent,
            vmin=vmin,
            vmax=vmax,
            cmap="RdBu_r",
        )
        ax.set_title(f"{roi_name} – pre-tic TFR")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        if epoch_type == 'pre_tic':
            ax.axvline(0, color="k", linestyle="--", linewidth=0.6)
    
    # Remove the empty subplot if  fewer than 6 ROIs
    if n_rois < 6:
        for j in range(n_rois, 6):
            fig.delaxes(axs.flatten()[j])

    # Add a single colorbar for all plots
    cbar = fig.colorbar(im, ax=axs, orientation="vertical", fraction=0.02, pad=0.04)
    cbar.set_label("Power (log ratio)")

    return fig