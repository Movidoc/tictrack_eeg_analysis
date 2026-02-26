# =======================================================================
# File : _02_Movidoc_tictrack_prepro_TTL_extraction_patients.py
# Purpose : Preprocess the data from the EEG signals from the .vhdr files
# Author  : Indira
# =======================================================================



# ============================================================
# Libraries
# ============================================================
import numpy as np
import os
import mne
from autoreject import Ransac  
from autoreject.utils import interpolate_bads  
from mne.preprocessing import ICA
import autoreject
from autoreject import get_rejection_threshold
import matplotlib.pyplot as plt

# Qt backend is required for interactive MNE visualizations
from qtpy import QtWidgets

# To ensure that a Qt application exists --> necessary for MNE plots using PyQt
app = QtWidgets.QApplication.instance()
if app is None:
    app = QtWidgets.QApplication([])



# ============================================================
# Define functions
# ============================================================


# ==================================================================
# Function: save_figure
# Purpose : to save all Matplotlib figures that are not MNEQtBrowser
# ==================================================================

def save_figure(fig, filename, folder="C:\\Users\\indira.lavocat\\MOVIDOC\\tictrack_eeg_analysis\\Figures-Patients"):
    # create folder if it does not exist
    if not os.path.exists(folder):
        os.makedirs(folder)
    # build the output path
    fig_path = os.path.join(folder, filename)
    # save the figure to the path
    fig.savefig(fig_path)
    # confirm to user that it has been done
    print(f"Figure saved: {fig_path}")


# ====================================================================
# Function : load_data
# Purpose : to load BrainVision .vhdr files & display raw + PSD signal
# ====================================================================

def load_data(FilePath):
    # extract the filename only
    filename = os.path.basename(FilePath)
    # remove the fixed prefix & suffix to extract the subject identifier
    subject_name = filename.replace("MOVIDOCTicTrack", "").replace(".vhdr", "")
    # display the file that is being processed
    print(f"\n--- Traitement du fichier : {subject_name} ---\n")

    # load the BrainVision data in memory
    raw = mne.io.read_raw_brainvision(FilePath, preload=True)
    # display the raw signal
    # # raw.plot(show=True)
    # display the PSD (Power Spectral Density) of the signal
    # # raw.plot_psd(show=True)

    # plot the raw signal
    # n_eeg = len(mne.pick_types(raw.info, eeg=True))

    # if n_eeg <= 32:
    #     fig = raw.plot(start = 600, duration = 300, title="Raw EEG Signal", show=True, scalings = 'auto', n_channels = 10, picks=["Cz", "C3", "C4", "Pz", "Fp1", "Fp2"])
    # elif n_eeg <= 64:
    #     fig = raw.plot(start = 600, duration = 300, title="Raw EEG Signal", show=True, scalings = 'auto', n_channels = 10, picks=["Cz", "FCz", "C3", "FC3", "CP3", "C4", "CP4", "FC4", "Pz", "CPz", "Fp1", "Fp2", "AF3", "AFz", "AF4"])

    # fig = raw.plot( start = 100, duration = 300, title="Raw EEG Signal", show=True, scalings = 'auto', n_channels = 64)
    # fig.set_size_inches(30,18)
    # fig.suptitle("Figure 1 : Raw EEG Signal")
    # print the useful info
    raw.info
    print(raw.ch_names)
    print(raw.info['description'])
    return raw, subject_name


# ======================================================================================
# Function : extract_stimuli
# Purpose : to read & extract the event markers (stimuli) from annotations from the data
# ======================================================================================

def extract_stimuli(raw):
    # extract the events & their IDs from BrainVision annotations
    events, event_id = mne.events_from_annotations(raw)
    print("Events list :")
    print(event_id)
    print(events) # array: columns = [sample, 0, event_code]

    # remove the events that occur at sample index 0 (non-informative)
    events_no_zero = events[events[:, 0] != 0]
    # convert the event sample indices to time (in seconds)
    events_times_sec = events_no_zero[:, 0] / raw.info['sfreq']

    # reverse mapping {event_code → annotation_name}
    id_to_name = {v: k for k, v in event_id.items()}
    # print a readable list of the events with timestamps
    for time, eid in zip(events_times_sec, events_no_zero[:, 2]):
        name = id_to_name.get(eid, f"ID {eid}")
        #print(f"{name} à {time:.3f} s")

    return events_times_sec, event_id


# ===================================================================
# Function : preprocess_data
# Purpose : to define the montage, apply a band-pass & a Notch filter
# ===================================================================

def preprocess_data(raw, subject_name, montage_name):


    eeg_channels = mne.pick_types(raw.info, meg=False, eeg=True)
    #raw.plot(duration=60, n_channels=len(eeg_channels), title='Raw EEG with bad channels marked', show=True)

    # apply the montage defined by the user
    raw.set_montage(montage_name)
    # visualize the electrode placement
    #raw.plot_sensors(show_names=True, show=True)

    # something wrong with the last second from patient BB28
    if subject_name == '_BB28-bis':
        raw.crop(tmax=1646)

    '''
    # Plot raw data from eyes open and spontaneous phase 
    fig = raw.plot(duration=10,start = 210, n_channels=len(eeg_channels), title='Raw EEG - eyes open phase (10s)', show=True)
    fig.set_size_inches(30,18)
    fig.suptitle("Figure 1 : Raw EEG Signal - eyes open phase (10s)")

    fig = raw.plot(duration=10,start = 720, n_channels=len(eeg_channels), title='Raw EEG - spontaneous tics phase (10s)', show=True)
    fig.set_size_inches(30,18)
    fig.suptitle("Figure 1 : Raw EEG Signal - spontaneous tics phase (10s)")
    '''
    # apply band-pass filter between 0.5 and 100 Hz
    raw = raw.filter(l_freq=0.1, h_freq=45)
    # visualize the band-passed signal
    #raw.plot(title="High/Low Pass Filter", show=True)

    # apply Notch filter at 50 Hz to remove the powerline noise
    raw = raw.notch_filter(freqs=[50], picks="data", method="spectrum_fit")
    # visualize the notched signal
    # fig = raw.plot(start = 300, duration = 20, title="Filtered data", show=True, n_channels = 10)
    # fig.set_size_inches(30,18)
    # fig.suptitle("Figure 2 : EEG Signal after Band-pass & Notch Filtering")
    
    return raw

# ============================================================================
# Function : Ransac_bad_channel_detection
# Purpose : to apply the Ransac method for automatic detection of bad channels
# ============================================================================
def Ransac_bad_channel_detection(raw, subject_name):
    eeg_channels = mne.pick_types(raw.info, meg=False, eeg=True)

    # average reference for Ransac
    raw_avg = raw.copy().set_eeg_reference('average') 


    temporary_epochs = mne.make_fixed_length_epochs(raw_avg, duration=3.0, overlap=0.0, preload=True)

    # Plot butterfly to visually inspect channels
    temporary_epochs.average().plot(spatial_colors=True, titles=f'Butterfly plot - {subject_name}')
    plt.show(block = True)

    # Check the max amplitude per channel to identify potential outliers
    data, times = raw.get_data(return_times=True)
    # Find channel with max absolute amplitude
    max_amp = np.max(np.abs(data), axis=1)
    bad_idx = np.argmax(max_amp)
    print(f"Spiky channel: {raw.ch_names[bad_idx]}, max amplitude: {max_amp[bad_idx]*1e6:.1f} µV")

    ransac = Ransac(n_jobs=1, verbose = True)
    ransac.fit_transform(temporary_epochs)
    raw.info['bads'].extend(ransac.bad_chs_)
    bad_channels = raw.info['bads']
    print("Ransac detected bad channels:", ransac.bad_chs_)

        # add bad channels 
    # if subject_name == '000010':
    #     raw.info['bads'].append('Fp2')
    #     raw.info['bads'].append('Fp1')
 


    #plot the average signal across channels to visually inspect the effect of bad channels (after exclusion)
    print(f"Bad channels marked for {subject_name} : {raw.info['bads']}")
    temporary_epochs = mne.make_fixed_length_epochs(raw, duration=10.0, overlap=2.0, preload=True)
    temporary_epochs.average().plot(spatial_colors=True, titles=f'Butterfly plot (after exclusion) - {subject_name}')
    plt.show(block = True)
    # maximum amplitude per channel after exclusion
    data, times = raw.get_data(return_times=True)
    max_amp = np.max(np.abs(data), axis=1)
    for i, ch in enumerate(raw.ch_names):
        print(f"{ch}: {max_amp[i]*1e6:.1f} µV")

    '''
    # Plot after preprocessing
    fig = raw.plot(duration=10,start = 210, n_channels=len(eeg_channels), title='Preprocessed EEG - eyes open phase (10s)', show=True)
    fig.set_size_inches(30,18)
    fig.suptitle("Figure 2 : Preprocessed EEG Signal - eyes open phase (10s)")
    fig = raw.plot(duration=10,start = 720, n_channels=len(eeg_channels), title='Preprocessed EEG - spontaneous tics phase (10s)', show=True)
    fig.set_size_inches(30,18)
    fig.suptitle("Figure 2 : Preprocessed EEG Signal - spontaneous tics phase (10s)")
    '''
    return raw, bad_channels


'''
Global_Autoreject is too slow and too strict (almost all epochs are removed)

# ===============================================================================================================
# Function : global_Autoreject
# Purpose : to apply the Autoreject method for automatic detection & interpolation of bad channels and bad epochs, used for ICA fitting only 
# ===============================================================================================================

def global_Autoreject(raw, subject_name):
    # create fixed-length epochs for the autoreject fitting
    raw_crop = raw.copy().crop(tmin=200, tmax=800) # use only the first 5 minutes of the signal for autoreject fitting to save time
    events = mne.make_fixed_length_events(raw_crop, duration=3)
    epochs_temp = mne.Epochs(raw_crop, events, tmin=0.0, tmax=3, baseline=None, preload=True)
    epochs_temp.average().plot(titles = f'Butterfly plot before autoreject (global) - {subject_name}') # plot with time vs amplitude for each channel
    ar = autoreject.AutoReject(n_interpolate=[1, 2, 3, 4], random_state=11,
                           n_jobs=1, verbose=True)
    ar.fit(epochs_temp)
    epochs_ar, reject_log = ar.transform(epochs_temp, return_log=True)

    # visualize the dropped epochs
    epochs_temp[reject_log.bad_epochs].plot(scalings=dict(eeg=100e-6))
    reject_log.plot('horizontal')
    epochs_temp.average().plot(titles = f'Butterfly plot after autoreject (global) - {subject_name}') # plot with time vs amplitude for each channel
    epochs_temp.plot_drop_log() # bar plot showing which channels caused the most rejections
    return epochs_ar, reject_log
'''

# =================================================================
# Function : rejection_threshold 
# Purpose : to compute the rejection threshold for bad epochs based on the autoreject method, used for ICA fitting only
# =================================================================
def rejection_threshold(raw, subject_name):
    raw_avg = raw.copy().set_eeg_reference('average')
    events = mne.make_fixed_length_events(raw_avg, duration=2)
    epochs_temp = mne.Epochs(raw_avg, events, tmin=0.0, tmax=2, baseline=None, preload=True)
    reject = get_rejection_threshold(epochs_temp, decim=2)
    epochs_temp.drop_bad(reject=reject)
    epochs_temp.average().plot(titles = f'Butterfly plot after epoch exclusion (rejection_threshold) - {subject_name}') # plot with time vs amplitude for each channel 
    plt.show(block = True)
    epochs_temp.plot_drop_log() #  bar plot showing which channels caused the most rejections
    print(f"Rejection thresholds for ICA fitting: {reject}")
    print(f"Number of epochs after rejection: {len(epochs_temp)}")
    return reject, epochs_temp # outputs a dict with the rejection threshold for each channel type 

# ==================================================================================================================
# Function: std_threshold 
# Purpose: to compute the rejection threshold for bad epochs based on a simple standard deviation method (z-score)
# https://github.com/weiglszonja/meeg-tools/blob/master/tutorials/preprocessing_tutorial_with_triggers.ipynb
# =================================================================================================================
def rejection_threshold_std(raw, subject_name, threshold=3.0):
    """
    Computes rejection threshold for bad epochs based on the FASTER algorithm.
    """
    from mne.preprocessing.bads import _find_outliers

    raw_avg = raw.copy().set_eeg_reference('average')
    events = mne.make_fixed_length_events(raw_avg, duration=2)
    epochs_temp = mne.Epochs(raw_avg, events, tmin=0.0, tmax=2, baseline=None, preload=True)

    def _deviation(data):
        """Computes the deviation from mean for each channel."""
        channels_mean = np.mean(data, axis=2)
        return channels_mean - np.mean(channels_mean, axis=0)

    metrics = {
        "amplitude": lambda x: np.mean(np.ptp(x, axis=2), axis=1),
        "deviation": lambda x: np.mean(_deviation(x), axis=1),
        "variance":  lambda x: np.mean(np.var(x, axis=2), axis=1),
    }

    epochs_data = epochs_temp.get_data()
    bad_epochs = []

    print(f"\nFASTER epoch rejection for {subject_name}:")
    for metric_name, metric_func in metrics.items():
        scores = metric_func(epochs_data)
        outliers = _find_outliers(scores, threshold=threshold)
        print(f"  Bad epochs by {metric_name}: {outliers}")
        for idx in outliers:
            print(f"Epoch {idx} (t={idx*2:.1f}s): score={scores[idx]:.4f}")
        bad_epochs.extend(outliers)

    bad_epochs = list(set(bad_epochs))
    print(f"\n  Total bad epochs (union): {len(bad_epochs)} → {bad_epochs}")

    epochs_temp.drop(bad_epochs, reason='FASTER')
    print(f"  Remaining epochs: {len(epochs_temp)}")

    epochs_temp.average().plot(titles=f'Butterfly plot after FASTER rejection - {subject_name}')
    plt.show(block=True)
    epochs_temp.plot_drop_log()

    return epochs_temp

# ================================================================
# Fuction : apply_ICA
# Purpose : to apply ICA for artifact removal (eye blinks, muscle artifacts...)
# ===============================================================

def apply_ICA(epochs_ica, raw, subject_name):

    '''
    #Ddebugging : check the max value in the raw data to see if it's not an outlier that could cause problems for ICA (BB28)
    # Find when the max value occurs
    data = raw.get_data()
    ch_idx = raw.ch_names.index('T7')
    time_of_max = np.argmax(np.abs(data[ch_idx]))
    print(f"Max occurs at: {raw.times[time_of_max]:.1f} seconds")

    # Plot around that time
    raw.plot(start=raw.times[time_of_max]-5, duration=10, scalings='auto')

    print("Channel units:", raw.info['chs'][0]['unit'])
    print("Channel unit multiplier:", raw.info['chs'][0]['unit_mul'])
    data = raw.get_data()
    max_chan_idx = np.argmax(np.max(np.abs(data), axis=1))
    print("Channel with max value:", raw.ch_names[max_chan_idx])
    print("Its max value:", np.max(np.abs(data[max_chan_idx])))

    fig = raw.plot(duration=40, start=raw.times[time_of_max]-15, scalings='auto', 
        picks = ['Fp1', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'TP9', 'CP5'])
    fig.set_size_inches(30,18)
    fig.suptitle("Figure 6 : EEG Signal before ICA correction - example channels (10s)")

    # Check if it's one channel or all
    print("Per channel max values:")
    for i, ch in enumerate(raw.ch_names):
        print(f"{ch}: {np.max(np.abs(data[i])):.6f}")
    '''

    # Plot the raw data and check the scale
    
    # ------ Average reference before ICA fitting -----
    #raw_avg = epochs_temp.copy().set_eeg_reference('average') 
    


    # ------ Band-pass filter before ICA fitting -----
    filt_raw = epochs_ica.copy().filter(l_freq=1.0, h_freq = None)
    picks_eeg = mne.pick_types(filt_raw.info, meg=False, eeg=True, eog=False,
                                exclude='bads')
    data = filt_raw.get_data()

    # ------ ICA fitting -----
    #ica = ICA(n_components=15, method = 'picard', max_iter="auto", random_state=97)
    ica = mne.preprocessing.ICA(
    n_components=20,  method="picard", max_iter="auto", random_state=97)
    ica.fit(filt_raw, picks = picks_eeg) # reject=reject epochs that exceed the threshold are not considered in ICA fitting 
 
    ica.plot_components(show=True) # topomaps showing each ICA component and its spatial distribution
    plt.show(block = True)

    ica.exclude = []


    # ------------- Comparison to the baseline channel --------------
    # find which ICs match the EOG pattern
    # eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name="FT10")
    # ica.exclude = eog_indices
    #print(f"ICA components matching EOG (eye blinks) : {eog_indices}")
    # barplot of ICA component "EOG match" scores
    # ica.plot_scores(eog_scores)
    # plot ICs applied to raw data, with EOG matches highlighted
    #ica.plot_sources(raw)

    # -------------- Manual exclusion --------------
    #manually exclude [0,1] components for patient DS26
    if subject_name == '000010' : # DS26
        ica.exclude = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 15, 16, 17, 19]

    if subject_name == '_BB28-bis': # BB28
        ica.exclude = [0, 2, 7]

    if subject_name == '000013' : # BC29
        ica.exclude = [0,1 , 2, 4, 6, 8 ]
    
    if subject_name == '000030' : # MM30
        ica.exclude = [0]
    
    if subject_name == '000031' : # SC31
        ica.exclude = [0]

    # 10 second segments gives ~160 segments instead of 800
    ica.plot_properties(epochs_ica[:100], picks=range(20), psd_args={'fmax': 45}) # plot the properties of the ICA components to help identify which ones to exclude (topomap, time series, power spectrum)
    #ica.plot_properties(raw)
    ica.plot_sources(raw, picks = range(20) , show=True) # plot the time series of the ICA components 
    # plot the ICA components after correction to verify the effect of the artifact removal
    ica.plot_overlay(raw, exclude=ica.exclude) # plot the raw signal before and after ICA correction to visualize the effect of artifact removal (overlaid)

    # 4. Check ICA
    print("N components fitted:", ica.n_components_)
    print("Explained variance:", ica.pca_explained_variance_)

    raw = ica.apply(raw.copy())

    # plot the corrected signal
    eeg_channels = mne.pick_types(raw.info, meg=False, eeg=True)
    fig = raw.plot(start = 210, duration = 10, n_channels=len(eeg_channels), title="ICA corrected data", show=True)
    fig.set_size_inches(30,18)
    fig.suptitle("Figure 3 : EEG Signal after ICA correction - open eyes phase (10s)")

    fig = raw.plot(start = 720, duration = 10, n_channels=len(eeg_channels), title="ICA corrected data", show=True)
    fig.set_size_inches(30,18)
    fig.suptitle("Figure 3 : EEG Signal after ICA correction - spontaneous tics phase (10s)")
    return raw

# ==============================================================
# Function : local_Autoreject()
# Purpose : to apply the Autoreject method for automatic detection & interpolation of bad channels and bad epochs 
# ==============================================================

def local_Autoreject(epochs, subject_name):
    n_epochs = len(epochs)
    
    if n_epochs < 5:
        print(f"WARNING: Too few epochs ({n_epochs}) for autoreject, skipping")
        return epochs
    
    # Set cv based on number of epochs
    n_splits = min(10, n_epochs)
    ar = autoreject.AutoReject(n_interpolate=[1, 2, 3, 4, 5], random_state=11,
                           n_jobs=1, verbose=True, cv=n_splits)
    ar.fit(epochs)  
    epochs_clean, reject_log = ar.transform(epochs, return_log=True)
    # See which epochs were rejected
    reject_log.plot('horizontal')
    # plot the rejected epochs

    epochs[reject_log.bad_epochs].plot( title=f"Rejected epochs - {subject_name}")

    # See the cleaned epochs
    epochs_clean.plot(show=True)

   
    return epochs_clean


# ==============================================================
# Function : apply_rest_reference
# Purpose : to re-reference the EEG signal using the REST method
# ==============================================================

def apply_rest_reference(raw, subject_name):
    print("Applying REST reference...")
    
    # build an anatomical sphere model for the REST computation
    Sphere = mne.make_sphere_model('auto', 'auto', raw.info)
    # create a volume source space inside this sphere
    Source = mne.setup_volume_source_space(sphere=Sphere, exclude=30.0, pos=5.0, verbose=False)
    # compute the forward model for referencing
    Forward = mne.make_forward_solution(raw.info, trans=None, src=Source, bem=Sphere, verbose=False)

    # apply the REST referencing
    raw_rest = raw.copy().set_eeg_reference('REST', forward=Forward)

    # # visualize the re-referenced signal
    # # visualize the PSD of the re-referenced signal
    # raw_rest.plot_psd(fmin=0, fmax=100, show=True)

    return raw_rest


# =========================================================================================================
# Function : recalibrate_from_first_event
# Purpose : to do the recalage by cropping the signal from the Stimulus/S  2 = instructions press_key phase
# =========================================================================================================

def recalibrate_from_first_event(raw, target_stim="Stimulus/S  2"):

    # annotations = raw.annotations

    # Verify if the annotations exist
    if raw.annotations is None or len(raw.annotations) == 0:
        print("No annotations found — cannot realign to target stimulus.")
        return raw
    
    # search the 1st occurrence of the target stimulus
    target_onsets = [onset for onset, desc in zip(raw.annotations.onset, raw.annotations.description) if desc == target_stim]

    # check if the list of event times is empty, if no events found → cannot crop
    if not target_onsets:
        print(f"Target stimulus '{target_stim}' not found — cannot realign.")
        return raw
    # if len(target_onsets) == 0:
    #     print(f"Target stimulus '{target_stim}' not found — cannot realign.")
    #     return raw
    
    # recalage timestamp = Stimulus/S  2
    first_stimulus_time = target_onsets[0]
    print(f"Cropping EEG at stimulus '{target_stim}' = {first_stimulus_time:.3f} s")

    # crop the signal from this timestamp
    raw_cropped = raw.copy().crop(tmin=first_stimulus_time)

    # filter the annotations only after the crop
    mask_valid = raw_cropped.annotations.onset >= first_stimulus_time
    new_onsets=raw_cropped.annotations.onset[mask_valid] - first_stimulus_time
    new_durations = raw_cropped.annotations.duration[mask_valid]
    new_descriptions = [d for d, v in zip (raw_cropped.annotations.description, mask_valid) if v]

    # create new readjusted annotations
    # cropped_annotations = mne.Annotations(
    # onset=raw_cropped.annotations.onset[mask_valid] - first_stimulus_time,
    # duration=raw_cropped.annotations.duration[mask_valid],
    # description=[d for d, v in zip (raw_cropped.annotations.description, mask_valid) if v]
    # )

    # create a new readjusted annotations object
    raw_cropped.set_annotations(mne.Annotations(onset=new_onsets, duration=new_durations, description=new_descriptions))

    # re-align the annotations relative to the new t=0
    # raw_cropped.set_annotations(cropped_annotations)

    print("DEBUG after set_annotations:", raw_cropped.annotations.onset[:5])

    # visualize the cropped signal
    #raw_cropped.plot(title="Signal cropped", show=True)
    return raw_cropped


# ======================================================================================================
# Function : extract_phases
# Purpose : to create the epochs for the specific experimental phases (spontaneous, imitated, retention)
# ======================================================================================================

def extract_phases(raw_cropped, subject_name, save_folder="C:\\Users\\indira.lavocat\\MOVIDOC\\tictrack_eeg_analysis\\EEG_Phases"):
    
    # define the TTL names for each phase of the experiment
    phases_dict = {
        "spontaneous_tics": {"start": "Stimulus/S  9", "end": "Stimulus/S 10"},
        "imitated_tics": {"start": "Stimulus/S 11", "end": "Stimulus/S 12"},
        "retention_tics": {"start": "Stimulus/S 13", "end": "Stimulus/S 14"}
    }

    # create destination folder if needed
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # iterate through each phase
    for phase_name, phase_TTL in phases_dict.items():
        try:
            annotations = raw_cropped.annotations
            # extract timestamps for the start & end TTL of the phase
            start_times = [ann['onset'] for ann, desc in zip(annotations, annotations.description) if desc == phase_TTL['start']]
            end_times = [ann['onset'] for ann, desc in zip(annotations, annotations.description) if desc == phase_TTL['end']]

            # if phase markers are missing → skip
            if len(start_times) == 0 or len(end_times) == 0:
                print(f"Phase {phase_name} not found for {subject_name}, missing TTL")
                continue

            # use the 1st occurrence of the start & end TTL of the phase
            start_time = start_times[0]
            end_time = min(end_times[0], raw_cropped.times[-1])

            print(f"{phase_name}: {start_time:.2f}s → {end_time:.2f}s")

            # crop the signal to isolate the phase
            phase_raw = raw_cropped.copy().crop(tmin=start_time, tmax=end_time)

            # build the output filename
            phase_file_path = os.path.join(save_folder, f"{subject_name}_{phase_name}_raw.fif")
            # save the cropped phase as a .fif file
            phase_raw.save(phase_file_path, overwrite=True)
            print(f"Saved phase : {phase_file_path}")

        except Exception as e:
            # catch & report any unexpected error
            print(f"Error while cutting the phase {phase_name} for {subject_name} : {e}")
            continue


# ===================================================================================
# Function : collect_ttl_with_phases
# Purpose : collect all TTLs after recalage & assign them to the experimental phases.
# ===================================================================================

def collect_ttl_with_phases(raw_cropped, subject_name):

    ttl_list = []

    # 1. Extract events and their IDs
    events, event_id = mne.events_from_annotations(raw_cropped)
    id_to_name = {v: k for k, v in event_id.items()}

    # convert sample index → time in seconds
    event_times_sec = events[:, 0] / raw_cropped.info['sfreq']

    # 2. Define the phases TTL mapping
    phases_dict = {
        "press_key": {"start": "Stimulus/S  3", "end": "Stimulus/S  4"},
        "eyes_closed": {"start": "Stimulus/S  5", "end": "Stimulus/S  6"},
        "eyes_open": {"start": "Stimulus/S  7", "end": "Stimulus/S  8"},
        "spontaneous_tics": {"start": "Stimulus/S  9", "end": "Stimulus/S 10"},
        "imitated_tics": {"start": "Stimulus/S 11", "end": "Stimulus/S 12"},
        "retention_tics": {"start": "Stimulus/S 13", "end": "Stimulus/S 14"}
    }

    # 3. Build time intervals for each phase
    phase_intervals = {}
    annotations = raw_cropped.annotations

    for phase_name, t in phases_dict.items():
        start = [ann['onset'] for ann, desc in zip(annotations, annotations.description) if desc == t["start"]]
        end   = [ann['onset'] for ann, desc in zip(annotations, annotations.description) if desc == t["end"]]

        if len(start) > 0 and len(end) > 0:
            phase_intervals[phase_name] = (start[0], end[0])
        else:
            phase_intervals[phase_name] = None  # Missing TTL → phase ignored

    # 4. Associate each TTL with its phase (or None)
    for time, eid in zip(event_times_sec, events[:, 2]):
        ttl_name = id_to_name[eid]
        ttl_phase = None

        for phase_name, interval in phase_intervals.items():
            if interval is None:
                continue
            start_t, end_t = interval
            if start_t <= time <= end_t:
                ttl_phase = phase_name
                break

        ttl_list.append({
            "ttl_name": ttl_name,
            "time": float(time),
            "phase": ttl_phase
        })

    print(f"Collected {len(ttl_list)} TTLs for {subject_name}.")
    return ttl_list


#########################################################################



# === code with functions ===

# List of the .vhdr files to load
# vhdr_files = [
#     "C:\\Users\\indira.lavocat\\MOVIDOC\\EEG\\EEG PATIENT FILES\\MOVIDOCTicTrack000010.vhdr", #DS26
#     "C:\\Users\\indira.lavocat\\MOVIDOC\\EEG\\EEG PATIENT FILES\\MOVIDOCTicTrack_BB28-bis.vhdr", #BB28
#     "C:\\Users\\indira.lavocat\\MOVIDOC\\EEG\\EEG PATIENT FILES\\MOVIDOCTicTrack000013.vhdr", #BC29
#     "C:\\Users\\indira.lavocat\\MOVIDOC\\EEG\\EEG PATIENT FILES\\MOVIDOCTicTrack000030.vhdr", #MM30
#     "C:\\Users\\indira.lavocat\\MOVIDOC\\EEG\\EEG PATIENT FILES\\MOVIDOCTicTrack000031.vhdr" #SC31
# ]

# vhdr_files = ["C:\\Users\\indira.lavocat\\MOVIDOC\\EEG\\EEG PATIENT FILES\\MOVIDOCTicTrack000010.vhdr"] #DS26


# if __name__ == "__main__":
#     for FilePath in vhdr_files:
#         try:
#             # 1️⃣ Load the data
#             raw, subject_name = load_data(FilePath)

#             # 2️⃣ Extract the stimuli
#             events_times_sec, event_id = extract_stimuli(raw)

#             # 3️⃣ Set the montage & Filter
#             raw_preprocessed = preprocess_data(raw, subject_name)

#             # 4️⃣ REST re-reference
#             raw_rest = apply_rest_reference(raw_preprocessed, subject_name)

#             # 5️⃣ Recalage from the 1st event (not null)
#             raw_cropped = recalibrate_from_first_event(raw_rest, events_times_sec) # raw_cropped = Readjusted_Signal_Figure_5
#             print("\n--- Annotations after the recalage ---")
#             # events_times_sec_cropped, event_id_cropped = extract_stimuli(raw_cropped)
#             ttl_info = collect_ttl_with_phases(raw_cropped, subject_name)
#             Readjusted_Signal_Figure_5 = raw_cropped.plot(
#                 title="Readjusted signal (from the 1st stimulus not at 0 s)",
#                 show=True
#             )

#             # 6️⃣ Cut & save the phases
#             extract_phases(raw_cropped, subject_name)

#         except Exception as e:
#             print(f"Erreur pour {FilePath} : {e}")
#             continue

#########################################################################