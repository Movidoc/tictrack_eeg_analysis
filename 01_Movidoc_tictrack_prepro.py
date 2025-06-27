# > Import the required libraries
# Libraries
import os
import mne
from mne import EpochsArray, create_info
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# > To save the figures that are not MNEQtBrowser
# Define a function to save all the Figures that are NOT MNEQtBrowser
def save_figure(fig, filename, folder="C:\\Users\\indira.lavocat\\MOVIDOC\\tictrack_eeg_analysis\\Figures"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    fig_path = os.path.join(folder, filename)
    fig.savefig(fig_path)
    print(f"Figure saved: {fig_path}")



# A. Data

# 1. Load the data

# Define the path to the .vhdr file
FolderPath = "C:\\Users\\indira.lavocat\\MOVIDOC\\EEG\\Sujets\\IndiraLAVOCAT" # need to adapt the last folder to suit the subject
# Looking for the .vhdr file in the folder
for file in os.listdir(FolderPath):
    if file.endswith(".vhdr"):
        FilePath = os.path.join(FolderPath, file)
        break
print(FilePath)

raw = mne.io.read_raw_brainvision(FilePath, preload=True)
raw.info
print(raw.ch_names)
print(raw.info['description']) # gives a note about the channels when there is one


# 2. Extract the stimuli

# Create an events dicionnary
events, event_id = mne.events_from_annotations(raw)
print("Events list (stimulus) :")
print(event_id)

# Display the tab of events
print("Events (sample, previous_id, event_id) :")
print(events)

# Convert the timestamps into seconds
events_no_zero = events[events[:, 0] != 0]  # <-- filters the events at 0.000 s
events_times_sec = events_no_zero[:, 0] / raw.info['sfreq'] # converts the timestamps into seconds
for time, eid in zip(events_times_sec, events_no_zero[:, 2]): # links each time in seconds to its event ID
    print(f"Stimulus {eid} à {time:.3f} s") # formats the number with 3 decimal

# > Alternative to display the stimulus name (and not its ID) with its timestamp in seconds
id_to_name = {v: k for k, v in event_id.items()}
for time, eid in zip(events_times_sec, events_no_zero[:, 2]):
    name = id_to_name.get(eid, f"ID {eid}")
    print(f"{name} à {time:.3f} s")



# B. Preprocessing

# 1. Define the montage

raw.set_montage("standard_1020") # to adapt according to the montage used during the exepriments

Sensors_Montage_Figure_1 = raw.plot_sensors(show_names=True)
save_figure(Sensors_Montage_Figure_1, "Figure1_SensorsMontage.png")
# fig.savefig("C:\\Users\\indira.lavocat\\MOVIDOC\\tictrack_eeg_analysis\\Figures\\Figure1_SensorsMontage.png")


# 2. Filter the data with a band-pass

# Define the high and low frequencies
HFreq = 30
LFreq = 1
raw_HighLowPassed = raw.filter(l_freq = LFreq, h_freq = HFreq)
# for ERPs, [1-30] Hz band-pass filter

# Plot the highpassed signal
Signal_HighLowPassed_Figure_2 = raw_HighLowPassed.plot(title = "High- and Low- passed Signal")


# 3. Filter the data with a Notch

# Define the parameters for the notch filter
if HFreq < 50:
    raw_Notched = raw_HighLowPassed
else:
    raw_Notched = raw_HighLowPassed.notch_filter(freqs = [50], picks = "data", method = "spectrum_fit")

# Plot the notched signal
Signal_Notched_Figure_3 = raw_Notched.plot(title = "Notched Signal")


# 4. Identify the bad channels


# 5. Re-reference the data

# REST method : advanced EEG re-referencemen
print("Application du référentiel REST...")

# Create a spherical model of the head based on the file information 
Sphere = mne.make_sphere_model('auto', 'auto', raw_Notched.info)

# Define the volume source space
Source = mne.setup_volume_source_space(sphere=Sphere, exclude=30.0, pos=5.0, mri=None, verbose=False)

# Calculate the forward model solution
Forward = mne.make_forward_solution(raw_Notched.info, trans=None, src=Source, bem=Sphere, verbose=False)

# Apply the REST reference
raw_REST = raw_Notched.copy().set_eeg_reference('REST', forward=Forward)

# Optionnal : visualisation after REST
Signal_REST_Figure_4 = raw_REST.plot(title="Signal après référence REST")
save_figure(Signal_REST_Figure_4, "Figure4_Signal_REST.png")


# 7. Recalage

# Reset the file time
if events_times_sec[0] == 0 and len(events_times_sec) > 1: # check if the 1st stimulus is at 0 s. If so, use the 2nd stimulus
    first_stimulus_time = events_times_sec[1]
    print(f"First stimulus is at 0. Using second stimulus at {first_stimulus_time:.3f} s")
else:
    first_stimulus_time = events_times_sec[0]
    print(f"First stimulus at {first_stimulus_time:.3f} s")

# Truncate the signal to start at this point
raw_cropped = raw_REST.copy().crop(tmin=first_stimulus_time)

# Reset the annotations by shifting all annotations by - first_stimulus_time
if raw.annotations is not None:
    raw_annotation_times = raw.annotations.onset - first_stimulus_time
    raw_cropped.set_annotations(
        mne.Annotations(
            onset=raw_annotation_times, # onset = raw_annotation_times with the new reset times
            duration=raw.annotations.duration,
            description=raw.annotations.description
        )
    )

# Plot truncated and recalculated data
Readjusted_Signal_Figure_5 = raw_cropped.plot(title="Readjusted signal (from the 1st stimulus not at 0 s)")



# C. Epoching

# 1. Phase 1 - Get the "press a key" baseline from the P1 phase

# > Get the stimuli in P1
# Define the parameters
begin_P1_stimulus = "Stimulus/S  3" # sent at the beginning of the P1 task
end_P1_stimulus = "Stimulus/S  4" # sent at the ending of the P1 task 

# Create a list with annotations and their times
list_annotations = list(zip(raw_cropped.annotations.onset, raw_cropped.annotations.description))

# Find all the segments in the P1 phase
P1_segments = []
i = 0
while i < len(list_annotations):
    onset, desc = list_annotations[i]
    if desc == begin_P1_stimulus:
        # Search the next end event
        for j in range(i + 1, len(list_annotations)):
            next_onset, next_desc = list_annotations[j]
            if next_desc == end_P1_stimulus:
                P1_segments.append((onset, next_onset))
                i = j  # continue after the stop
                break
    i += 1

# Get all the stimuli present in the segments
stimuli_in_P1 = []
for start, end in P1_segments:
    for onset, desc in list_annotations:
        if start <= onset <= end:
            # Do not take into account the beginning and ending stimuli
            if desc not in (begin_P1_stimulus, end_P1_stimulus):
                stimuli_in_P1.append((onset, desc))

# Display & check the stimuli found
print(f"{len(stimuli_in_P1)} stimuli trouvés entre '{begin_P1_stimulus}' et '{end_P1_stimulus}' :")
for onset, desc in stimuli_in_P1:
    print(f"{desc} à {onset:.3f} s")

# > Get the the signal -1 second before & +1 second after each stimulus
# Define the window around each stimulus
tmin = -2.0 # 2 secondes before
tmax = 2.0 # 2 secondes after

# Get the sampling frequency
sfreq = raw_cropped.info['sfreq']

# Initiate a list to stock the extracted values (expected shape per segment : n_channels x n_times)
P1_signal_segments = []

for onset, desc in stimuli_in_P1:
    start = onset + tmin
    end = onset + tmax
    # Check if the window is not outside the signal bounds
    if start < 0 or end > raw_cropped.times[-1]:
        print(f"⚠️ Stimulus at {onset:.2f}s ignored (window [{start:.2f}, {end:.2f}] out of limits)")
        continue
    # Get the segment
    P1_segment = raw_cropped.copy().crop(tmin=start, tmax=end).get_data() # shape: (n_channels, n_times)
    P1_signal_segments.append(P1_segment)

# Convert into an array numpy : shape = (n_events, n_channels, n_times)
P1_segments_array = np.array(P1_signal_segments)

# > Create epochs for each segment
# Create an info object to build an EpochsArray object
info = create_info(
    ch_names=raw_cropped.ch_names,
    sfreq=sfreq,
    ch_types="eeg"  # to adapt according to the sensors nature
)

# Create the EpochsArray object from P1_segments_array
epochs_P1 = EpochsArray(P1_segments_array, info) # expected shape : (n_epochs, n_channels, n_times)

# > Save the epochs into a file
# Save into a .fif file
save_P1_path = "C:\\Users\\indira.lavocat\\MOVIDOC\\tictrack_eeg_analysis\\.fif_files\\P1_epochs.fif"
epochs_P1.save(save_P1_path, overwrite=True)
print(f"✅ Segments sauvegardés dans : {save_P1_path}")

# > Display the mean segment
# Calculer la moyenne sur tous les événements : shape = (n_channels, n_times)
mean_segment = np.mean(P1_segments_array, axis=0)

# > OPTIONNAL : Display the result shape
print(f"\n✅ {len(P1_signal_segments)} valide segments used.")
print(f"Shape of the mean segment : {mean_segment.shape} (n_channels, n_times)")

# Create the temporal axe for the mean segment
n_times = mean_segment.shape[1]
times = np.linspace(tmin, tmax, n_times)

# Trace each channel in a seperated figure
for ch_idx, ch_name in enumerate(raw_cropped.ch_names):
    signal = mean_segment[ch_idx, :]
    
    plt.figure(figsize=(8, 4))
    plt.plot(times, signal, label=f'Channel : {ch_name}')
    plt.title(f"Mean segment – Channel {ch_name}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (µV)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# 2. Phase 2 (P2) - Get the "eyes closed" baseline from the P2a phase

# Define the parameters
begin_P2a_stimulus = "Stimulus/S  5" # sent at the beginning of the P2a task
end_P2a_stimulus = "Stimulus/S  6" # sent at the ending of the P2a task

# Search the stimuli times in the annotations
onset_start = None
onset_end = None

for onset, desc in zip(raw_cropped.annotations.onset, raw_cropped.annotations.description):
    if desc == begin_P2a_stimulus and onset_start is None:
        onset_start = onset
    elif desc == end_P2a_stimulus and onset_start is not None:
        onset_end = onset
        break # stopping the loop as soon as the pair of stimuli is found

# Check the segment found
if onset_start is not None and onset_end is not None:
    print(f"Segment detected : from {onset_start:.2f} s to {onset_end:.2f} s")

    # Get the signal segment
    P2a_segment = raw_cropped.copy().crop(tmin=onset_start, tmax=onset_end)
    data, times = P2a_segment.get_data(return_times=True) # data shape: (n_channels, n_times)

    # Calculate the mean absolute value (over the time axis)
    mean_abs_values = np.mean(np.abs(data), axis=1) # shape: (n_channels,)
    
    # Display the mean absolute value for each channel
    for ch_name, mean_val in zip(raw_cropped.ch_names, mean_abs_values):
        # print(f"{ch_name} : mean absolute value = {mean_val:.3f} µV")
        print(f"{ch_name} : mean absolute value = {mean_val:.8f} µV")

else:
    print("❌ Stimuli not found in the annotations.")

### 8.1.c. Get the "eyes open" baseline from the P2b phase

# Define the parameters
begin_P2b_stimulus = "Stimulus/S  7" # sent at the beginning of the P2a task
end_P2b_stimulus = "Stimulus/S  8" # sent at the ending of the P2a task

# Search the stimuli times in the annotations
onset_start = None
onset_end = None

for onset, desc in zip(raw_cropped.annotations.onset, raw_cropped.annotations.description):
    if desc == begin_P2b_stimulus and onset_start is None:
        onset_start = onset
    elif desc == end_P2b_stimulus and onset_start is not None:
        onset_end = onset
        break # stopping the loop as soon as the pair of stimuli is found

# Check the segment found
if onset_start is not None and onset_end is not None:
    print(f"Segment detected : from {onset_start:.2f} s to {onset_end:.2f} s")

    # Get the signal segment
    P2b_segment = raw_cropped.copy().crop(tmin=onset_start, tmax=onset_end)
    data, times = P2b_segment.get_data(return_times=True) # data shape: (n_channels, n_times)

    # Calculate the mean absolute value (over the time axis)
    mean_abs_values = np.mean(np.abs(data), axis=1) # shape: (n_channels,)
    
    # Display the mean absolute value for each channel
    for ch_name, mean_val in zip(raw_cropped.ch_names, mean_abs_values):
        # print(f"{ch_name} : mean absolute value = {mean_val:.3f} µV")
        print(f"{ch_name} : mean absolute value = {mean_val:.8f} µV")

else:
    print("❌ Stimuli not found in the annotations.")



###



# 9. Seperation in epochs
# tmin = -0.2  # 200 ms before the event
# tmax = 0.8   # 800 ms after the event
# epochs = mne.Epochs(raw, events, event_id=event_id,
#                     tmin=tmin, tmax=tmax, baseline=(None, 0),
#                     preload=True)
# epochs.plot_drop_log()

# 10. Define an automatic reject of the artifacts (optional)
# epochs.plot_drop_log()
# epochs.drop_bad()

# 11. Averaging (ERP)
# evoked = epochs.average()
# evoked.plot() # evoked does not accept any "title"
# fig = evoked.plot_image(picks='eeg')
# fig.suptitle("ERP (moyenne des epochs)")

# 12. Topography
# evoked.plot_topomap(times=[0.1, 0.2, 0.3], ch_type='eeg')

# plt.show()



# 10.1.a Save the analysis in .fif (average epochs)
# evoked.save("MOVIDOC_TicTrack_000002_erp-ave.fif") # saves the average epochs
# => To re-load it :
# from mne import read_evokeds
# evoked = read_evokeds("MOVIDOC_TicTrack_000002_erp-ave.fif", condition=0)

# 10.1.b Save the analysis in .fif (complete epochs)
# epochs.save("MOVIDOC_TicTrack_000002_epochs.fif", overwrite=True) # saves the complete epochs
# To re-load it :
# from mne import read_epochs
# epochs = read_epochs("MOVIDOC_TicTrack_000002_epochs.fif")

# 10.2 Save the analysis in .csv
# import pandas as pd
# => Convert the ERP (evoked) into a DataFrame
# df = pd.DataFrame(evoked.data.T, columns=evoked.ch_names)
# df.insert(0, "time", evoked.times)  # Ajoute la colonne temps en 1ère position
# => Save as a .csv
# df.to_csv("MOVIDOC_TicTrack_000002_erp.csv", index=False)