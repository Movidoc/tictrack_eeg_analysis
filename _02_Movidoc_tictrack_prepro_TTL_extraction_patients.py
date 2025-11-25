# =======================================================================
# File : _02_Movidoc_tictrack_prepro_TTL_extraction_patients.py
# Purpose : Preprocess the data from the EEG signals from the .vhdr files
# Author  : Indira
# =======================================================================



# ============================================================
# Libraries
# ============================================================

import os
import mne

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
    raw.plot(show=True)
    # display the PSD (Power Spectral Density) of the signal
    raw.plot_psd(show=True)

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
        print(f"{name} à {time:.3f} s")

    return events_times_sec, event_id


# ===================================================================
# Function : preprocess_data
# Purpose : to define the montage, apply a band-pass & a Notch filter
# ===================================================================

def preprocess_data(raw, subject_name, montage_name):
    # apply the montage defined by the user
    raw.set_montage(montage_name)
    # visualize the electrode placement
    raw.plot_sensors(show_names=True, show=True)

    # apply band-pass filter between 0.5 and 100 Hz
    raw = raw.filter(l_freq=0.5, h_freq=100)
    # visualize the band-passed signal
    raw.plot(title="High/Low Pass Filter", show=True)

    # apply Notch filter at 50 Hz to remove the powerline noise
    raw = raw.notch_filter(freqs=[50], picks="data", method="spectrum_fit")
    # visualize the notched signal
    raw.plot(title="Notch Filter", show=True)

    return raw


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

    # visualize the re-referenced signal
    raw_rest.plot(title="Signal REST", show=True)
    # visualize the PSD of the re-referenced signal
    raw_rest.plot_psd(fmin=0, fmax=100, show=True)

    return raw_rest


# =========================================================================================================
# Function : recalibrate_from_first_event
# Purpose : to do the recalage by cropping the signal from the Stimulus/S  2 = instructions press_key phase
# =========================================================================================================

def recalibrate_from_first_event(raw, target_stim="Stimulus/S  2"):

    # Verify if the annotations exist
    if raw.annotations is None or len(raw.annotations) == 0:
        print("No annotations found — cannot realign to target stimulus.")
        return raw
    
    # search the 1st occurrence of the target stimulus
    target_onsets = [
        onset for onset, desc in zip(raw.annotations.onset, raw.annotations.description)
        if desc == target_stim
    ]

    # check if the list of event times is empty, if no events found → cannot crop
    if len(target_onsets) == 0:
        print(f"Target stimulus '{target_stim}' not found — cannot realign.")
        return raw
    
    # recalage timestamp = Stimulus/S  2
    first_stimulus_time = target_onsets[0]
    print(f"Cropping EEG at stimulus '{target_stim}' = {first_stimulus_time:.3f} s")

    # crop the signal from this timestamp
    raw_cropped = raw.copy().crop(tmin=first_stimulus_time)

    # re-align the annotations relative to the new t=0
    if raw.annotations is not None:
        # shift all the annotation onsets backward by the cropping time
        new_onsets = raw.annotations.onset - first_stimulus_time
        # determine which annotations occur AFTER cropping
        mask_valid = new_onsets >= 0
        # replace the annotations in the cropped raw object
        raw_cropped.set_annotations(mne.Annotations(
            onset=new_onsets[mask_valid], # shifted onsets
            duration=raw.annotations.duration[mask_valid], # original durations
            description=[d for d, v in zip(raw.annotations.description, mask_valid) if v] # descriptions
        ))

    # visualize the cropped signal
    raw_cropped.plot(title="Signal cropped", show=True)
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



# List of the .vhdr files to load
vhdr_files = [
    "C:\\Users\\indira.lavocat\\MOVIDOC\\EEG\\EEG PATIENT FILES\\MOVIDOCTicTrack000010.vhdr", #DS26
    "C:\\Users\\indira.lavocat\\MOVIDOC\\EEG\\EEG PATIENT FILES\\MOVIDOCTicTrack_BB28-bis.vhdr", #BB28
    "C:\\Users\\indira.lavocat\\MOVIDOC\\EEG\\EEG PATIENT FILES\\MOVIDOCTicTrack000013.vhdr", #BC29
    "C:\\Users\\indira.lavocat\\MOVIDOC\\EEG\\EEG PATIENT FILES\\MOVIDOCTicTrack000030.vhdr", #MM30
    "C:\\Users\\indira.lavocat\\MOVIDOC\\EEG\\EEG PATIENT FILES\\MOVIDOCTicTrack000031.vhdr" #SC31
]

# vhdr_files = ["C:\\Users\\indira.lavocat\\MOVIDOC\\EEG\\EEG PATIENT FILES\\MOVIDOCTicTrack000010.vhdr"] #DS26

#########################################################################



if __name__ == "__main__":
    for FilePath in vhdr_files:
        try:
            # 1️⃣ Load the data
            raw, subject_name = load_data(FilePath)

            # 2️⃣ Extract the stimuli
            events_times_sec, event_id = extract_stimuli(raw)

            # 3️⃣ Set the montage & Filter
            raw_preprocessed = preprocess_data(raw, subject_name)

            # 4️⃣ REST re-reference
            raw_rest = apply_rest_reference(raw_preprocessed, subject_name)

            # 5️⃣ Recalage from the 1st event (not null)
            raw_cropped = recalibrate_from_first_event(raw_rest, events_times_sec) # raw_cropped = Readjusted_Signal_Figure_5
            print("\n--- Annotations after the recalage ---")
            # events_times_sec_cropped, event_id_cropped = extract_stimuli(raw_cropped)
            ttl_info = collect_ttl_with_phases(raw_cropped, subject_name)
            Readjusted_Signal_Figure_5 = raw_cropped.plot(
                title="Readjusted signal (from the 1st stimulus not at 0 s)",
                show=True
            )

            # 6️⃣ Cut & save the phases
            extract_phases(raw_cropped, subject_name)

        except Exception as e:
            print(f"Erreur pour {FilePath} : {e}")
            continue

#########################################################################



# ============================================================
# A. Data
# ============================================================


# ================
# 1. Load the data
# ================

# for FilePath in vhdr_files:
#     try:
#         filename = os.path.basename(FilePath)
#         subject_name = filename.replace("MOVIDOCTicTrack", "").replace(".vhdr", "")
#         print(f"\n--- Treatment of the file : {subject_name} ---\n")

#         raw = mne.io.read_raw_brainvision(FilePath, preload=True)
#         raw.plot(show=True)
#         raw.plot_psd(show=True)
#         raw.info
#         print(raw.ch_names)
#         print(raw.info['description']) # gives a note about the channels when there is one


#         # ======================
#         # 2. Extract the stimuli
#         # ======================

#         # Create an events dicionnary
#         events, event_id = mne.events_from_annotations(raw)
#         print("Events list (stimulus) :")
#         print(event_id)

#         # Display the tab of events
#         print("Events (sample, previous_id, event_id) :")
#         print(events)

#         # Convert the timestamps into seconds
#         events_no_zero = events[events[:, 0] != 0]  # <-- filters the events at 0.000 s
#         events_times_sec = events_no_zero[:, 0] / raw.info['sfreq'] # converts the timestamps into seconds
#         for time, eid in zip(events_times_sec, events_no_zero[:, 2]): # links each time in seconds to its event ID
#             print(f"Stimulus {eid} à {time:.3f} s") # formats the number with 3 decimal

#         # > Alternative to display the stimulus name (and not its ID) with its timestamp in seconds
#         id_to_name = {v: k for k, v in event_id.items()}
#         for time, eid in zip(events_times_sec, events_no_zero[:, 2]):
#             name = id_to_name.get(eid, f"ID {eid}")
#             print(f"{name} à {time:.3f} s")



#         # ============================================================
#         # B. Preprocessing
#         # ============================================================


#         # =====================
#         # 1. Define the montage
#         # =====================

#         raw.set_montage("standard_1020") # to adapt according to the montage used during the exepriments

#         Sensors_Montage_Figure_1 = raw.plot_sensors(show_names=True, show=True)
#         # save_figure(Sensors_Montage_Figure_1, "Figure1_SensorsMontage.png")
#         # Sensors_Montage_Figure_1.fig.savefig("Figure1_SensorsMontage.png")


#         # ===================================
#         # 2. Filter the data with a band-pass
#         # ===================================

#         # Define the high and low frequencies
#         HFreq = 100
#         LFreq = 0.5
#         raw_HighLowPassed = raw.filter(l_freq = LFreq, h_freq = HFreq)

#         # Plot the highpassed signal
#         Signal_HighLowPassed_Figure_2 = raw_HighLowPassed.plot(title = "High- and Low- passed Signal", show=True)
#         # save_figure(Signal_HighLowPassed_Figure_2, f"Figure2_{subject_name}_Signal-HighLowPassed.png")
#         # Signal_HighLowPassed_Figure_2.fig.savefig(f"Figure2_{subject_name}_Signal-HighLowPassed.png")


#         # ===============================
#         # 3. Filter the data with a Notch
#         # ===============================

#         # Define the parameters for the notch filter
#         if HFreq < 50:
#             raw_Notched = raw_HighLowPassed
#         else:
#             raw_Notched = raw_HighLowPassed.notch_filter(freqs = [50], picks = "data", method = "spectrum_fit")

#         # Plot the notched signal
#         Signal_Notched_Figure_3 = raw_Notched.plot(title = "Notched Signal", show=True)
#         # save_figure(Signal_Notched_Figure_3, f"Figure3_{subject_name}_Signal-Notched.png")
#         # Signal_Notched_Figure_3.fig.savefig(f"Figure3_{subject_name}_Signal-Notched.png")


#         # ============================
#         # 4. Identify the bad channels
#         # ============================


#         # ========================
#         # 5. Re-reference the data
#         # ========================

#         # REST method : advanced EEG re-referencement
#         print("Application REST referencial ...")

#         # Create a spherical model of the head based on the file information 
#         Sphere = mne.make_sphere_model('auto', 'auto', raw_Notched.info)

#         # Define the volume source space
#         Source = mne.setup_volume_source_space(sphere=Sphere, exclude=30.0, pos=5.0, mri=None, verbose=False)

#         # Calculate the forward model solution
#         Forward = mne.make_forward_solution(raw_Notched.info, trans=None, src=Source, bem=Sphere, verbose=False)

#         # Apply the REST reference
#         raw_REST = raw_Notched.copy().set_eeg_reference('REST', forward=Forward)

#         # Optionnal : visualisation after REST
#         Signal_REST_Figure_4 = raw_REST.plot(title="Signal after REST reference", show=True)
#         # save_figure(Signal_REST_Figure_4, f"Figure4_{subject_name}_Signal_REST.png")
#         # Signal_REST_Figure_4.fig.savefig(f"Figure4_{subject_name}_Signal_REST.png")
#         Signal_REST_PSD_Figure_4_bis = raw_REST.plot_psd(fmin=0, fmax=100, show=True) # PSD = Power Spectrum Density
#         # save_figure(Signal_REST_PSD_Figure_4_bis, f"Figure4bis_{subject_name}_Signal_REST_PSD.png")
#         # Signal_REST_PSD_Figure_4_bis.fig.savefig(f"Figure4bis_{subject_name}_Signal_REST_PSD.png")


#         # ===========
#         # 6. Recalage
#         # ===========

#         # Reset the file time
#         if events_times_sec[0] == 0 and len(events_times_sec) > 1: # check if the 1st stimulus is at 0 s. If so, use the 2nd stimulus
#             first_stimulus_time = events_times_sec[1]
#             print(f"First stimulus is at 0. Using second stimulus at {first_stimulus_time:.3f} s")
#         else:
#             first_stimulus_time = events_times_sec[0]
#             print(f"First stimulus at {first_stimulus_time:.3f} s")

#         # Truncate the signal to start at this point
#         raw_cropped = raw_REST.copy().crop(tmin=first_stimulus_time)

#         # Reset the annotations by shifting all annotations by - first_stimulus_time
#         if raw_REST.annotations is not None:
#             # raw_annotation_times = raw.annotations.onset - first_stimulus_time
#             onset_times = raw_REST.annotations.onset - first_stimulus_time
#             mask_valid = onset_times >= 0
#             raw_cropped.set_annotations(
#                 mne.Annotations(
#                     # onset=raw_annotation_times # onset = raw_annotation_times with the new reset times
#                     onset=onset_times[mask_valid],
#                     # duration=raw.annotations.duration
#                     duration=raw_REST.annotations.duration[mask_valid],
#                     # description=raw.annotations.description
#                     description=[d for d, valid in zip(raw_REST.annotations.description, mask_valid) if valid]
#                 )
#             )
#             print("\n=== Annotations after the recalage ===")
#             print(raw_cropped.annotations)

#             print("\n=== All the annotations labels ===")
#             print(set(raw_cropped.annotations.description))

#         # Plot truncated and recalculated data
#         Readjusted_Signal_Figure_5 = raw_cropped.plot(title="Readjusted signal (from the 1st stimulus not at 0 s)", show=True)



#         # ============================================================
#         # C. Epoching
#         # ============================================================


#         # =======================================
#         # 1. Phases dictionnary (with TTL limits)
#         # =======================================

        # phases_dict = {
        # "spontaneous_tics": {"start": "Stimulus/S  9", "end": "Stimulus/S 10"},
        # "imitated_tics": {"start": "Stimulus/S 11", "end": "Stimulus/S 12"},
        # "retention_tics": {"start": "Stimulus/S 13", "end": "Stimulus/S 14"}
        # }


        # # =====================
        # # 2. Go over each phase
        # # =====================
#         for phase_name, phase_TTL in phases_dict.items():
#             try:
#                 # Get the beginning & end timestamps based on the annotations
#                 annotations = raw_cropped.annotations
#                 start_times = [ann['onset'] for ann, desc in zip(annotations, annotations.description) if desc == phase_TTL['start']]
#                 end_times = [ann['onset'] for ann, desc in zip(annotations, annotations.description) if desc == phase_TTL['end']]
        
#                 if len(start_times) == 0 or len(end_times) == 0:
#                     print(f"Phase {phase_name} not found for {subject_name}, missing TTL")
#                     continue
        
#                 start_time = start_times[0]
#                 end_time = end_times[0]

#                 print(f"{phase_name}: {start_time:.2f}s → {end_time:.2f}s")

#                 # Extract the phase with crop()
#                 end_time = min(end_times[0], raw_cropped.times[-1])
#                 phase_raw = raw_cropped.copy().crop(tmin=start_time, tmax=end_time)

#                 # Save the file for this phase
#                 folder_phase = "C:\\Users\\indira.lavocat\\MOVIDOC\\tictrack_eeg_analysis\\EEG_Phases"
#                 if not os.path.exists(folder_phase):
#                     os.makedirs(folder_phase)

#                 phase_file_path = os.path.join(folder_phase, f"{subject_name}_{phase_name}_raw.fif")
#                 phase_raw.save(phase_file_path, overwrite=True)
#                 print(f"Saved phase : {phase_file_path}")
#             except Exception as e:
#                 print(f"Error while cutting the phase {phase_name} for {subject_name} : {e}")
#                 continue

#     except Exception as e:
#         print(f"Erreur pour {FilePath} : {e}")
#         continue

# app.exec_()