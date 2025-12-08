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
    # # raw.plot(show=True)
    # display the PSD (Power Spectral Density) of the signal
    # # raw.plot_psd(show=True)

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
    # apply the montage defined by the user
    raw.set_montage(montage_name)
    # visualize the electrode placement
    # # raw.plot_sensors(show_names=True, show=True)

    # apply band-pass filter between 0.5 and 100 Hz
    raw = raw.filter(l_freq=1, h_freq=40)
    # visualize the band-passed signal
    # raw.plot(title="High/Low Pass Filter", show=True)

    # apply Notch filter at 50 Hz to remove the powerline noise
    raw = raw.notch_filter(freqs=[50], picks="data", method="spectrum_fit")
    # visualize the notched signal
    # raw.plot(title="Notch Filter", show=True)

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

    # # visualize the re-referenced signal
    # raw_rest.plot(title="Signal REST", show=True)
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