# Libraries
import os
import mne

# import sys
# sys.path.append("C:\\Users\\indira.lavocat\\MOVIDOC\\tictrack_eeg_analysis")
# from Movidoc_tictrack_prepro_patients_functions import *

from qtpy import QtWidgets

app = QtWidgets.QApplication.instance()
if app is None:
    app = QtWidgets.QApplication([])



# ============================================================
# Define functions
# ============================================================


# =================================================
# To save all the figures that are not MNEQtBrowser
# =================================================

def save_figure(fig, filename, folder="C:\\Users\\indira.lavocat\\MOVIDOC\\tictrack_eeg_analysis\\Figures-Patients"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    fig_path = os.path.join(folder, filename)
    fig.savefig(fig_path)
    print(f"Figure saved: {fig_path}")


# ================
# To load the data
# ================

def load_data(FilePath):
    filename = os.path.basename(FilePath)
    subject_name = filename.replace("MOVIDOCTicTrack", "").replace(".vhdr", "")
    print(f"\n--- Traitement du fichier : {subject_name} ---\n")

    raw = mne.io.read_raw_brainvision(FilePath, preload=True)
    raw.plot(show=True)
    raw.plot_psd(show=True)

    raw.info
    print(raw.ch_names)
    print(raw.info['description'])
    return raw, subject_name


# ======================
# To extract the stimuli
# ======================

def extract_stimuli(raw):
    events, event_id = mne.events_from_annotations(raw)
    print("Events list :")
    print(event_id)
    print(events)

    events_no_zero = events[events[:, 0] != 0]
    events_times_sec = events_no_zero[:, 0] / raw.info['sfreq']

    id_to_name = {v: k for k, v in event_id.items()}
    for time, eid in zip(events_times_sec, events_no_zero[:, 2]):
        name = id_to_name.get(eid, f"ID {eid}")
        print(f"{name} à {time:.3f} s")

    return events_times_sec, event_id


# =================================================================
# To define the montage, filter the data with a band-pass & a Notch
# =================================================================

def preprocess_data(raw, subject_name):
    raw.set_montage("standard_1020")
    raw.plot_sensors(show_names=True, show=True)

    raw = raw.filter(l_freq=0.5, h_freq=100)
    raw.plot(title="High/Low Pass Filter", show=True)

    raw = raw.notch_filter(freqs=[50], picks="data", method="spectrum_fit")
    raw.plot(title="Notch Filter", show=True)

    return raw


# ========================
# To re-reference the data
# ========================

def apply_rest_reference(raw, subject_name):
    print("Applying REST reference...")
    
    Sphere = mne.make_sphere_model('auto', 'auto', raw.info)
    Source = mne.setup_volume_source_space(sphere=Sphere, exclude=30.0, pos=5.0, verbose=False)
    Forward = mne.make_forward_solution(raw.info, trans=None, src=Source, bem=Sphere, verbose=False)

    raw_rest = raw.copy().set_eeg_reference('REST', forward=Forward)

    raw_rest.plot(title="Signal REST", show=True)
    raw_rest.plot_psd(fmin=0, fmax=100, show=True)

    return raw_rest


# ==================
# To do the recalage
# ==================

def recalibrate_from_first_event(raw, events_times_sec):
    if len(events_times_sec) == 0:
        print("Aucun événement détecté — recalage impossible.")
        return raw

    first_stimulus_time = events_times_sec[1] if events_times_sec[0] == 0 and len(events_times_sec) > 1 else events_times_sec[0]
    print(f"First stimulus at {first_stimulus_time:.3f}s")

    raw_cropped = raw.copy().crop(tmin=first_stimulus_time)

    if raw.annotations is not None:
        new_onsets = raw.annotations.onset - first_stimulus_time
        valid_mask = new_onsets >= 0
        raw_cropped.set_annotations(mne.Annotations(
            onset=new_onsets[valid_mask],
            duration=raw.annotations.duration[valid_mask],
            description=[d for d, v in zip(raw.annotations.description, valid_mask) if v]
        ))

    raw_cropped.plot(title="Signal cropped", show=True)
    return raw_cropped


# ====================
# To create the epochs
# ====================

def extract_phases(raw_cropped, subject_name, save_folder="C:\\Users\\indira.lavocat\\MOVIDOC\\tictrack_eeg_analysis\\EEG_Phases"):
    
    # Define the phases dictionnary
    phases_dict = {
        "spontaneous_tics": {"start": "Stimulus/S  9", "end": "Stimulus/S 10"},
        "imitated_tics": {"start": "Stimulus/S 11", "end": "Stimulus/S 12"},
        "retention_tics": {"start": "Stimulus/S 13", "end": "Stimulus/S 14"}
    }

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Loop for the phases
    for phase_name, phase_TTL in phases_dict.items():
        try:
            annotations = raw_cropped.annotations
            # Get the beginning & end timestamps
            start_times = [ann['onset'] for ann, desc in zip(annotations, annotations.description) if desc == phase_TTL['start']]
            end_times = [ann['onset'] for ann, desc in zip(annotations, annotations.description) if desc == phase_TTL['end']]

            if len(start_times) == 0 or len(end_times) == 0:
                print(f"Phase {phase_name} not found for {subject_name}, missing TTL")
                continue

            start_time = start_times[0]
            end_time = min(end_times[0], raw_cropped.times[-1])

            print(f"{phase_name}: {start_time:.2f}s → {end_time:.2f}s")

            # Cut the phase
            phase_raw = raw_cropped.copy().crop(tmin=start_time, tmax=end_time)

            # Save the file for the phase
            phase_file_path = os.path.join(save_folder, f"{subject_name}_{phase_name}_raw.fif")
            phase_raw.save(phase_file_path, overwrite=True)
            print(f"Saved phase : {phase_file_path}")

        except Exception as e:
            print(f"Error while cutting the phase {phase_name} for {subject_name} : {e}")
            continue



# ============================================================
# A. Data
# ============================================================


# ================
# 1. Load the data
# ================

# List of the.vhdr files to load
vhdr_files = [
    "C:\\Users\\indira.lavocat\\MOVIDOC\\EEG\\EEG PATIENT FILES\\MOVIDOCTicTrack000010.vhdr", #DS26
    "C:\\Users\\indira.lavocat\\MOVIDOC\\EEG\\EEG PATIENT FILES\\MOVIDOCTicTrack_BB28-bis.vhdr", #BB28
    "C:\\Users\\indira.lavocat\\MOVIDOC\\EEG\\EEG PATIENT FILES\\MOVIDOCTicTrack000013.vhdr", #BC29
    "C:\\Users\\indira.lavocat\\MOVIDOC\\EEG\\EEG PATIENT FILES\\MOVIDOCTicTrack000030.vhdr", #MM30
    "C:\\Users\\indira.lavocat\\MOVIDOC\\EEG\\EEG PATIENT FILES\\MOVIDOCTicTrack000031.vhdr" #SC31
]

#########################################################################
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