# Libraries
import os
import mne

# > To save the figures that are not MNEQtBrowser
# Define a function to save all the Figures that are NOT MNEQtBrowser
def save_figure(fig, filename, folder="C:\\Users\\indira.lavocat\\MOVIDOC\\tictrack_eeg_analysis\\Figures-Patients"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    fig_path = os.path.join(folder, filename)
    fig.savefig(fig_path)
    print(f"Figure saved: {fig_path}")



# ============================================================
# A. Data
# ============================================================


# ================
# 1. Load the data
# ================

# List of the.vhdr files to load
vhdr_files = [
    "C:\\Users\\indira.lavocat\\MOVIDOC\\EEG\\EEG patients files\\MOVIDOCTicTrack000010.vhdr", #DS26
    "C:\\Users\\indira.lavocat\\MOVIDOC\\EEG\\EEG patients files\\MOVIDOCTicTrack_BB28-bis.vhdr", #BB28
    "C:\\Users\\indira.lavocat\\MOVIDOC\\EEG\\EEG patients files\\MOVIDOCTicTrack000013.vhdr", #BC29
    "C:\\Users\\indira.lavocat\\MOVIDOC\\EEG\\EEG patients files\\MOVIDOCTicTrack000030.vhdr", #MM30
    "C:\\Users\\indira.lavocat\\MOVIDOC\\EEG\\EEG patients files\\MOVIDOCTicTrack000031.vhdr" #SC31
]

for FilePath in vhdr_files:
    try:
        filename = os.path.basename(FilePath) # "MOVIDOCTicTrack_SC31.vhdr"
        subject_name = filename.replace("MOVIDOCTicTrack_", "").replace(".vhdr", "")
        
        print(f"\n--- Treatment of the file : {subject_name} ---\n")

        raw = mne.io.read_raw_brainvision(FilePath, preload=True)
        raw.info
        print(raw.ch_names)
        print(raw.info['description']) # gives a note about the channels when there is one


        # ======================
        # 2. Extract the stimuli
        # ======================

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



        # ============================================================
        # B. Preprocessing
        # ============================================================


        # =====================
        # 1. Define the montage
        # =====================

        raw.set_montage("standard_1020") # to adapt according to the montage used during the exepriments

        Sensors_Montage_Figure_1 = raw.plot_sensors(show_names=True, show=False)
        # save_figure(Sensors_Montage_Figure_1, "Figure1_SensorsMontage.png")
        Sensors_Montage_Figure_1.fig.savefig("Figure1_SensorsMontage.png")


        # ===================================
        # 2. Filter the data with a band-pass
        # ===================================

        # Define the high and low frequencies
        HFreq = 100
        LFreq = 0.5
        raw_HighLowPassed = raw.filter(l_freq = LFreq, h_freq = HFreq)

        # Plot the highpassed signal
        Signal_HighLowPassed_Figure_2 = raw_HighLowPassed.plot(title = "High- and Low- passed Signal", show=False)
        # save_figure(Signal_HighLowPassed_Figure_2, f"Figure2_{subject_name}_Signal-HighLowPassed.png")
        # Signal_HighLowPassed_Figure_2.fig.savefig(f"Figure2_{subject_name}_Signal-HighLowPassed.png")


        # ===============================
        # 3. Filter the data with a Notch
        # ===============================

        # Define the parameters for the notch filter
        if HFreq < 50:
            raw_Notched = raw_HighLowPassed
        else:
            raw_Notched = raw_HighLowPassed.notch_filter(freqs = [50], picks = "data", method = "spectrum_fit")

        # Plot the notched signal
        Signal_Notched_Figure_3 = raw_Notched.plot(title = "Notched Signal", show=False)
        # save_figure(Signal_Notched_Figure_3, f"Figure3_{subject_name}_Signal-Notched.png")
        # Signal_Notched_Figure_3.fig.savefig(f"Figure3_{subject_name}_Signal-Notched.png")


        # ============================
        # 4. Identify the bad channels
        # ============================


        # ========================
        # 5. Re-reference the data
        # ========================

        # REST method : advanced EEG re-referencement
        print("Application REST referencial ...")

        # Create a spherical model of the head based on the file information 
        Sphere = mne.make_sphere_model('auto', 'auto', raw_Notched.info)

        # Define the volume source space
        Source = mne.setup_volume_source_space(sphere=Sphere, exclude=30.0, pos=5.0, mri=None, verbose=False)

        # Calculate the forward model solution
        Forward = mne.make_forward_solution(raw_Notched.info, trans=None, src=Source, bem=Sphere, verbose=False)

        # Apply the REST reference
        raw_REST = raw_Notched.copy().set_eeg_reference('REST', forward=Forward)

        # Optionnal : visualisation after REST
        Signal_REST_Figure_4 = raw_REST.plot(title="Signal après référence REST", show=False)
        # save_figure(Signal_REST_Figure_4, f"Figure4_{subject_name}_Signal_REST.png")
        # Signal_REST_Figure_4.fig.savefig(f"Figure4_{subject_name}_Signal_REST.png")
        Signal_REST_PSD_Figure_4_bis = raw_REST.plot_psd(fmin=0, fmax=100, show=False) # PSD = Power Spectrum Density
        # save_figure(Signal_REST_PSD_Figure_4_bis, f"Figure4bis_{subject_name}_Signal_REST_PSD.png")
        # Signal_REST_PSD_Figure_4_bis.fig.savefig(f"Figure4bis_{subject_name}_Signal_REST_PSD.png")


        # ===========
        # 6. Recalage
        # ===========

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


    except Exception as e:
        print(f"Erreur pour {FilePath} : {e}")
        continue

