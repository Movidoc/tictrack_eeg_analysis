# =======================================================================
# File : _04_Movidoc_tictrack_merge_extractions_patients.py
# Purpose : Merge the TTL & tics beginning/end by patient
# Author  : Indira
# =======================================================================



# ============================================================
# Libraries
# ============================================================

from _02_Movidoc_tictrack_prepro_TTL_extraction_patients import (
    load_data,
    extract_stimuli,
    preprocess_data, 
    apply_rest_reference,
    recalibrate_from_first_event,
    collect_ttl_with_phases
)

from _03_Movidoc_tictrack_tic_extraction_patients import extract_tics_from_excel

from pprint import pprint

import mne



# ============================================================
# Disable all MNE interactive plots globally
# ============================================================

# Save the original plot functions
original_plot = mne.io.BaseRaw.plot
original_plot_psd = mne.io.BaseRaw.plot_psd

# Define new functions that force show=False
def plot_no_show(self, *args, **kwargs):
    kwargs['show'] = False
    return original_plot(self, *args, **kwargs)

def plot_psd_no_show(self, *args, **kwargs):
    kwargs['show'] = False
    return original_plot_psd(self, *args, **kwargs)

# Monkey-patch the MNE Raw plotting functions
mne.io.BaseRaw.plot = plot_no_show
mne.io.BaseRaw.plot_psd = plot_psd_no_show



# ============================================================
# Define functions
# ============================================================


# ==============================================================
# Function : assign_phase_to_tics
# Purpose : assign the corresponding phase to the extracted tics
# ==============================================================

def assign_phase_to_tics(tics, phases_dict):

    # list that will contain the tics with the associated phase
    tics_with_phase = []

    # for each extracted tic (start_time, end_time)
    for start, end in tics:
        tic_phase = None # initialization of the phase at None

        # for each phase (with its timestamps)
        for phase_name, (p_start, p_end) in phases_dict.items():
            # if a tic is in & out a phase → it belongs to this phase
            if end >= p_start and start <= p_end:
                tic_phase = phase_name
                break

        # add the tic with its associated phase to the list
        tics_with_phase.append({
            "start": float(start),
            "end": float(end),
            "phase": tic_phase
        })

    return tics_with_phase


# ===============================================================================
# Function : run_full_pipeline_for_patient
# Purpose : run the full pipeline by running the functions from the 02 & 03 files
# ===============================================================================

def run_full_pipeline_for_patient(vhdr_path, excel_path, fps, min_absence_frames):
    
    # 1.a. Process EEG file (.vhdr)
    raw, subject_name = load_data(vhdr_path) # charge the EEG file & get the name of the subject
    # events_times, _ = extract_stimuli(raw) # extract the TTL/events from the signal BEFORE the recalage
    raw_pre = preprocess_data(raw, subject_name, montage_name=p["montage"]) # filter the signal & apply the montage
    raw_rest = apply_rest_reference(raw_pre, subject_name) # apply the REST reference
    print("Before crop:", raw_rest.annotations.onset[:5])
    raw_cropped = recalibrate_from_first_event(raw_rest, target_stim="Stimulus/S  2") # readjust the signal from the 1st significative TTL
    events_times, event_id = extract_stimuli(raw_cropped) # extract the TTL/events from the signal AFTER the recalage
    print("After crop:", raw_cropped.annotations.onset[:5])

    # verification step
    # stim2_times = [time for time, eid in zip(events_times, event_id) if event_id.get(eid) == "Stimulus/S  2"]
    # print("Time of the Stimulus/S  2 after recalage:", stim2_times)

    # 1.b. Extract TTL information
    ttl_info = collect_ttl_with_phases(raw_cropped, subject_name) # get the TTL list & their phases

    # get the time of Stimulus/S 2 pour recalage
    stim2_time = next((ttl["time"] for ttl in ttl_info if ttl["ttl_name"] == "Stimulus/S  2"), 0.0)
    print(f"Stimulus/S 2 time: {stim2_time:.3f} s")

    # define the phases via TTLs
    phases_ttl = {
        # "press_key": {"start": "Stimulus/S  3", "end": "Stimulus/S  4"},
        # "eyes_closed": {"start": "Stimulus/S  5", "end": "Stimulus/S  6"},
        # "eyes_open": {"start": "Stimulus/S  7", "end": "Stimulus/S  8"},
        "spontaneous_tics": {"start": "Stimulus/S  9", "end": "Stimulus/S 10"},
        "imitated_tics": {"start": "Stimulus/S 11", "end": "Stimulus/S 12"},
        "retention_tics": {"start": "Stimulus/S 13", "end": "Stimulus/S 14"}
    }

    # dictionnary that will contain the exact timestamps of beginning & end of each phase
    phases_dict = {}

    # for each phase defined via TTL
    for phase_name, ttl_names in phases_ttl.items():
        # get the TTL timestamp of the beginning of this phase
        start_time = next((ttl["time"] for ttl in ttl_info if ttl["ttl_name"] == ttl_names["start"]), None)
        # get the TTL timestamp of the end of this phase
        end_time   = next((ttl["time"] for ttl in ttl_info if ttl["ttl_name"] == ttl_names["end"]), None)

        # if the TTL of the beginning is missing
        if start_time is None:
            raise ValueError(f"Start TTL missing for phase '{phase_name}' in subject {subject_name}")
        # if the TTL of the end is missing
        if end_time is None:
            if phase_name == "retention_tics":
                # get the last timestamp of the signal
                end_time = raw_cropped.times[-1]
            else:
                raise ValueError(f"End TTL missing for phase '{phase_name}' in subject {subject_name}")
        
        # save the tuple (start, end) for each phase
        phases_dict[phase_name] = (start_time, end_time)
    
    for phase_name in phases_dict:
        start, end = phases_dict[phase_name]
        phases_dict[phase_name] = (start - stim2_time, end - stim2_time)

    # print the debug after the loop to check
    print("\n===== DEBUG: Phases readjusted timestamps for this patient =====")
    for phase, (s, e) in phases_dict.items():
        print(f"{phase:20s}  start={s:.3f}  end={e:.3f}")

    # 2. Extract tics from Excel
    tics = extract_tics_from_excel(excel_file=excel_path, fps=fps, min_absence_frames=min_absence_frames) # extract the tics from Excel

    # assign the phases after the recalage
    tics_info = assign_phase_to_tics(tics , phases_dict) # associate each tic to its phase

    # 3. Merge into one Python object
    full_output = {
        "subject": subject_name, # name of the subject
        "ttl": ttl_info, # list of the TTL with phases
        "tics": tics_info # list of the tics with phases
    }

    # return the full object for the patient
    return full_output

#########################################################################



# dictionnary to stock the results of all the patients
results = {}

patients = [
    {
        "montage": "standard_1020", # montage with 32 electrodes (from DS26 to BC29 : always 32 electrodes)
        "vhdr": "C:\\Users\\indira.lavocat\\MOVIDOC\\PATIENT FILES\\EEG PATIENT FILES\\MOVIDOCTicTrack_BB28-bis.vhdr",
        "excel": "C:\\Users\\indira.lavocat\\MOVIDOC\\PATIENT FILES\\EXCEL PATIENT FILES\\BB28_annotations_binary-table_cutted.xlsx",
        # "phase_start": 302600 / 1000.0, # start spontaneous_tics
        # "phase_end": 1591880 / 1000.0, # end retention tics (929960 = end spontaneous_tics)
        "fps": 25,
        "min_absence_frames": 25
    },
    {
        "montage": "standard_1020", # montage with 32 electrodes (from DS26 to BC29 : always 32 electrodes)
        "vhdr": "C:\\Users\\indira.lavocat\\MOVIDOC\\PATIENT FILES\\EEG PATIENT FILES\\MOVIDOCTicTrack000013.vhdr",
        "excel": "C:\\Users\\indira.lavocat\\MOVIDOC\\PATIENT FILES\\EXCEL PATIENT FILES\\BC29_annotations_binary-table_cutted_xlsx_Lizbeth.xlsx",
        # "phase_start": 358.215, # start spontaneous_tics
        # "phase_end": 1088.67, # end spontaneous_tics
        "fps": 30,
        "min_absence_frames": 30
    },
    {
        "montage": "standard_1005", # montage with 64 electrodes (from MM30 : always 64 electrodes)
        "vhdr": "C:\\Users\\indira.lavocat\\MOVIDOC\\PATIENT FILES\\EEG PATIENT FILES\\MOVIDOCTicTrack000031.vhdr",
        "excel": "C:\\Users\\indira.lavocat\\MOVIDOC\\PATIENT FILES\\EXCEL PATIENT FILES\\SC31_annotations_binary-table_cutted.xlsx",
        # "phase_start": 345.972, # start spontaneous_tics
        # "phase_end": 1668.645, # end retention_tics (970.167 = end spontaneous_tics)
        "fps": 30,
        "min_absence_frames": 30
    }
]

# iterate on all the patients to execute the complete pipeline
for p in patients:
    results[p["vhdr"]] = run_full_pipeline_for_patient(
        vhdr_path=p["vhdr"], # path to the EEG file
        excel_path=p["excel"], # path to the Excel file
        # phase_start=p["phase_start"],
        # phase_end=p["phase_end"],
        fps=p["fps"], # fps of the Excel file
        min_absence_frames=p["min_absence_frames"] # minimum of frames of "Absence" to define the beginning & end of the tics
    )

print("Pipeline finished for all the patients.")


# print the result by patient
for vhdr_path, patient in results.items():
    print("\n" + "="*60)
    print(f" Patient : {patient['subject']}")
    print("="*60)

    # print("\nTTLs :")
    # pprint(patient["ttl"])

    print("\nTics :")
    for tic in patient["tics"]:
        print(f"start={tic['start']:.3f}  end={tic['end']:.3f}  phase={tic['phase']}")

print("Patients analysés :", list(results.keys()))

#########################################################################