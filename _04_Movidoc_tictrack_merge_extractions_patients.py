# =======================================================================
# File : _04_Movidoc_tictrack_merge_extractions_patients.py
# Purpose : Merge the TTL & tics beginning/end by patient
# Author  : Indira
# =======================================================================

print(">>> LE SCRIPT A BIEN ÉTÉ LANCÉ")



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

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re



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


# ============================================================================
# Function : extract_eeg_phase_times_from_ttl
# Purpose : extract the timestamps of the TTLs used to realign the Excel times
# ============================================================================

def extract_eeg_phase_times_from_ttl(ttl_info):

    target_stims = [
        "Stimulus/S  3",
        "Stimulus/S  5",
        "Stimulus/S  7",
        "Stimulus/S  9",
        "Stimulus/S 11",
        "Stimulus/S 13"
    ]

    eeg_times = []

    for stim in target_stims:
        time = next((ttl["time"] for ttl in ttl_info if ttl["ttl_name"] == stim), None)
        if time is None:
            raise ValueError(f"Stimulus {stim} not found in ttl_info — impossible to calculate eeg_phase_times.")
        eeg_times.append(time)

    return eeg_times


# ============================================================================
# Function : realign_excel_to_eeg
# Purpose : realign the Excel times on the EEG times
# ============================================================================
def realign_excel_to_eeg(excel_times, eeg_times):
    """
    ----------
    Purpose
    ----------
    Calculate the linear transformation to realign the Excel times on the EEG times.

    ----------
    Parameters
    ----------
    excel_times : list of the Excel phases (beginning) times [S3, S5, S7, S9, S11, S13].
    eeg_times   : list of the corresponding EEG times.

    ----------
    Returns
    ----------
    List of Excel times realign on the EEG times.
    """

    import numpy as np

    if len(excel_times) != len(eeg_times):
        raise ValueError("The lists excel_times & eeg_times need to have the same length.")

    # linear fit : eeg_time = a * excel_time + b
    a, b = np.polyfit(excel_times, eeg_times, 1)

    # Apply the realignment on all the Excel times
    corrected_times = [a * t + b for t in excel_times]
    
    return corrected_times, a, b


# ===============================================================================
# Function : run_full_pipeline_for_patient
# Purpose : run the full pipeline by running the functions from the 02 & 03 files
# ===============================================================================

def run_full_pipeline_for_patient(vhdr_path, excel_path, fps, min_absence_frames, montage_name):

    print(f"\n===== START Patient {vhdr_path} =====")
    
    # 1.a. Process EEG file (.vhdr)
    raw, subject_name = load_data(vhdr_path) # charge the EEG file & get the name of the subject
    # events_times, _ = extract_stimuli(raw) # extract the TTL/events from the signal BEFORE the recalage
    raw_pre = preprocess_data(raw, subject_name, montage_name=montage_name) # filter the signal & apply the montage
    raw_rest = apply_rest_reference(raw_pre, subject_name) # apply the REST reference
    print("Before crop:", raw_rest.annotations.onset[:5])
    print("Checkpoint 1 : EEG preprocessing OK")
    raw_cropped = recalibrate_from_first_event(raw_rest, target_stim="Stimulus/S  2") # readjust the signal from the 1st significative TTL
    events_times, event_id = extract_stimuli(raw_cropped) # extract the TTL/events from the signal AFTER the recalage
    print("After crop:", raw_cropped.annotations.onset[:5])

    # 1.b. Extract TTL information
    ttl_info = collect_ttl_with_phases(raw_cropped, subject_name) # get the TTL list & their phases

    # Get the time of Stimulus/S  2
    stim2_time = next((ttl["time"] for ttl in ttl_info if ttl["ttl_name"] == "Stimulus/S  2"), None)
    if stim2_time is None:
        raise ValueError("Stimulus/S  2 introuvable dans ttl_info. Impossible de recaler.")
    # Realign all the TTLs on the Stimulus/S  2
    for ttl in ttl_info:
        ttl["time"] = round(ttl["time"] - stim2_time, 3)
    


    # Extract automatically the timestamps of the TTLs used to realign the Excel times to the EEG times
    eeg_phase_times = extract_eeg_phase_times_from_ttl(ttl_info)
    print("\n===== EEG phase times (from TTL) =====")
    print(eeg_phase_times)

    # Realign Excel -> EEG
    excel_corrected_times, slope, intercept = realign_excel_to_eeg(excel_times=p["excel_phase_times"], eeg_times=eeg_phase_times)
    print("\n===== Excel times realigned on EEG =====")
    print(excel_corrected_times)
    print(f"Linear drift: slope={slope:.6f}, intercept={intercept:.6f}")
    print("Checkpoint 2 : TTL extracted and realigned OK")



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
        # if end_time is None:
        #     if phase_name == "retention_tics":
        #         # get the last timestamp of the signal
        #         end_time = raw_cropped.times[-1]
        #     else:
        #         raise ValueError(f"End TTL missing for phase '{phase_name}' in subject {subject_name}")
        
        # save the tuple (start, end) for each phase
        phases_dict[phase_name] = (start_time, end_time)
    
    # for phase_name in phases_dict:
    #     start, end = phases_dict[phase_name]
    #     phases_dict[phase_name] = (start - stim2_time, end - stim2_time)

    print("Checkpoint 3 : phases_dict OK")

    # print the debug after the loop to check
    print("\n===== DEBUG: Phases readjusted timestamps for this patient =====")
    for phase, (s, e) in phases_dict.items():
        print(f"{phase:20s}  start={s:.3f}  end={e:.3f}")

    # 2. Extract tics from Excel
    tics_original = extract_tics_from_excel(excel_file=excel_path, fps=fps, min_absence_frames=min_absence_frames) # extract the tics from Excel
    print("Checkpoint 4 : tics extracted from Excel")

    tics_original_with_phases = assign_phase_to_tics(tics_original , phases_dict) # associate each tic to its phase
    print("Checkpoint 5 : tics extracted from Excel & phases assigned")

    # create a version realigned on EEG using the linear drift parameters ----
    tics_corrected = [(start * slope + intercept, end * slope + intercept) for start, end in tics_original]
    print("Checkpoint 6 : tics extracted from Excel are corrected")

    # assign the phases after the recalage
    tics_corrected_with_phases = assign_phase_to_tics(tics_corrected , phases_dict) # associate each tic to its phase
    print("Checkpoint 7 : tics extracted from Excel are corrected & phases assigned")

    # 3. Merge into one Python object
    full_output = {
        "subject": subject_name, # name of the subject
        "excel_corrected_times": excel_corrected_times,
        "linear_drift_params": {"slope": slope, "intercept": intercept},
        "ttl": ttl_info, # list of the TTL with phases
        # "tics": tics_info # list of the tics with phases
        "tics_original": tics_original_with_phases,
        "tics_corrected": tics_corrected_with_phases
    }

    full_output["phases_dict"] = phases_dict

    # 4. Build merged TTL + tics timeline
    phases_to_keep = ["spontaneous_tics", "imitated_tics", "retention_tics"]
    merged_ttl_tics = build_merged_ttl_tics(patient=full_output, phases_to_keep=phases_to_keep)
    full_output["merged_ttl_tics"] = merged_ttl_tics

    print("Checkpoint 8 : ready to return full_output")

    # return the full object for the patient
    return full_output


# ============================================================================
# Function : build_merged_ttl_tics
# Purpose  : create single merged timeline of corrected tics & TTL times
# ============================================================================

def build_merged_ttl_tics(patient, phases_to_keep):
    """
    Build a merged list of:
    - start_i / end_i from patient["tics_corrected"]
    - time_Sxx or time_Sxx_i from patient["ttl"]
    # Filtered by phases_to_keep.
    """

    filter_dict = {
        'D': 21, 
        'F': 22,
        'S': 23,
        'T': 24,
        'right': 25,
        'start_spont': 9,
        'end_spont': 10,
        'start_imit': 11,
        'end_imit': 12,
        'start_ret': 13,
        'end_ret': 14
        }
    print(f"\n\n\nThis is the dictionnary : {filter_dict}\n\n\n")
    
    merged = []

    # ------------------------------------------------------------
    # 1. Index the tics : start_1, end_1, start_2, end_2, ...
    # ------------------------------------------------------------
    tics = patient["tics_corrected"]
    phases_dict = patient["phases_dict"]

    for i, tic in enumerate(tics, start=1):
        tic_phase = tic["phase"]

        if tic_phase in phases_to_keep:
            start_key = f"start_{i}"
            end_key = f"end_{i}"

            merged.append({start_key: round(tic["start"], 3)})
            merged.append({end_key: round(tic["end"], 3)})

    # ------------------------------------------------------------
    # 2. Index TTLs: S9, S25_3, ...
    # ------------------------------------------------------------

    ttl_list = patient["ttl"]

    # Build list of intervals to keep
    intervals = [phases_dict[p] for p in phases_to_keep]
    # count occurrences per stimulus name
    stim_counts = {}

    for ttl in ttl_list:
        # ttl_phase = ttl["phase"] # e.g. 'imitated_tics'
        ttl_time = ttl["time"]
        ttl_name = ttl["ttl_name"] # e.g. 'Stimulus/S  9'

        # keep TTL only if its timestamp falls inside one of the intervals
        inside = any(start <= ttl_time <= end for start, end in intervals)

        if not inside:
            continue

        # Extract the number after 'S'
        try:
            # for filter_key, filter_value in filter_dict.items():
            #     if filter_value == int(ttl_name.split("S")[2]) :
            #         stim_num = filter_key
            num = int(ttl_name.split("/S")[1].strip())
        except:
            continue

        # key = stim_num
        key = {v: k for k, v in filter_dict.items()}.get(num, None)
        if key is None:
            continue
        merged.append({key: round(ttl_time, 3)})

    # ------------------------------------------------------------
    # 3. Sort all items by time (value inside the dict)
    # ------------------------------------------------------------
    merged_sorted = sorted(merged, key=lambda d: list(d.values())[0])

    return merged_sorted


# ==========================================================================================
# Function : plot_events_timeline
# Purpose  : display a graph that represent all the D, F, start_ & end_ from merged_ttl_tics
# ==========================================================================================

def plot_events_timeline(merged_ttl_tics, patient_name, phase_start_key=None, phase_end_key=None):

    t_start, t_end = None, None

    if phase_start_key is not None:
        if isinstance(phase_start_key, (int, float)):
            t_start = phase_start_key
        else:
            t_start = next((list(d.values())[0] for d in merged_ttl_tics if list(d.keys())[0] == phase_start_key), None)

    if phase_end_key is not None:
        if isinstance(phase_end_key, (int, float)):
            t_end = phase_end_key
        else:
            t_end = next((list(d.values())[0] for d in merged_ttl_tics if list(d.keys())[0] == phase_end_key), None)
    
    if t_start is not None and t_end is not None and t_end < t_start:
        raise ValueError(f"End time ({t_end}) is before start time ({t_start})")

    # filter the events to display
    filtered = []
    for d in merged_ttl_tics:
        key = list(d.keys())[0]
        value = list(d.values())[0]

        # keep only D, F, start_i, end_i
        if key in ["D", "F"] or re.match(r"start_\d+", key) or re.match(r"end_\d+", key):
            if (t_start is not None and value < t_start) or (t_end is not None and value > t_end):
                continue
            filtered.append(d)

    times = [list(d.values())[0] for d in filtered]
    labels = [list(d.keys())[0] for d in filtered]

    # for d in filtered:
    #     key = list(d.keys())[0]
    #     value = list(d.values())[0]
    #     times.append(value)
    #     labels.append(key)

    category_colors = {"D": "green", "F": "red", "start": "yellow", "end": "orange"}
    colors = []
    for l in labels:
        if l.startswith("start_"):
            colors.append(category_colors["start"])
        elif l.startswith("end_"):
            colors.append(category_colors["end"])
        else:
            colors.append(category_colors.get(l, "black"))  # default black
    
    y_positions = [0] * len(filtered)

    plt.figure(figsize=(14, 6))
    # plt.rcParams['scatter.marker'] = 'x'
    plt.scatter(times, y_positions, s=80, c=colors)
    plt.yticks([])
    plt.xlabel("Time (s)")
    plt.ylabel("")
    plt.title(f"Events timeline — {patient_name}")
    plt.grid(axis='x', linestyle='--', alpha=0.4)

    legend_patches = [
        mpatches.Patch(color="yellow", label="Start (start_i)"),
        mpatches.Patch(color="orange", label="End (end_i)"),
        mpatches.Patch(color="green",  label="D"),
        mpatches.Patch(color="red",    label="F")
    ]
    plt.legend(handles=legend_patches, loc="upper right")

    if t_start is not None and t_end is not None:
        plt.xlim(t_start - 0.1, t_end + 0.1) # petite marge

    plt.tight_layout()
    plt.show()

#########################################################################



# dictionnary to stock the results of all the patients
results = {}

patients = [
    {
        "montage": "standard_1005", # montage with 64 electrodes (from MM30 : always 64 electrodes)
        "vhdr": "C:\\Users\\indira.lavocat\\MOVIDOC\\PATIENT FILES\\EEG PATIENT FILES\\MOVIDOCTicTrack000030.vhdr", # MM30
        "excel": "C:\\Users\\indira.lavocat\\MOVIDOC\\PATIENT FILES\\EXCEL PATIENT FILES\\MM30_annotations_binary-table_cutted.xlsx",
        "fps": 30,
        "min_absence_frames": 30,
        "excel_phase_times": [6.633, 68.277, 196.581, 323.994, 931.194, 995.049]
    }
]

"""     {
        "montage": "standard_1020", # montage with 32 electrodes (from DS26 to BC29 : always 32 electrodes)
        "vhdr": "C:\\Users\\indira.lavocat\\MOVIDOC\\PATIENT FILES\\EEG PATIENT FILES\\MOVIDOCTicTrack_BB28-bis.vhdr", #BB28
        "excel": "C:\\Users\\indira.lavocat\\MOVIDOC\\PATIENT FILES\\EXCEL PATIENT FILES\\BB28_annotations_binary-table_cutted.xlsx",
        "fps": 25,
        "min_absence_frames": 25,
        "excel_phase_times": [9.840, 27.320, 156.880, 302.600, 929.960, 991.760]
    },
    {
        "montage": "standard_1020", # montage with 32 electrodes (from DS26 to BC29 : always 32 electrodes)
        "vhdr": "C:\\Users\\indira.lavocat\\MOVIDOC\\PATIENT FILES\\EEG PATIENT FILES\\MOVIDOCTicTrack000013.vhdr", # BC29
        "excel": "C:\\Users\\indira.lavocat\\MOVIDOC\\PATIENT FILES\\EXCEL PATIENT FILES\\BC29_annotations_binary-table_cutted_xlsx_Lizbeth.xlsx",
        "fps": 30,
        "min_absence_frames": 30,
        "excel_phase_times": [17.391, 65.670, 201.597, 358.215, 1088.670, 1204.038]
    },
        {
        "montage": "standard_1005", # montage with 64 electrodes (from MM30 : always 64 electrodes)
        "vhdr": "C:\\Users\\indira.lavocat\\MOVIDOC\\PATIENT FILES\\EEG PATIENT FILES\\MOVIDOCTicTrack000031.vhdr", # SC31
        "excel": "C:\\Users\\indira.lavocat\\MOVIDOC\\PATIENT FILES\\EXCEL PATIENT FILES\\SC31_annotations_binary-table_cutted.xlsx",
        "fps": 30,
        "min_absence_frames": 30,
        "excel_phase_times": [12.243, 62.040, 191.664, 345.972, 970.200, 1074.546]
    } """

# iterate on all the patients to execute the complete pipeline
for p in patients:
    try:
        results[p["vhdr"]] = run_full_pipeline_for_patient(
            vhdr_path=p["vhdr"], # path to the EEG file
            excel_path=p["excel"], # path to the Excel file
            # phase_start=p["phase_start"],
            # phase_end=p["phase_end"],
            fps=p["fps"], # fps of the Excel file
            min_absence_frames=p["min_absence_frames"], # minimum of frames of "Absence" to define the beginning & end of the tics,
            montage_name=p["montage"]
        )
    except Exception as e:
        print(f"\n❌ ERROR for patient: {p['vhdr']}")
        print(e)

print("Pipeline finished for all the patients.")


# print the result by patient
for vhdr_path, patient in results.items():
    print("\n" + "="*60)
    print(f" Patient : {patient['subject']}")
    print("="*60)

    print("\nTTLs :")
    pprint(patient["ttl"])

    # print("\nTics :")
    # for tic in patient["tics"]:
    #     print(f"start={tic['start']:.3f}  end={tic['end']:.3f}  phase={tic['phase']}")

    # print("\n--- Tics BEFORE EEG realignment (Excel original) ---")
    # for tic in patient["tics_original"]:
    #     print(f"start={tic['start']:.3f}  end={tic['end']:.3f}  phase={tic['phase']}")

    print("\n--- Tics AFTER EEG realignment (Excel corrected) ---")
    for tic in patient["tics_corrected"]:
        print(f"start={tic['start']:.3f}  end={tic['end']:.3f}  phase={tic['phase']}")
    
    print("\n--- Merged TTL + Tics timeline ---")
    for item in patient["merged_ttl_tics"]:
        print(item)
    
    # plot_events_timeline(
    #     merged_ttl_tics = patient["merged_ttl_tics"],
    #     patient_name = patient["subject"],
    #     phase_start_key=patient["phases_dict"]["spontaneous_tics"][0],
    #     phase_end_key=patient["phases_dict"]["spontaneous_tics"][1]
    #     )

print("Patients analysés :", list(results.keys()))

######################################################################### python _04_Movidoc_tictrack_merge_extractions_patients.py