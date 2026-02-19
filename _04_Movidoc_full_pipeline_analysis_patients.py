# =======================================================================
# File : _04_Movidoc_full_pipeline_analysis_patients.py
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
    apply_ICA,
    apply_rest_reference,
    recalibrate_from_first_event,
    collect_ttl_with_phases
)

from _03_Movidoc_tictrack_tic_extraction_patients import extract_tics_from_excel
from _05_Movidoc_tictrack_urge_extraction_function import (
    deduplicate_imitated_tics,
    analyse_merged_ttl_tics_imitated,
    deduplicate_suppressed_tics,
    analyse_merged_ttl_tics_suppressed,
)
from pprint import pprint

import mne
# from mne.time_frequency import psd_welch
# from mne.time_frequency.psd import psd_welch
from mne.time_frequency import psd_array_multitaper

import matplotlib.pyplot as plt
plt.ion()
import matplotlib.patches as mpatches
import re

import numpy as np

import os

import random


# Base directory for this project
base_dir = "/Users/Tysia/Desktop/movidoc/tictrack_eeg_analysis"
patient_files_dir = os.path.join(base_dir, "PATIENT FILES")
eeg_dir = os.path.join(patient_files_dir, "EEG PATIENT FILES")
excel_dir = os.path.join(patient_files_dir, "EXCEL PATIENT FILES")




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
        'end_ret': 14,
        'start_close': 5,
        'end_close': 6,
        'start_open': 7,
        'end_open': 8,
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


# ===============================================================================
# Function : run_full_pipeline_for_patient
# Purpose : run the full pipeline by running the functions from the 02 & 03 files
# ===============================================================================

def run_full_pipeline_for_patient(vhdr_path, excel_path, fps, min_absence_frames, montage_name, excel_phase_times):

    print(f"\n===== START Patient {vhdr_path} =====")
    
    # 1.a. Process EEG file (.vhdr)
    raw, subject_name = load_data(vhdr_path) # charge the EEG file & get the name of the subject
    print(f"Subject name: {subject_name}")
    # events_times, _ = extract_stimuli(raw) # extract the TTL/events from the signal BEFORE the recalage
    raw_pre, bad_channels = preprocess_data(raw, subject_name, montage_name=montage_name) # filter the signal & apply the montage
    raw_pre = apply_ICA(raw_pre, subject_name) # apply ICA to clean the signal from eye/muscle artifacts
    raw_rest = apply_rest_reference(raw_pre, subject_name) # apply the REST reference
    print("Before crop:", raw_rest.annotations.onset[:5])
    # print("Checkpoint 1 : EEG preprocessing OK")
    raw_cropped = recalibrate_from_first_event(raw_rest, target_stim="Stimulus/S  2") # readjust the signal from the 1st significative TTL
    events_times, event_id = extract_stimuli(raw_cropped) # extract the TTL/events from the signal AFTER the recalage
    print("After crop:", raw_cropped.annotations.onset[:5])

    # 1.b. Extract TTL information
    ttl_info = collect_ttl_with_phases(raw_cropped, subject_name) # get the TTL list & their phases

    # Get the time of Stimulus/S  2
    stim2_time_original = next((ttl["time"] for ttl in ttl_info if ttl["ttl_name"] == "Stimulus/S  2"), None)
    if stim2_time_original is None:
        raise ValueError("Stimulus/S  2 introuvable dans ttl_info. Impossible de recaler.")
    # Realign all the TTLs on the Stimulus/S  2
    for ttl in ttl_info:
        ttl["time"] = round(ttl["time"] - stim2_time_original, 3)
    


    # Extract automatically the timestamps of the TTLs used to realign the Excel times to the EEG times
    eeg_phase_times = extract_eeg_phase_times_from_ttl(ttl_info)
    print("\n===== EEG phase times (from TTL) =====")
    print(eeg_phase_times)

    # Realign Excel -> EEG
    excel_corrected_times, slope, intercept = realign_excel_to_eeg(excel_times=excel_phase_times, eeg_times=eeg_phase_times)
    print("\n===== Excel times realigned on EEG =====")
    print(excel_corrected_times)
    print(f"Linear drift: slope={slope:.6f}, intercept={intercept:.6f}")
    # print("Checkpoint 2 : TTL extracted and realigned OK")



    # define the phases via TTLs
    phases_ttl = {
        "press_key": {"start": "Stimulus/S  3", "end": "Stimulus/S  4"},
        "eyes_closed": {"start": "Stimulus/S  5", "end": "Stimulus/S  6"},
        "eyes_open": {"start": "Stimulus/S  7", "end": "Stimulus/S  8"},
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

    # print("Checkpoint 3 : phases_dict OK")

    # print the debug after the loop to check
    print("\n===== DEBUG: Phases readjusted timestamps for this patient =====")
    for phase, (s, e) in phases_dict.items():
        print(f"{phase:20s}  start={s:.3f}  end={e:.3f}")

    # 2. Extract tics from Excel
    tics_original = extract_tics_from_excel(excel_file=excel_path, fps=fps, min_absence_frames=min_absence_frames) # extract the tics from Excel
    # print("Checkpoint 4 : tics extracted from Excel")

    tics_original_with_phases = assign_phase_to_tics(tics_original , phases_dict) # associate each tic to its phase
    # print("Checkpoint 5 : tics extracted from Excel & phases assigned")

    # create a version realigned on EEG using the linear drift parameters ----
    tics_corrected = [(start * slope + intercept, end * slope + intercept) for start, end in tics_original]
    # print("Checkpoint 6 : tics extracted from Excel are corrected")

    # assign the phases after the recalage
    tics_corrected_with_phases = assign_phase_to_tics(tics_corrected , phases_dict) # associate each tic to its phase
    # print("Checkpoint 7 : tics extracted from Excel are corrected & phases assigned")

    # 3. Merge into one Python object
    full_output = {
        "subject": subject_name, # name of the subject
        "montage": montage_name,
        "excel_corrected_times": excel_corrected_times,
        "linear_drift_params": {"slope": slope, "intercept": intercept},
        "ttl": ttl_info, # list of the TTL with phases
        # "tics": tics_info # list of the tics with phases
        "tics_original": tics_original_with_phases,
        "tics_corrected": tics_corrected_with_phases,
        "phases_dict": phases_dict,
        "stim2_time_original": stim2_time_original,
        "raw_cropped": raw_cropped,
        "bad_channels": bad_channels
    }

    # full_output["phases_dict"] = phases_dict
    # full_output["stim2_time_original"] = stim2_time_original

    # 4. Build merged TTL + tics timeline
    phases_to_keep = ["spontaneous_tics", "imitated_tics", "retention_tics"]
    merged_ttl_tics = build_merged_ttl_tics(patient=full_output, phases_to_keep=phases_to_keep)
    full_output["merged_ttl_tics"] = merged_ttl_tics

    # print("Checkpoint 8 : ready to return full_output")

    # return the full object for the patient
    return full_output


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
    # plt.show() # need to close one graph to obtain the next one, and so on
    plt.show(block=False)
    plt.pause(0.1)
    plt.close()


#########################################################################


# ===================================================================================================================
# Function : analyse_merged_ttl_tics
# Purpose  : analyse merged_ttl_tics to display all the beginning of urges by displaying all the 'D' and/or 'start_i' 
# ===================================================================================================================

def analyse_merged_ttl_tics_spontaneous(merged_ttl_tics, phase_start_key='start_spont', phase_end_key='end_spont'):

    # Extract the indices of the start & end of the selected phase
    t_start = next((list(d.values())[0] for d in merged_ttl_tics if list(d.keys())[0] == phase_start_key), None)
    t_end   = next((list(d.values())[0] for d in merged_ttl_tics if list(d.keys())[0] == phase_end_key), None)

    # Create a filtered list of the selected phase
    phase_events = [d for d in merged_ttl_tics if t_start <= list(d.values())[0] <= t_end]

    # create the list that will contain all the times of beginning (start_i or D)
    results_list = []

    i = 0

    while i < len(phase_events):

        # key = list(phase_events[i].keys())[0]
        # value = list(phase_events[i].values())[0]
        event_dict = phase_events[i]
        key, value = next(iter(event_dict.items()))

        # Case : there is a 'D'
        if key == 'D':

            # Look before the D the first (from the D) start_i or end_i
            found_back = None
            for j in range(i-1, -1, -1):
                k, v = next(iter(phase_events[j].items()))
                if k.startswith('start_'):
                    found_back = ('start_i', v)
                    break
                elif k.startswith('end_'):
                    found_back = ('end_i', v)
                    break
                elif k == 'D':
                    found_back = ('D', v)
                    break
            
            # Case 1 : D begins (another D before)
            if found_back is not None and found_back[0] == 'D':
                print(f"D pressed at {value}")
                results_list.append({"type": "D_after_D", "D_time": value})

            # Case 2 : start before D
            if found_back is not None and found_back[0] == 'start_i':
                print(f"start_tic at {found_back[1]} then D pressed at {value}")
                results_list.append({"type": "start_then_D", "start_time": found_back[1]})

            # Case 3 : D begins (end_i before D)
            elif found_back is not None and found_back[0] == 'end_i':
                    
                # Look forward for first start_i or F
                found_forward = None
                for j_forward in range(i+1, len(phase_events)):
                    k_fwd, v_fwd = next(iter(phase_events[j_forward].items()))
                    if k_fwd.startswith('start_'):
                        found_forward = ('start_i', v_fwd)
                        break
                    elif k_fwd == 'F':
                        found_forward = ('F', v_fwd)
                        break

                if found_forward is not None:

                    # Case 3.a. : D then start
                    if found_forward[0] == 'start_i':
                        print(f"D pressed at {value} then visible tic at {found_forward[1]}")
                        results_list.append({"type": "D_then_start", "D_time": value})

                    # Case 3.b. : D then F, no visible tic
                    elif found_forward[0] == 'F':
                        print(f"D pressed at {value} without any visible tic then F pressed at {found_forward[1]}")
                        results_list.append({"type": "D_then_F","D_time": value})

            # Continue from the line after the D that has been analysed
            i += 1
        
        else:
            i += 1

    return results_list


#########################################################################

# ====================================================================================
# Function : shift_urges_times
# Purpose  : add the Stimulus/S  2 time to each urge timestamp to match the TTLs times
# ====================================================================================

def shift_urges_times(urges_list, stim2_time):

    """
    ----------
    Purpose
    ----------
    Add the EEG time of Stimulus/S  2 to each urge time.

    ----------
    Parameters
    ----------
    urges_list : list of dict
        Each dict has a 'start_time' or 'D_time'.
    stim2_time : float
        Time of Stimulus/S  2 in EEG (seconds).

    ----------
    Returns
    -------
    shifted_times : list of floats
        EEG times of urges.
    """

    shifted_times = []

    for u in urges_list:
        if 'start_time' in u:
            shifted_times.append(u['start_time'] + stim2_time)
        elif 'D_time' in u:
            shifted_times.append(u['D_time'] + stim2_time)
        elif 'S_time' in u:
            shifted_times.append(u['S_time'] + stim2_time)
        elif 'T_time' in u:
            shifted_times.append(u['T_time'] + stim2_time)
        else:
            raise ValueError(f"Unexpected dictionary structure: {u}")

    return shifted_times


# ==============================================================================
# Function : extract_random_epochs_in_phase
# Purpose  : extract random EEG epochs of fixed duration within a selected phase
# ==============================================================================

def extract_random_epochs_in_phase(raw_cropped, start_time, end_time, n_epochs=50, epoch_duration=2.5, event_id=1, seed=None):

    """
    ----------
    Purpose
    ----------
    Extract `n_epochs` random EEG segments of `epoch_duration` seconds within the interval defined by `ttl_start_name` and `ttl_end_name`.

    ----------
    Parameters
    ----------
    raw_cropped : mne.io.Raw
        Preprocessed EEG signal.
    ttl_start_name : str
        Name of the TTL marking the start of the interval (e.g. "Stimulus/S  5").
    ttl_end_name : str
        Name of the TTL marking the end of the interval (e.g. "Stimulus/S  6").
    ttl_list : list of dict
        List of TTLs, each dict containing "ttl_name" and "time".
    n_epochs : int
        Number of random epochs to extract.
    epoch_duration : float
        Duration of each epoch in seconds.

    ----------
    Returns
    ----------
    epochs : mne.Epochs
        MNE Epochs object containing the extracted random epochs.
    """

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

# =====================================================
# Function : extract_pre_tic_eeg_segments
# Purpose  : cut 1.5-second EEG segments before each urge
# =====================================================

def extract_pre_tic_epochs(raw_cropped, urge_times, pre_seconds=2.0, post_seconds=0.5):

    """
    ----------
    Purpose
    ----------
    Extract EEG segments of `pre_seconds` before each urge.

    ----------
    Parameters
    ----------
    raw : mne.io.Raw
        Preprocessed EEG signal.
    urge_times : list of float
        Times of urges in seconds.
    pre_seconds : float
        Duration to extract before each urge.

    ----------
    Returns
    ----------
    segments : list of mne.io.RawArray or Epochs
        EEG segments for each urge.
    """

    # segments = []
    # for t in urge_times:
    #     t_start = max(t - pre_seconds, 0)
    #     t_end = t
    #     segment = raw_cropped.copy().crop(tmin=t_start, tmax=t_end)
    #     segments.append(segment)
    # return segments

    events = []

    # convert urge_times → events array
    for t in urge_times:
        # sample = int(t * raw_cropped.info["sfreq"])
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


# =========================================================================================
# Function: full_pipeline_extract_pre_tic_epochs
# Purpose: run the full pipeline to extract the pre-tic epochs for a given patient and phase
# =========================================================================================
def full_pipeline_extract_pre_tic_epochs(patient, phase, nb_random_epochs):
    """Extract pre-tic epochs. Returns differ by phase:
    - spontaneous: (pre_tic_epochs, random_epochs, bad_channels)
    - imitated/suppressed: (pre_tic_category1, pre_tic_category2, random_epochs, bad_channels)
    
    Returns None for any epoch category that is empty.
    """
    results = run_full_pipeline_for_patient(
        vhdr_path=patient["vhdr"], excel_path=patient["excel"],
        fps=patient["fps"], min_absence_frames=patient["min_absence_frames"],
        montage_name=patient["montage"], excel_phase_times=patient["excel_phase_times"]
    )
    
    phase_map = {
        "spontaneous_phase": ("start_spont", "end_spont", analyse_merged_ttl_tics_spontaneous),
        "imitated_phase": ("start_imit", "end_imit", analyse_merged_ttl_tics_imitated),
        "suppressed_phase": ("start_ret", "end_ret", analyse_merged_ttl_tics_suppressed),
    }
    
    filter_dict = {
        'start_spont': 9, 'end_spont': 10, 'start_imit': 11, 'end_imit': 12,
        'start_ret': 13, 'end_ret': 14, 'start_open': 7, 'end_open': 8,
    }
    
    phase_start, phase_end, analyse_func = phase_map[phase]
    
    def make_epochs(urge_list, pre_sec=3.0, post_sec=2):
            if not urge_list:
                return None
            urges_shifted = shift_urges_times(urge_list, results["stim2_time_original"])
            if not urges_shifted:
                return None
            return extract_pre_tic_epochs(results["raw_cropped"], urges_shifted, pre_sec, post_sec)

    # Analyze tics
    analysis_result = analyse_func(
        merged_ttl_tics=results["merged_ttl_tics"],
        phase_start_key=phase_start,
        phase_end_key=phase_end
    )
    
    # Extract random epochs (common for all phases)
    ttl_info = results["ttl"]
    start_open = next(t["time"] for t in ttl_info if int(t["ttl_name"].split()[-1]) == filter_dict["start_open"])
    end_open = next(t["time"] for t in ttl_info if int(t["ttl_name"].split()[-1]) == filter_dict["end_open"])
    
    random_epochs = extract_random_epochs_in_phase(
        results["raw_cropped"], start_open, end_open,
        nb_random_epochs, epoch_duration=3,
        event_id=999, seed=42
    )
    
    # Phase-specific returns
    if phase == "spontaneous_phase":
        print("Processing spontaneous phase...")
        pre_tic_epochs = make_epochs(analysis_result, 3.0, 2.0)
        if pre_tic_epochs is None:
            print("WARNING: No spontaneous epochs found")
        return pre_tic_epochs, random_epochs, results["bad_channels"]
    
    else:
        # Imitated or suppressed (both return two lists)
        list1, list2 = analysis_result
        
        print(f"Processing {phase} - category 1...")
        epochs1 = make_epochs(list1)
        epochs2 = make_epochs(list2)
        if epochs1 is None:
            print(f"WARNING: No category 1 epochs found for {phase}")
        
        print(f"Processing {phase} - category 2 (real)...")
        epochs2 = make_epochs(list2)
        if epochs2 is None:
            print(f"WARNING: No category 2 (real) epochs found for {phase}")
        
        return epochs1, epochs2, random_epochs, results["bad_channels"]


# ===============================================================
# Function : compute_psd_per_channel_per_epoch
# Purpose  : compute PSD for each epoch and each selected channel
# ===============================================================

def compute_psd_per_channel_per_epoch(epochs, channels_to_use, fmin=1, fmax=40, bandwidth=3.0, normalize=True, method="minmax"):

    """
    Returns:
        psd_dict[epoch_idx][channel] = (freqs, psd_values)
    """

    # pick only the selected channels
    epochs_sel = epochs.copy().pick(channels_to_use)

    # compute PSD using multitaper method
    psd = epochs_sel.compute_psd(method='multitaper', fmin=fmin, fmax=fmax, bandwidth=bandwidth, n_jobs=1) # average=False suppressed
    # psds, freqs = mne.time_frequency.psd_multitaper(epochs_sel, fmin=1, fmax=40, bandwidth=3.0, n_jobs=1, average=None)
    psds, freqs = psd.get_data(return_freqs=True) # shape → (n_epochs, n_channels, n_freqs)

    # normalization
    if normalize:

        if method == "minmax":
            psds_min = psds.min(axis=2, keepdims=True)
            psds_max = psds.max(axis=2, keepdims=True)
            psds = (psds - psds_min) / (psds_max - psds_min + 1e-12)

        elif method == "zscore":
            mean = psds.mean(axis=2, keepdims=True)
            std  = psds.std(axis=2, keepdims=True)
            psds = (psds - mean) / (std + 1e-12)

        elif method == "unit_energy":
            total = psds.sum(axis=2, keepdims=True)
            psds = psds / (total + 1e-12)

        else:
            raise ValueError(f"Unknown normalization method '{method}'")

    psd_dict = {}

    for ep in range(len(epochs_sel)):
        psd_dict[ep] = {}
        for c_idx, ch_name in enumerate(epochs_sel.ch_names):
            psd_dict[ep][ch_name] = (freqs, psds[ep, c_idx, :])

    return psd_dict


# ==============================================================
# Function : plot_psd_per_channel
# Purpose  : display PSD per channel among the selected channels
# ==============================================================

def plot_psd_per_channel(psd_dict, epoch_idx):

    """
    Plot PSD for each channel for a given epoch index.
    """

    plt.figure(figsize=(8, 5))
    for ch, (freqs, psd) in psd_dict[epoch_idx].items():
        plt.plot(freqs, psd, label=ch)

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density")
    plt.title(f"PSD per channel — Epoch {epoch_idx}")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ===========================================================
# Function : save_psd_per_channel
# Purpose  : save PSD per channel among the selected channels
# ===========================================================

def save_psd_per_channel(psd_dict, subject_name, epoch_type):

    """
    -----------
    Parameters:
    -----------
    psd_dict : dict
        PSD per epoch and per channel.
    subject_name : str
        Name of the patient.
    epoch_type : str
        "random_epochs" ou "epochs_pre_tic"
    """

    # create a folder for each patient and epochs
    folder = os.path.join(psd_img_dir, f"{subject_name}_{epoch_type}")
    os.makedirs(folder, exist_ok=True)
    
    for ep_idx, ch_dict in psd_dict.items():
        for ch_name, (freqs, psd_vals) in ch_dict.items():
            plt.figure(figsize=(6,4))
            plt.plot(freqs, psd_vals)
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("PSD")
            plt.title(f"PSD {ch_name} - Epoch {ep_idx}")
            plt.tight_layout()
            
            # Name of the file
            filename = f"PSD_{ch_name}_{ep_idx}_{epoch_type}_{subject_name}.png"
            plt.savefig(os.path.join(folder, filename))
            plt.close()


# ================================================================
# Function : compute_mean_psd_ROI_per_epoch
# Purpose  : average PSD across selected channels (ROI), per epoch
# ================================================================

def compute_mean_psd_ROI_per_epoch(epochs, channels_to_use, fmin=1, fmax=40, bandwidth=3.0, normalize=True, method="minmax"):
    """
    Returns:
        mean_psd_dict[epoch_idx] = (freqs, mean_psd_across_channels)
    """

    epochs_sel = epochs.copy().pick(channels_to_use)

    # Compute PSD with multitaper method
    psd = epochs_sel.compute_psd(method='multitaper', fmin=fmin, fmax=fmax, bandwidth=bandwidth, n_jobs=1)
    psds, freqs = psd.get_data(return_freqs=True)  # shape = (n_epochs, n_channels, n_freqs)
    # psds, freqs = mne.time_frequency.psd_welch(epochs_sel, fmin=fmin, fmax=fmax, n_fft=512, average=None)
    # psds shape → (n_epochs, n_channels, n_freqs)

    # normalization
    if normalize:

        if method == "minmax":
            psds_min = psds.min(axis=2, keepdims=True)
            psds_max = psds.max(axis=2, keepdims=True)
            psds = (psds - psds_min) / (psds_max - psds_min + 1e-12)

        elif method == "zscore":
            mean = psds.mean(axis=2, keepdims=True)
            std  = psds.std(axis=2, keepdims=True)
            psds = (psds - mean) / (std + 1e-12)

        elif method == "unit_energy":
            total = psds.sum(axis=2, keepdims=True)
            psds = psds / (total + 1e-12)

        else:
            raise ValueError(f"Unknown normalization method '{method}'")

    mean_psd_dict = {}

    for ep in range(len(epochs_sel)):
        mean_psd = psds[ep].mean(axis=0)  # mean across channels
        mean_psd_dict[ep] = (freqs, mean_psd)

    return mean_psd_dict


# ==========================================================
# Function : plot_mean_psd_ROI
# Purpose  : display PSD per group of selected channel (ROI)
# ==========================================================

def plot_mean_psd_ROI(mean_psd_dict, epoch_idx):

    """
    Plot mean PSD across channels for one ROI for a given epoch index.
    """

    freqs, psd_mean = mean_psd_dict[epoch_idx]

    plt.figure(figsize=(8, 5))
    plt.plot(freqs, psd_mean)

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density")
    plt.title(f"Mean PSD (ROI) — Epoch {epoch_idx}")
    plt.tight_layout()
    plt.show()


# ==========================================================
# Function : save_mean_psd_ROI
# Purpose  : save PSD per group of selected channel (ROI)
# ==========================================================

def save_mean_psd_ROI(mean_psd_dict, output_folder, subject_name, epoch_type):

    """
    -----------
    Parameters:
    -----------
    mean_psd_dict : dict
        PSD moyennes par epoch et ROI.
    subject_name : str
        Nom du patient.
    epoch_type : str
        "random_epochs" ou "epochs_pre_tic"
    """

    # folder = os.path.join(psd_img_dir, f"{subject_name}_{epoch_type}_ROI")
    os.makedirs(output_folder, exist_ok=True)
    
    for roi_name, (freqs, psd_vals) in mean_psd_dict.items():
        save_path = os.path.join(
            output_folder,
            f"{subject}_{epoch_type}_{roi_name}_meanPSD.npy"
        )
        np.save(save_path, {"freqs": freqs, "psd": psd_vals})
        print(f"Saved: {save_path}")


#########################################################################


# ===============================================================
# Function : average_epochs_per_roi
# Purpose  : calculate the mean of the epochs of 1 patient by ROI
# ===============================================================

def average_epochs_per_roi(epochs, roi_lists):

    """
    epochs : mne.Epochs
    roi_lists : dict, ROI_name -> list des canaux
    """

    avg_per_roi = {}

    for roi_name, ch_names in roi_lists.items():

        # Select the channels of the ROI
        data = epochs.copy().pick_channels(ch_names).get_data()  # shape = (n_epochs, n_channels, n_times)

        # calculate the mean of all the epochs & all the channels of the ROI
        mean_signal = data.mean(axis=(0, 1))  # shape = (n_times,)
        avg_per_roi[roi_name] = mean_signal

    return avg_per_roi


# =======================================================
# Function : compute_psd_from_signal
# Purpose  : calculate the PSD of the mean epochs per ROI
# =======================================================

def compute_psd_from_signal(signal, sfreq, fmin=1, fmax=40):

    """
    signal : ndarray (n_times,)
    sfreq : fréquence d'échantillonnage
    """

    # info = mne.create_info(ch_names=["avg"], sfreq=sfreq, ch_types=["eeg"])
    # raw = mne.io.RawArray(signal[np.newaxis, :], info)
    # psd = raw.compute_psd(method="multitaper", fmin=fmin, fmax=fmax)
    # psds, freqs = psd.get_data(return_freqs=True)
    psds, freqs = psd_array_multitaper(signal, sfreq=sfreq, fmin=fmin, fmax=fmax, adaptive=True, normalization='full', verbose=False)

    # return freqs, psds[0, 0, :]  # fréquence et PSD
    return freqs, psds


# ================================================================================
# Function : group_average_psd
# Purpose  : calculate the mean of all the patients for 1 "type" of epochs per ROI
# ================================================================================

def group_average_psd(patient_epochs_dict, roi_lists, fmin=1, fmax=40):

    """
    patient_epochs_dict : dict patient_name -> epochs (MNE Epochs)
    roi_lists : dict ROI_name -> canaux
    """

    roi_signals_all = {roi: [] for roi in roi_lists.keys()}

    # 1. Mean by patient for each ROI ---

    # Mean by patient
    for patient_name, epochs in patient_epochs_dict.items():
        avg_per_roi = average_epochs_per_roi(epochs, roi_lists)
        for roi, signal in avg_per_roi.items():
            roi_signals_all[roi].append(signal)
    

    # 2. Mean of all the patients

    roi_group_psd = {}

    for roi, signals in roi_signals_all.items():
        # signals = list of mean signals (one by patient)
        if len(signals) == 0:
            continue

        # group mean of the temporal signal
        group_avg_signal = np.mean(signals, axis=0)

        # calculate the PSD
        sfreq = epochs.info["sfreq"]
        freqs, psd_vals = compute_psd_from_signal(group_avg_signal, sfreq, fmin=fmin, fmax=fmax)

        roi_group_psd[roi] = (freqs, psd_vals)

    return roi_group_psd


# ===============================================================
# Function : compute_group_psd
# Purpose  : generate the final PSD per ROI for all the patients
# ===============================================================

def compute_group_psd(roi_group_avg, sfreq, fmin=1, fmax=40):

    roi_psd = {}

    for roi, signal in roi_group_avg.items():
        freqs, psd = compute_psd_from_signal(signal, sfreq, fmin, fmax)
        roi_psd[roi] = (freqs, psd)

    return roi_psd


# ===========================================================
# Function : save_group_psd
# Purpose  : save the final PSDs per ROI for all the patients
# ===========================================================

def save_group_psd(roi_psd_dict, out_dir, tag):

    os.makedirs(out_dir, exist_ok=True)

    for roi_name, (freqs, psd_vals) in roi_psd_dict.items():
        plt.figure(figsize=(6,4))
        plt.plot(freqs, psd_vals)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("PSD")
        plt.title(f"Group PSD ROI {roi_name} - {tag}")
        plt.tight_layout()

        fname = os.path.join(out_dir, f"group_PSD_{roi_name}_{tag}.png")

        plt.savefig(fname)
        plt.close()


##############################################################

'''

# dictionnary to stock the results of all the patients
results = {}

patients = [
    {
        "montage": "standard_1020", # montage with 32 electrodes (from DS26 to BC29 : always 32 electrodes)
        "vhdr": os.path.join(eeg_dir, "MOVIDOCTicTrack000010.vhdr"),
        "excel": os.path.join(excel_dir, "DS26_annotations_binary-table_cutted.xlsx"),
        "fps": 30,
        "min_absence_frames": 30,
        "excel_phase_times": [13.728, 50.028, 190.080, 349.272, 980.727, 1277.991]
    },
    {
        "montage": "standard_1020", # montage with 32 electrodes (from DS26 to BC29 : always 32 electrodes)
        "vhdr": os.path.join(eeg_dir, "MOVIDOCTicTrack_BB28-bis.vhdr"), #BB28
        "excel": os.path.join(excel_dir, "BB28_annotations_binary-table_cutted.xlsx"),
        "fps": 25,
        "min_absence_frames": 25,
        "excel_phase_times": [9.840, 27.320, 156.880, 302.600, 929.960, 991.760]
    },
    {
        "montage": "standard_1020", # montage with 32 electrodes (from DS26 to BC29 : always 32 electrodes)
        "vhdr": os.path.join(eeg_dir, "MOVIDOCTicTrack000013.vhdr"), # BC29
        "excel": os.path.join(excel_dir, "BC29_annotations_binary-table_cutted_xlsx_Lizbeth.xlsx"),
        "fps": 30,
        "min_absence_frames": 30,
        "excel_phase_times": [17.391, 65.670, 201.597, 358.215, 1088.670, 1204.038]
    },
    {
        "montage": "standard_1005", # montage with 64 electrodes (from MM30 : always 64 electrodes)
        "vhdr": os.path.join(eeg_dir, "MOVIDOCTicTrack000030.vhdr"), # MM30
        "excel": os.path.join(excel_dir, "MM30_annotations_binary-table_cutted.xlsx"),
        "fps": 30,
        "min_absence_frames": 30,
        "excel_phase_times": [6.633, 68.277, 196.581, 323.994, 931.194, 995.049]
    },
    {
        "montage": "standard_1005", # montage with 64 electrodes (from MM30 : always 64 electrodes)
        "vhdr": os.path.join(eeg_dir, "MOVIDOCTicTrack000031.vhdr"), # SC31
        "excel": os.path.join(excel_dir, "SC31_annotations_binary-table_cutted.xlsx"),
        "fps": 30,
        "min_absence_frames": 30,
        "excel_phase_times": [12.243, 62.040, 191.664, 345.972, 970.200, 1074.546]
    }
]

# iterate on all the patients to execute the complete pipeline
for p in patients:
    try:
        results[p["vhdr"]] = run_full_pipeline_for_patient(
            vhdr_path=p["vhdr"], # path to the EEG file
            excel_path=p["excel"], # path to the Excel file
            fps=p["fps"], # fps of the Excel file
            min_absence_frames=p["min_absence_frames"], # minimum of frames of "Absence" to define the beginning & end of the tics,
            montage_name=p["montage"],
            excel_phase_times=p["excel_phase_times"]
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

    # print("\nTTLs :")
    # pprint(patient["ttl"])

    # print("\nTics :")
    # for tic in patient["tics"]:
    #     print(f"start={tic['start']:.3f}  end={tic['end']:.3f}  phase={tic['phase']}")

    # print("\n--- Tics BEFORE EEG realignment (Excel original) ---")
    # for tic in patient["tics_original"]:
    #     print(f"start={tic['start']:.3f}  end={tic['end']:.3f}  phase={tic['phase']}")

    # print("\n--- Tics AFTER EEG realignment (Excel corrected) ---")
    # for tic in patient["tics_corrected"]:
    #     print(f"start={tic['start']:.3f}  end={tic['end']:.3f}  phase={tic['phase']}")
    
    # print("\n--- Merged TTL + Tics timeline ---")
    # for item in patient["merged_ttl_tics"]:
    #     print(item)
    
    plot_events_timeline(
        merged_ttl_tics = patient["merged_ttl_tics"],
        patient_name = patient["subject"],
        phase_start_key=patient["phases_dict"]["spontaneous_tics"][0],
        phase_end_key=patient["phases_dict"]["spontaneous_tics"][1]
        )

print("Patients analysés :", list(results.keys()))

#########################################################################



# Creating lists of the urges by patient

print("\n===== Analyse des merged_ttl_tics par patient =====\n")

urges_lists = {}

for vhdr_path, patient in results.items():

    subject = patient['subject']

    print("\n" + "="*60)
    print(f" Patient : {patient['subject']}")
    print("="*60)
    
    analysis_list = analyse_merged_ttl_tics(
        merged_ttl_tics=patient["merged_ttl_tics"],
        phase_start_key='start_spont',  # or patient['phases_dict']['spontaneous_tics'][0] if we want the exact timestamp
        phase_end_key='end_spont'       # or patient['phases_dict']['spontaneous_tics'][1] if we want the exact timestamp
    )

    # Save the lists
    urges_lists[f"list_urges_{subject}"] = analysis_list
    print("Saved as:", f"list_urges_{subject}")
    print(analysis_list)

#########################################################################



# Extracting eyes_closed & urges epochs for all patients --- AND --- Compute the PSDs of each epoch of all the patients

# To save the PSDs
psd_img_dir = "psd_images"
os.makedirs(psd_img_dir, exist_ok=True)

# create dictionnaries to stock the epochs for each patient -> 1 entry by patient pre_urge_epochs["000031"]=epochs_MNE
pre_urge_epochs = {}
random_epochs_in_phase = {}
results_psd = {}
mean_psd_rois_all_patients = {}
all_random_epochs = {}
all_pre_tic_epochs = {}

channels_to_use_32 = ["Cz", "C3", "C4", "Pz", "Fp1", "Fp2"]
channels_to_use_64 = ["Cz", "FCz", "C3", "FC3", "CP3", "C4", "CP4", "FC4", "Pz", "CPz", "Fp1", "Fp2", "AF3", "AFz", "AF4"]

roi_lists_32 = {
    "midline_premotor": ["Cz"],
    "left_sensorimotor": ["C3"],
    "right_sensorimotor": ["C4"],
    "midline_posterior": ["Pz"],
    "midline_prefrontal": ["Fp1", "Fp2"]
}
roi_lists_64 = {
    "midline_premotor": ["Cz", "FCz"],
    "left_sensorimotor": ["C3", "FC3", "CP3"],
    "right_sensorimotor": ["C4", "CP4", "FC4"],
    "midline_posterior": ["Pz", "CPz"],
    "midline_prefrontal": ["Fp1", "Fp2", "AF3", "AFz", "AF4"]
}


for vhdr_path, patient in results.items():
    subject = patient['subject']
    print(f"\nProcessing patient {subject}")

    # Choice of the ROI depending on the patient's montaget
    if patient["montage"] == "standard_1020":
        channels_to_use = channels_to_use_32
        roi_lists = roi_lists_32
    elif patient["montage"] == "standard_1005":
        channels_to_use = channels_to_use_64
        roi_lists = roi_lists_64
    else:
        raise ValueError(f"Montage inconnu pour le patient {subject}: {patient['montage']}")


    # A. Extract random epochs from the eyes_closed phase

    raw_cropped = patient["raw_cropped"]
    ttl_info = patient["ttl"]

    # 1️⃣ Get the Stimulus/S that fix the limits of the eyes_closed phase
    start_time = next(ttl["time"] for ttl in ttl_info if ttl["ttl_name"] == "Stimulus/S  5")
    end_time   = next(ttl["time"] for ttl in ttl_info if ttl["ttl_name"] == "Stimulus/S  6")

    # 2️⃣ Extract the random epochs in the selected phase
    random_epochs = extract_random_epochs_in_phase(
        raw_cropped=patient["raw_cropped"],
        start_time=start_time,
        end_time=end_time,
        n_epochs=50,
        epoch_duration=2.5,
        event_id=999,
        seed=42
    )

    # 3️⃣ Save epochs into a .fif file (one file per patient)
    os.makedirs("epochs_random_eyes_closed", exist_ok=True)
    random_epochs.save(f"epochs_random_eyes_closed/random_epochs_{subject}.fif", overwrite=True)


    # B. Extract the urge epochs

    # 1️⃣ Shift urges times into EEG reference
    urges_list = urges_lists[f"list_urges_{subject}"] # get the list of urges of the patient
    stim2_time_patient = patient["stim2_time_original"] # get the original timestamp of Stimulus/S  2
    urges_times_eeg = shift_urges_times(urges_list, stim2_time=stim2_time_patient) # convert the urges times back to the original EEG referencial

    # 2️⃣ Extract EEG epochs (2s before each urge)
    epochs_pre_tic = extract_pre_tic_epochs(raw_cropped, urges_times_eeg, pre_seconds=2.0, post_seconds=0.5)

    # 3️⃣ Save epochs into a .fif file (one file per patient)
    os.makedirs("epochs_pre_urge", exist_ok=True)
    epochs_pre_tic.save(f"epochs_pre_urge/pre_urge_epochs_{subject}.fif", overwrite=True)


    # C. Compute the PSDs on each epoch of the patient

    psd_random = compute_psd_per_channel_per_epoch(random_epochs, channels_to_use, fmin=1, fmax=40)
    mean_psd_random = {}
    for roi_name, roi_channels in roi_lists.items():
        mean_psd_random[roi_name] = compute_mean_psd_ROI_per_epoch(random_epochs, roi_channels, fmin=1, fmax=40)

    psd_pre_tic = compute_psd_per_channel_per_epoch(epochs_pre_tic, channels_to_use, fmin=1, fmax=40)
    mean_psd_pre_tic = {}
    for roi_name, roi_channels in roi_lists.items():
        mean_psd_pre_tic[roi_name] = compute_mean_psd_ROI_per_epoch(epochs_pre_tic, roi_channels, fmin=1, fmax=40)
    
    # Save the PSDs for random_epochs
    # save_psd_per_channel(psd_random, subject_name=subject, epoch_type="random_epochs")
    # save_mean_psd_ROI(mean_psd_random, subject_name=subject, epoch_type="random_epochs")

    # Save the PSDs for pre-tic (urges) epochs
    # save_psd_per_channel(psd_pre_tic, subject_name=subject, epoch_type="epochs_pre_tic")
    # save_mean_psd_ROI(mean_psd_pre_tic, subject_name=subject, epoch_type="epochs_pre_tic")


    # OPTIONAL - D. Display the PSDs for all the epochs of the patient

    # print(f"Plotting PSDs for random epochs ({len(random_epochs)} epochs)")
    # for ep in range(len(random_epochs)):
    #     plot_psd_per_channel(psd_random, epoch_idx=ep)
    #     for roi_name in mean_psd_random:
    #         plot_mean_psd_ROI(mean_psd_random[roi_name], epoch_idx=ep)
    
    # print(f"Plotting PSDs for pre-tic epochs ({len(epochs_pre_tic)} epochs)")
    # for ep in range(len(epochs_pre_tic)):
    #     plot_psd_per_channel(psd_pre_tic, epoch_idx=ep)
    #     for roi_name in mean_psd_random:
    #         plot_mean_psd_ROI(mean_psd_random[roi_name], epoch_idx=ep)
    

    # E. Store the results

    results_psd[subject] = {
        "random_epochs": random_epochs,
        "epochs_pre_tics": epochs_pre_tic,
        "psd_random": psd_random,
        "mean_psd_random": mean_psd_random,
        "psd_pre_tic": psd_pre_tic,
        "mean_psd_pre_tic": mean_psd_pre_tic,
        "n_epochs_pre_tic": len(epochs_pre_tic),
        "n_epochs_random": len(random_epochs)
    }
# print("\n=== epochs extraction + PSDs generation complete for all the patients ===")


# prepare the containers per ROI (we'll use the name of the ROIs defined in roi_lists_32 & roi_lists_64)
roi_names_32 = list(roi_lists_32.keys())
roi_names_64 = list(roi_lists_64.keys())

# stock the mean signals per ROI for both random_epochs_in_phase & pre_urge_epoch
roi_signals_random = {roi: [] for roi in roi_lists_32.keys()}  # les noms de ROI sont les mêmes pour 32/64
roi_signals_pre = {roi: [] for roi in roi_lists_32.keys()}

group_sfreq = None


for subject, info in results_psd.items():

    epochs_random = info["random_epochs"]
    epochs_pre = info["epochs_pre_tics"]

    patient_obj = None

    for pth, p in results.items():
        if p["subject"] == subject:
            patient_obj = p
            break
    if patient_obj is None:
        print("Warning: patient object not found for", subject)
        continue

    if patient_obj["montage"] == "standard_1020":
        rois_for_patient = roi_lists_32
    else:
        rois_for_patient = roi_lists_64
    
    print(f"ROIs pour ce patient: {list(rois_for_patient.keys())}")
    for roi, chans in rois_for_patient.items():
        print(f"{roi} -> {chans}")

    # save sfreq if empty
    if group_sfreq is None:
        group_sfreq = epochs_random.info["sfreq"]

    # 1) temporal mean per ROI for THIS patient (random epochs)
    avg_per_roi_random = average_epochs_per_roi(epochs_random, rois_for_patient)
    for roi, signal in avg_per_roi_random.items():
        # si signal est None ou vide -> skip
        if signal is None:
            continue
        roi_signals_random[roi].append(signal)
    # DEBUG: vérifier la forme et les valeurs des signaux
    # print(f"\n--- Random epochs --- Patient: {subject}")
    # for roi, signal in avg_per_roi_random.items():
    #     print(f"ROI: {roi}, signal shape: {signal.shape}, first 5 samples: {signal[:5]}")

    # 2) temporal mean per ROI for THIS patient (pre-urge epochs)
    avg_per_roi_pre = average_epochs_per_roi(epochs_pre, rois_for_patient)
    for roi, signal in avg_per_roi_pre.items():
        if signal is None:
            continue
        # DEBUG
        # print(f"\n\n\n Patient {subject}, ROI {roi}, id(signal): {id(signal)} \n")
        roi_signals_pre[roi].append(signal)
    # DEBUG: vérifier la forme et les valeurs des signaux
    # print(f"\n--- Pre-urge epochs --- Patient: {subject}")
    # for roi, signal in avg_per_roi_pre.items():
    #     print(f"ROI: {roi}, signal shape: {signal.shape}, first 5 samples: {signal[:5]}")

# 3) inter-patients mean 1 calcul of the PSD  of the mean signal of the group
roi_group_psd_random = {}
roi_group_psd_pre = {}

# DEBUG
# print(f"\n\n\n ROI signals random (roi_signals_random) : {roi_signals_random} \n\n\n")
# print(f"\n\n\n ROI signals random (roi_signals_pre) : {roi_signals_pre} \n\n\n")

for roi in roi_signals_random.keys():
    sigs = roi_signals_random[roi]
    if len(sigs) == 0:
        print(f"No signals for ROI {roi} in random_epochs -> skipping")
        continue
    group_avg_signal = np.mean(np.stack(sigs, axis=0), axis=0)
    freqs, psd_vals = compute_psd_from_signal(group_avg_signal, sfreq=group_sfreq, fmin=1, fmax=40)
    # DEBUG
    # print(f"For ROI : {roi} in Random : \nFreqs : {freqs}\nValue : {psd_vals}")
    roi_group_psd_random[roi] = (freqs, psd_vals)

for roi in roi_signals_pre.keys():
    sigs = roi_signals_pre[roi]
    if len(sigs) == 0:
        print(f"No signals for ROI {roi} in pre_urge_epochs -> skipping")
        continue
    group_avg_signal = np.mean(np.stack(sigs, axis=0), axis=0)
    freqs, psd_vals = compute_psd_from_signal(group_avg_signal, sfreq=group_sfreq, fmin=1, fmax=40)
    # DEBUG
    print(f"For ROI : {roi} in PRE : \nFreqs : {freqs}\nValue : {psd_vals}")
    roi_group_psd_pre[roi] = (freqs, psd_vals)

# 4) Save the final PSDs
out_dir = "group_psd"
save_group_psd(roi_group_psd_random, out_dir, tag="random_epochs_in_phase")
save_group_psd(roi_group_psd_pre, out_dir, tag="pre_urge_epochs")

np.save("roi_group_psd_random.npy", roi_group_psd_random, allow_pickle=True)
np.save("roi_group_psd_pre.npy", roi_group_psd_pre, allow_pickle=True)

# 5) Optionnal: save the PSD data into a numpy (.npz) to reuse it later if needed
# np.savez(os.path.join(out_dir, "group_psd_random.npz"), **{r: np.array(v[1]) for r, v in roi_group_psd_random.items()})
# np.savez(os.path.join(out_dir, "group_psd_pre.npz"), **{r: np.array(v[1]) for r, v in roi_group_psd_pre.items()})

print("\n✅ Group PSDs computed and saved in", out_dir)

#########################################################################


# Generation of the 5 final graphs (PSDs)

roi_list = [
    "left_sensorimotor",
    "right_sensorimotor",
    "midline_premotor",
    "midline_prefrontal",
    "midline_posterior"
]

group_psd_dir = base_dir

path_pre = os.path.join(group_psd_dir, f"roi_group_psd_pre.npy")
path_rand = os.path.join(group_psd_dir, f"roi_group_psd_random.npy")
files_pre = np.load(path_pre, allow_pickle=True).item()
files_rand = np.load(path_rand, allow_pickle=True).item()


psd_output_dir = os.path.join(base_dir, "final_5_PSD")
os.makedirs(psd_output_dir, exist_ok=True)

epsilon = 1e-14

for roi in roi_list:
    # DEBUG
    # print(f"Checking ROI {roi}...")
    if roi not in files_pre or roi not in files_rand:
        print(f"ROI {roi} not found in the files, skipping.")
        continue
    
    freqs_rand, psd_rand = files_rand[roi]
    freqs_pre, psd_pre = files_pre[roi]
    # DEBUG
    # print(f"\n\n\n For ROI: {roi}, \n psd_pre: {psd_pre} \n")

    # normalization step between 1 & 30 Hz
    mask_pre = freqs_pre <= 30
    mask_rand = freqs_rand <= 30
    min_pre, max_pre = psd_pre[mask_pre].min(), psd_pre[mask_pre].max()
    min_rand, max_rand = psd_rand[mask_rand].min(), psd_rand[mask_rand].max()

    # DEBUG
    print(f"\n ROI {roi} : PSD min = {psd_pre.min()}, PSD max = {psd_pre.max()}, PSD mean={psd_pre.mean()}")

    psd_rand_norm = (psd_rand - min_rand) / (max_rand - min_rand)
    # psd_rand_norm = (psd_rand - psd_rand.min()) / (psd_rand.max() - psd_rand.min())
    psd_pre_norm = (psd_pre - min_pre) / (max_pre - min_pre)
    # psd_pre_norm = (psd_pre - psd_pre.min()) / (psd_pre.max() - psd_pre.min())

    # Avoid the 0 for log
    psd_rand_norm = np.clip(psd_rand_norm, epsilon, None)
    psd_pre_norm = np.clip(psd_pre_norm, epsilon, None)

    # DEBUG
    print(f"\n\n\n For ROI: {roi}, \n psd_pre_norm: {psd_pre_norm} \n")

    plt.figure(figsize=(10, 5))
    plt.plot(freqs_rand, psd_rand_norm, label='Random epochs (norm.)', color='orange')
    plt.plot(freqs_pre, psd_pre_norm, label='Pre-urge epochs (norm.)', color='purple')
    
    plt.title(f"PSD Comparison for ROI: {roi}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Normalized PSD (0-1)")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Code if log scales wanted
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.ylim(0, 1)

    # Save figure to file
    plt.savefig(os.path.join(psd_output_dir, f"normalized_PSD_1-30Hz_{roi}_2s.png"))
    plt.savefig(os.path.join(psd_output_dir, f"normalized_PSD_1-30Hz_{roi}_2s.eps"), format='eps')
    plt.close("all")



# Test 1 patient by 1 patient

# roi_1 = "left_sensorimotor"
# roi_2 = "right_sensorimotor"
# roi_3 = "midline_premotor"
# roi_4 = "midline_prefrontal"
# roi_5 = "midline_posterior"

# roi = roi_5

# freqs_rand, psd_rand = files_rand[roi]
# freqs_pre, psd_pre = files_pre[roi]

# # normalization step between 1 & 30 Hz
# mask_pre = freqs_pre <= 30
# mask_rand = freqs_rand <= 30
# min_pre, max_pre = psd_pre[mask_pre].min(), psd_pre[mask_pre].max()
# min_rand, max_rand = psd_rand[mask_rand].min(), psd_rand[mask_rand].max()

# psd_rand_norm = (psd_rand - min_rand) / (max_rand - min_rand)
# # psd_rand_norm = (psd_rand - psd_rand.min()) / (psd_rand.max() - psd_rand.min())
# psd_pre_norm = (psd_pre - min_pre) / (max_pre - min_pre)
# # psd_pre_norm = (psd_pre - psd_pre.min()) / (psd_pre.max() - psd_pre.min())

# # Avoid the 0 for log
# psd_rand_norm = np.clip(psd_rand_norm, epsilon, None)
# psd_pre_norm = np.clip(psd_pre_norm, epsilon, None)

# plt.figure(figsize=(10, 5))
# plt.plot(freqs_rand, psd_rand_norm, label='Random epochs (norm.)', color='orange')
# plt.plot(freqs_pre, psd_pre_norm, label='Pre-urge epochs (norm.)', color='purple')

# plt.title(f"PSD Comparison for ROI: {roi}")
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("Normalized PSD (0-1)")
# plt.legend()
# plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# # Save figure to file
# plt.savefig(os.path.join(psd_output_dir, f"normalized_PSD_1-30Hz_{roi}_2s_ROI-5.png"))
# plt.savefig(os.path.join(psd_output_dir, f"normalized_PSD_1-30Hz_{roi}_2s_ROI-5.eps"), format='eps')
# plt.close("all")
'''