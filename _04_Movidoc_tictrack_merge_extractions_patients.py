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



# ============================================================
# Define the function
# ============================================================

def run_full_pipeline_for_patient(vhdr_path, excel_path, phase_start, phase_end, fps, min_absence_frames):
    
    # 1. Process EEG file (.vhdr)
    raw, subject_name = load_data(vhdr_path)
    events_times, _ = extract_stimuli(raw)
    raw_pre = preprocess_data(raw, subject_name)
    raw_rest = apply_rest_reference(raw_pre, subject_name)
    raw_cropped = recalibrate_from_first_event(raw_rest, events_times)

    # 1b. Extract TTL information
    ttl_info = collect_ttl_with_phases(raw_cropped, subject_name)

    # 2. Extract tics from Excel
    tics = extract_tics_from_excel(
        excel_file=excel_path,
        phase_start_s=phase_start,
        phase_end_s=phase_end,
        fps=fps,
        min_absence_frames=min_absence_frames
    )

    tics_info = [{"start": float(s), "end": float(e)} for s, e in tics]

    # 3. Merge into one Python object
    full_output = {
        "subject": subject_name,
        "ttl": ttl_info,
        "tics": tics_info
    }

    return full_output

#########################################################################



results = {}

patients = [
    {
        "vhdr": "C:\\Users\\indira.lavocat\\MOVIDOC\\PATIENT FILES\\EEG PATIENT FILES\\MOVIDOCTicTrack_BB28-bis.vhdr",
        "excel": "C:\\Users\\indira.lavocat\\MOVIDOC\\PATIENT FILES\\EXCEL PATIENT FILES\\BB28_annotations_binary-table_cutted.xlsx",
        "phase_start": 302600 / 1000.0,
        "phase_end": 929960 / 1000.0,
        "fps": 25,
        "min_absence_frames": 25
    },
    {
        "vhdr": "C:\\Users\\indira.lavocat\\MOVIDOC\\PATIENT FILES\\EEG PATIENT FILES\\MOVIDOCTicTrack000013.vhdr",
        "excel": "C:\\Users\\indira.lavocat\\MOVIDOC\\PATIENT FILES\\EXCEL PATIENT FILES\\BC29_annotations_binary-table_cutted_xlsx_Lizbeth.xlsx",
        "phase_start": 358.215,
        "phase_end": 1088.67,
        "fps": 30,
        "min_absence_frames": 30
    },
    {
        "vhdr": "C:\\Users\\indira.lavocat\\MOVIDOC\\PATIENT FILES\\EEG PATIENT FILES\\MOVIDOCTicTrack000031.vhdr",
        "excel": "C:\\Users\\indira.lavocat\\MOVIDOC\\PATIENT FILES\\EXCEL PATIENT FILES\\SC31_annotations_binary-table_cutted.xlsx",
        "phase_start": 345.972,
        "phase_end": 970.167,
        "fps": 30,
        "min_absence_frames": 30
    }
]

for p in patients:
    results[p["vhdr"]] = run_full_pipeline_for_patient(
        vhdr_path=p["vhdr"],
        excel_path=p["excel"],
        phase_start=p["phase_start"],
        phase_end=p["phase_end"],
        fps=p["fps"],
        min_absence_frames=p["min_absence_frames"]
    )

print("Pipeline finished for all the patients.")

#########################################################################