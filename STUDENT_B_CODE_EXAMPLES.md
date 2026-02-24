# Student B's Code - Practical Usage Examples

## Quick Reference: How Data Flows Through the Code

---

## 1️⃣ STAGE 1: Load & Preprocess EEG

### Direct File Loading
```python
from _02_Movidoc_tictrack_prepro_TTL_extraction_patients import (
    load_data,
    extract_stimuli,
    preprocess_data, 
    apply_rest_reference,
    recalibrate_from_first_event,
    collect_ttl_with_phases
)

# Load raw EEG
vhdr_path = "PATIENT FILES/EEG PATIENT FILES/MOVIDOCTicTrack000010.vhdr"
raw, subject_name = load_data(vhdr_path)

# Output:
# raw = MNE RawArray object with annotations (TTL events)
# subject_name = "000010"
```

### Preprocessing Chain
```python
# Step 1: Extract raw TTL events
events_raw, event_ids = extract_stimuli(raw)
# Output: array of shape (n_events, 3) with sample indices & event codes

# Step 2: Filter & montage the signal
raw_filtered, bad_channels = preprocess_data(raw, subject_name="000010", montage_name="standard_1020")
# - Fixes channels
# - Applies high-pass (1 Hz) & low-pass (100 Hz) filters  
# - Applies electrode montage
# Output: Filtered MNE Raw object + list of bad channels

# Step 3: Apply REST reference
raw_rest = apply_rest_reference(raw_filtered, subject_name)
# Output: Re-referenced data

# Step 4: Realign timeline from first meaningful event (S 2)
raw_cropped = recalibrate_from_first_event(raw_rest, target_stim="Stimulus/S  2")
# Output: EEG signal cropped & time-shifted to start at S 2

# Step 5: Extract TTL info with phase assignment
ttl_info = collect_ttl_with_phases(raw_cropped, subject_name)
# Output: List of dicts
# [
#     {"ttl_name": "Stimulus/S  9", "time": 10.234, "phase": "spontaneous_tics"},
#     {"ttl_name": "Stimulus/S 21", "time": 12.350, "phase": "spontaneous_tics"},
#     ...
# ]
```

---

## 2️⃣ STAGE 2: Extract & Align Tics from Excel

### Extract Tics from Excel
```python
from _03_Movidoc_tictrack_tic_extraction_patients import extract_tics_from_excel

excel_path = "PATIENT FILES/EXCEL PATIENT FILES/DS26_annotations_binary-table_cutted.xlsx"
fps = 30  # frames per second for this subject
min_absence_frames = 30

tics_original = extract_tics_from_excel(
    excel_file=excel_path,
    fps=fps,
    min_absence_frames=min_absence_frames
)

# Output: List of dicts (ORIGINAL EXCEL TIMES, NOT YET ALIGNED)
# [
#     {"start": 0.400, "end": 1.200, "phase": None},
#     {"start": 5.150, "end": 6.050, "phase": None},
#     ...
# ]
```

### Align to EEG Timeline (Linear Drift Correction)
```python
from _04_Movidoc_full_pipeline_analysis_patients import (
    assign_phase_to_tics,
    realign_excel_to_eeg,
    extract_eeg_phase_times_from_ttl
)

# Define phase boundaries from EEG TTLs
eeg_phase_times = extract_eeg_phase_times_from_ttl(ttl_info)
# Output: {"spontaneous_tics": (10.234, 450.567), "imitated_tics": ...}

# Get phase boundaries from Excel (hardcoded reference points)
excel_phase_times = [13.728, 50.028, 190.080, 349.272, 980.727, 1277.991]

# Calculate linear drift: EEG_time = slope * Excel_time + intercept
tics_corrected, slope, intercept = realign_excel_to_eeg(
    excel_times=excel_phase_times,
    eeg_times=list(eeg_phase_times.values())
)

# Output:
# tics_corrected = [
#     {"start": 0.402, "end": 1.205, "phase": "spontaneous_tics"},
#     ...
# ]
# slope ≈ 1.00234  (EEG is ~0.234% faster)
# intercept ≈ -0.056  (small time shift)

# Now assign phases to corrected tics
phases_dict = {
    "spontaneous_tics": (10.234, 450.567),
    "imitated_tics": (460.123, 850.456),
    ...
}

tics_with_phases = assign_phase_to_tics(tics_corrected, phases_dict)
```

---

## 3️⃣ STAGE 3: Create Unified Timeline

### Merge TTLs + Tics into Single Timeline
```python
from _04_Movidoc_full_pipeline_analysis_patients import build_merged_ttl_tics

# Prepare patient data dict (output from preprocessing)
patient_data = {
    "tics_corrected": tics_with_phases,
    "ttl": ttl_info,
    "phases_dict": phases_dict,
    ...
}

# Build merged timeline
merged_ttl_tics = build_merged_ttl_tics(
    patient=patient_data,
    phases_to_keep=["spontaneous_tics", "imitated_tics"]
)

# Output: Chronologically sorted list mixing all event types
# [
#     {"start_spont": 10.234},        # Phase boundary
#     {"D": 12.350},                  # Subject pressed D (tic onset)
#     {"start_1": 12.345},            # Tic 1 from Excel
#     {"F": 18.728},                  # Subject pressed F (tic offset)
#     {"end_1": 18.720},              # Tic 1 end
#     {"start_2": 25.100},            # Tic 2 starts
#     {"D": 25.105},                  # Subject also pressed D
#     {"end_2": 31.560},              # Tic 2 ends
#     ...
# ]

# CRITICAL: This merged list is what Student B analyzes!
```

---

## 4️⃣ STAGE 4: Analyze & Extract Epochs

### Analyze Tics (Real vs Imitated)
```python
from _04_Movidoc_full_pipeline_analysis_patients import analyse_merged_ttl_tics
from Exploratory_Analysis_MS.function_per_phase_analyse_merged_ttl import analyse_merged_ttl_tics_imitated

# Analyze within a specific phase
analysis_list = analyse_merged_ttl_tics(
    merged_ttl_tics=merged_ttl_tics,
    phase_start_key="start_spont",    # Phase boundaries
    phase_end_key="end_spont",
)

# Output: List of validated tics with metadata
# [
#     {
#         "type": "start_then_T",  # tic with T marker
#         "start_time": 12.345
#     },
#     {
#         "type": "end_then_T",
#         "start_time": 25.100
#     },
#     ...
# ]
```

### Extract Pre-tic EEG Epochs
```python
from _04_Movidoc_full_pipeline_analysis_patients import extract_pre_tic_epochs, shift_urges_times
from mne import Epochs

# Shift tic times from Excel reference to EEG reference
stim2_time = 0.234  # From preprocessing

urges_times_eeg = shift_urges_times(
    urges_list=analysis_list,
    stim2_time=stim2_time
)

# Extract EEG windows around tics (e.g., -2 to +0.5 seconds)
pre_tic_epochs = extract_pre_tic_epochs(
    raw_cropped=raw_cropped,
    urge_times=urges_times_eeg,
    tmin=-2.0,      # 2 seconds before tic
    tmax=0.5,       # 0.5 seconds after
)

# Output: MNE Epochs object (n_epochs, n_channels, n_samples)
# Can now be used for spectral analysis
```

---

## 🎯 The COMPLETE INTEGRATED PIPELINE

This is what Student B's notebook does:

```python
from _04_Movidoc_full_pipeline_analysis_patients import run_full_pipeline_for_patient

# Define patient
patient = {
    "montage": "standard_1020",
    "vhdr": "PATIENT FILES/EEG PATIENT FILES/MOVIDOCTicTrack000010.vhdr",
    "excel": "PATIENT FILES/EXCEL PATIENT FILES/DS26_annotations_binary-table_cutted.xlsx",
    "fps": 30,
    "min_absence_frames": 30,
    "excel_phase_times": [13.728, 50.028, 190.080, 349.272, 980.727, 1277.991]
}

# RUN EVERYTHING IN ONE COMMAND
results = run_full_pipeline_for_patient(
    vhdr_path=patient["vhdr"],
    excel_path=patient["excel"],
    fps=patient["fps"],
    min_absence_frames=patient["min_absence_frames"],
    montage_name=patient["montage"],
    excel_phase_times=patient["excel_phase_times"]
)

# What you get back:
# {
#     "subject": "000010",
#     "montage": "standard_1020",
#     "raw_cropped": <MNE Raw>,           # Processed EEG
#     "bad_channels": ["Fp1"],            # Channels to ignore
#     "ttl": [...],                       # TTL events with phases
#     "tics_original": [...],             # From Excel (original times)
#     "tics_corrected": [...],            # Aligned to EEG
#     "phases_dict": {...},               # Phase boundaries
#     "excel_corrected_times": [...],     # Realigned Excel reference points
#     "linear_drift_params": {"slope": 1.00234, "intercept": -0.056},
#     "stim2_time_original": 0.234,
#     "merged_ttl_tics": [...]            # The unified timeline
# }

print(results["subject"])  # "000010"
print(results["linear_drift_params"]["slope"])  # 1.00234
print(len(results["tics_corrected"]))  # Number of tics found
```

---

## 📋 Data Format Reference

### TTL Info Format
```python
ttl_info = [
    {
        "ttl_name": "Stimulus/S  9",       # BrainVision trigger name
        "time": 10.234,                     # Time in seconds (after realignment)
        "phase": "spontaneous_tics"         # Experimental phase
    },
    {
        "ttl_name": "Stimulus/S 21",
        "time": 12.350,
        "phase": "spontaneous_tics"
    }
]
```

### Tics Format (before alignment)
```python
tics_original = [
    {
        "start": 0.400,           # Frame-based time (from Excel)
        "end": 1.200,
        "phase": None             # Phase not yet assigned
    }
]
```

### Tics Format (after alignment)
```python
tics_corrected = [
    {
        "start": 0.402,           # Time aligned to EEG
        "end": 1.205,
        "phase": "spontaneous_tics"  # Now it has a phase
    }
]
```

### Merged Timeline Format
```python
merged_ttl_tics = [
    {"start_spont": 10.234},      # Phase boundary
    {"D": 12.350},                # Key press event
    {"start_1": 12.345},          # Tic marker
    {"end_1": 18.720},
    # All sorted by time value
]
```

---

## 🔍 Key Insights for Your BIDS Refactoring

1. **Data Dependencies**: The code assumes Excel times can be aligned to EEG via linear drift
   - If this breaks in BIDS format, check the slope/intercept values

2. **Hardcoded Reference Points**: `excel_phase_times` are manual per-subject
   - Should be extracted from your BIDS task description or dataset

3. **Montage is Critical**: Determines which channels are valid
   - Must match what's in your EEG header

4. **TTL Event Codes are Standard**:
   - S 9-14: Phase markers
   - S 21-24: Behavioral markers (D, F, T, S keys)
   - These map to your experimental protocol

5. **Order Matters**: Preprocessing must happen before alignment
   - Can't extract accurate TTL times from raw signal

---

## 🚀 Common Operations

### Get all tics from a specific phase
```python
spont_tics = [t for t in results["tics_corrected"] if t["phase"] == "spontaneous_tics"]
```

### Get bad channels to exclude from analysis
```python
bad_ch = results["bad_channels"]
good_channels = [ch for ch in results["raw_cropped"].ch_names if ch not in bad_ch]
```

### Check drift magnitude
```python
slope = results["linear_drift_params"]["slope"]
drift_percent = (slope - 1.0) * 100
print(f"EEG is {drift_percent:.2f}% faster than Excel")  # Usually <1%
```

### Extract tics from specific phase AND condition
```python
def get_phase_tics(results, phase_name):
    return [t for t in results["tics_corrected"] 
            if t["phase"] == phase_name]

spont = get_phase_tics(results, "spontaneous_tics")
imitated = get_phase_tics(results, "imitated_tics")
suppressed = get_phase_tics(results, "retention_tics")
```

