# Student B's Code - Function Reference Guide

## 🎯 Map of Key Functions

All these functions are **imported from Student A's files** and **used/extended by Student B**.

---

## 📂 File: `_02_Movidoc_tictrack_prepro_TTL_extraction_patients.py`
**Purpose**: EEG Data Loading & Preprocessing

| Function | Input | Output | Does What |
|----------|-------|--------|-----------|
| `load_data(FilePath)` | `.vhdr` file path | `(raw, subject_name)` | Loads BrainVision file using MNE, extracts subject ID from filename |
| `extract_stimuli(raw)` | MNE Raw object | `(events, event_id)` | Extracts TTL markers from raw signal annotations |
| `preprocess_data(raw, subject_name, montage_name)` | Raw signal, montage name | `(raw_filtered, bad_channels)` | Filters (1-100 Hz), applies montage, detects bad channels via RANSAC |
| `apply_rest_reference(raw, subject_name)` | Filtered raw | Re-referenced raw | Applies REST (Reference Electrode Standardization) reference |
| `recalibrate_from_first_event(raw, target_stim)` | Raw, target TTL name | Cropped & reindexed raw | Realigns signal timeline to start at a specific TTL (usually "Stimulus/S  2") |
| `collect_ttl_with_phases(raw_cropped, subject_name)` | Preprocessed raw | `List[Dict]` | Extracts all TTL timestamps and assigns them to experimental phases |

---

## 📂 File: `_03_Movidoc_tictrack_tic_extraction_patients.py`
**Purpose**: Extract Tic Times from Excel

| Function | Input | Output | Does What |
|----------|-------|--------|-----------|
| `extract_tics_from_excel(excel_file, fps, min_absence_frames)` | Excel path, FPS, filter param | `List[Dict]` | Reads Excel annotations, converts frames→seconds, filters tics by minimum gap |

---

## 📂 File: `_04_Movidoc_full_pipeline_analysis_patients.py`
**Purpose**: Integration & Analysis Core

| Function | Input | Output | Does What |
|----------|-------|--------|-----------|
| `run_full_pipeline_for_patient(...)` | Paths, parameters | `Dict` | **THE MAIN FUNCTION** - runs all preprocessing + tic extraction + returns complete results dict |
| `assign_phase_to_tics(tics, phases_dict)` | Tics list, phase boundaries | Tics with phases assigned | Maps each tic time to its experimental phase |
| `extract_eeg_phase_times_from_ttl(ttl_info)` | TTL list | `Dict[phase_name, (start, end)]` | Extracts phase boundary times from TTL markers |
| `realign_excel_to_eeg(excel_times, eeg_times)` | Two time lists | `(corrected_times, slope, intercept)` | Calculates linear drift correction: `EEG = slope×Excel + intercept` |
| `build_merged_ttl_tics(patient, phases_to_keep)` | Patient dict, phase names | `List[Dict]` | **Creates the unified timeline** mixing tics + TTLs, sorted by time |
| `analyse_merged_ttl_tics(merged_ttl_tics, phase_start_key, phase_end_key)` | Merged timeline, phase markers | `List[Dict]` | Analyzes tic-TTL relationships within a phase, validates tic detection |
| `extract_pre_tic_epochs(raw_cropped, urge_times, tmin, tmax)` | EEG + tic times | MNE Epochs object | Slices EEG into windows around tic onsets (e.g., -2 to +0.5s) |
| `shift_urges_times(urges_list, stim2_time)` | Urge times, reference time | Time-shifted urges | Converts Excel reference times to EEG reference times |

---

## 📂 File: `Exploratory_Analysis_MS/function_per_phase_analyse_merged_ttl.py`
**Purpose**: Student B's Custom Analysis Functions

| Function | Input | Output | Does What |
|----------|-------|--------|-----------|
| `analyse_merged_ttl_tics_imitated(merged_ttl_tics, phase_start_key, phase_end_key, ...)` | Unified timeline | `List[Dict]` | **Student B's function** - categorizes tics as "real" or "imitated" based on D/F key alignment |

---

## 🔗 Function Call Chain (Data Flow)

```
run_full_pipeline_for_patient()
│
├─ EEG BRANCH:
│  ├─ load_data()
│  ├─ extract_stimuli()
│  ├─ preprocess_data()
│  ├─ apply_rest_reference()
│  ├─ recalibrate_from_first_event()
│  └─ collect_ttl_with_phases()
│
├─ EXCEL BRANCH:
│  ├─ extract_tics_from_excel()
│  └─ assign_phase_to_tics()
│
├─ ALIGNMENT BRANCH:
│  ├─ extract_eeg_phase_times_from_ttl()
│  └─ realign_excel_to_eeg()  ← Calculates drift!
│
└─ INTEGRATION:
   └─ build_merged_ttl_tics()  ← CRITICAL: Merges everything
```

**Student B's notebook then calls:**
```
run_full_pipeline_for_patient() [returns results dict]
│
├─ build_merged_ttl_tics() [creates unified timeline]
│
└─ analyse_merged_ttl_tics_imitated() [Student B's custom analysis]
   │
   ├─ shift_urges_times()
   │
   └─ extract_pre_tic_epochs()
      │
      └─ tfr_array_morlet() [Time-frequency analysis]
```

---

## 🔴 Critical Functions to Understand First

### 1. `run_full_pipeline_for_patient()` 
**Why it matters**: This is the MAIN INTEGRATION POINT
- Takes raw .vhdr + Excel files
- Returns fully processed data ready for analysis
- **Handles the critical drift correction**

**You MUST understand**:
- Linear drift calculation (slope & intercept)
- How `stim2_time_original` sets the timeline reference
- What goes into the returned `results` dict

**Code excerpt**:
```python
def run_full_pipeline_for_patient(vhdr_path, excel_path, fps, min_absence_frames, montage_name, excel_phase_times):
    # 1. Load & preprocess EEG
    raw, subject_name = load_data(vhdr_path)
    raw_cropped = recalibrate_from_first_event(...)
    ttl_info = collect_ttl_with_phases(raw_cropped, subject_name)
    
    # 2. Extract tics from Excel
    tics_original = extract_tics_from_excel(excel_path, fps, min_absence_frames)
    
    # 3. CRITICAL: Calculate drift correction
    eeg_phase_times = extract_eeg_phase_times_from_ttl(ttl_info)
    tics_corrected, slope, intercept = realign_excel_to_eeg(
        excel_times=excel_phase_times,
        eeg_times=eeg_phase_times
    )
    
    # 4. Return everything
    return {
        "subject": subject_name,
        "raw_cropped": raw_cropped,
        "ttl": ttl_info,
        "tics_corrected": tics_corrected,
        "linear_drift_params": {"slope": slope, "intercept": intercept},
        ...
    }
```

---

### 2. `build_merged_ttl_tics()`
**Why it matters**: This creates the UNIFIED TIMELINE
- Merges tic onsets/offsets with behavioral key presses
- Creates a single sorted list mixing all event types
- This is the **DATA STRUCTURE that Student B analyzes**

**You MUST understand**:
- Input is `patient` dict (output from `run_full_pipeline_for_patient()`)
- Output is a list of single-key dicts (one event per dict)
- Each dict has format: `{event_name: timestamp}`
- All timestamps are in EEG time domain
- Sorted chronologically

**Example output**:
```python
[
    {"start_spont": 10.234},
    {"D": 12.350},              # Subject key press
    {"start_1": 12.345},        # Tic from Excel
    {"end_1": 18.720},
    {"F": 18.728},              # Subject key press
    {"start_2": 25.100},
    ...
]
```

This is **NOT a dataframe**. This is a **list of dicts** where each dict has exactly one key-value pair.

---

### 3. `analyse_merged_ttl_tics_imitated()` 
**Why it matters**: This is Student B's CUSTOM ANALYSIS
- Takes the unified timeline
- Analyzes relationships between tics & key presses
- Categorizes tics as "real" (matched with T key) or "imitated" (D→F pairs)
- Returns validated tic list

**Location**: `Exploratory_Analysis_MS/function_per_phase_analyse_merged_ttl.py`

**Logic**:
- Real tic: Excel tic + T marker (spontaneous tic occurred)
- Imitated tic: D key (start) → F key (end) with tic inside
- Invalid tic: No behavioral markers around it

**You MUST understand**:
- This function validates which tics are behaviorally confirmed
- It's phase-specific (analyzes spontaneous, imitated, suppressed separately)
- Output is a list of validated tics with metadata

---

### 4. `extract_pre_tic_epochs()`
**Why it matters**: Prepares data for spectral analysis
- Takes raw EEG + tic times
- Extracts fixed-length windows (e.g., -2 to +0.5 seconds)
- Returns MNE Epochs object (ready for wavelet analysis)

**You MUST understand**:
- Input: `raw_cropped` (the preprocessed EEG) + list of tic times
- Output: MNE Epochs (n_epochs × n_channels × n_samples)
- Can then pipe to `tfr_array_morlet()` for spectrograms

---

## ⚠️ Common Pitfalls

### ❌ "My tics don't align!"
**Cause**: Linear drift not properly calculated
- Check if `excel_phase_times` are correct (6 reference points)
- Verify those correspond to actual TTL markers in EEG
- Look at slope value: if >1.05 or <0.95, something's wrong

### ❌ "No TTLs extracted!"
**Cause**: Wrong TTL name format
- BrainVision format uses "Stimulus/S XXX" (with spaces!)
- Not "S XXX" or "Stim XXX"
- Check file: `ttl_name in ttl_info[0]` for exact format

### ❌ "Bad channels not detected"
**Cause**: Montage might not cover all electrodes
- If electrode is in .vhdr but not in montage, it won't be checked
- Solution: Include all electrodes in montage selection

### ❌ "Phase assignment wrong"
**Cause**: Phase boundaries shifted
- Verify `phases_dict` contains correct time ranges
- Usually extracted from TTL markers S 9-14
- If no S 14 (end marker), phase boundary will be None

---

## 🔧 How to Debug

### Print the key data structures:

```python
# After preprocessing
print("TTL info sample:")
print(ttl_info[:3])

# After tic extraction
print("Tics (original Excel times):")
print(tics_original[:3])

# After alignment
print("Tics (aligned to EEG):")
print(tics_corrected[:3])

# After merging
print("Merged timeline sample:")
print(merged_ttl_tics[:10])

# Results from analysis
print("Validated tics:")
for urge in analysis_list:
    print(f"  Type: {urge['type']}, Start: {urge.get('start_time', 'N/A')}")
```

### Check linear drift:
```python
slope = results["linear_drift_params"]["slope"]
intercept = results["linear_drift_params"]["intercept"]

print(f"Slope: {slope:.6f}")          # Should be close to 1.0
print(f"Intercept: {intercept:.6f}")  # Should be close to 0

# Calculate drift magnitude
drift_ms = abs((slope - 1.0) * 1000)  # Convert to milliseconds
print(f"Drift: ±{drift_ms:.2f} ms")  # Usually <20 ms for 30 minute sessions
```

---

## 📍 Where is Each Function Used?

| Function | Used In | Purpose |
|----------|---------|---------|
| `load_data()` | `run_full_pipeline_for_patient()` | Load EEG |
| `extract_stimuli()` | `run_full_pipeline_for_patient()` | Get TTL events |
| `preprocess_data()` | `run_full_pipeline_for_patient()` | Filter & clean |
| `apply_rest_reference()` | `run_full_pipeline_for_patient()` | Re-reference |
| `recalibrate_from_first_event()` | `run_full_pipeline_for_patient()` | Realign timeline |
| `collect_ttl_with_phases()` | `run_full_pipeline_for_patient()` | Map TTLs to phases |
| `extract_tics_from_excel()` | `run_full_pipeline_for_patient()` | Read tic times |
| `assign_phase_to_tics()` | `run_full_pipeline_for_patient()` | Categorize tics |
| `extract_eeg_phase_times_from_ttl()` | `run_full_pipeline_for_patient()` | Get EEG phase bounds |
| `realign_excel_to_eeg()` | `run_full_pipeline_for_patient()` | Calculate drift |
| `build_merged_ttl_tics()` | Student B's notebook | Create unified timeline |
| `analyse_merged_ttl_tics_imitated()` | Student B's notebook | Validate tics |
| `extract_pre_tic_epochs()` | Student B's notebook | Prepare for TFR |

---

## 🎓 Learning Path

1. **Start here**: Read `run_full_pipeline_for_patient()` 
   - Understand the overall flow
   
2. **Then**: Zoom into `build_merged_ttl_tics()`
   - Understand the unified timeline structure
   
3. **Then**: Read `analyse_merged_ttl_tics_imitated()` 
   - See how Student B analyzes the timeline
   
4. **Finally**: Look at `extract_pre_tic_epochs()`
   - Understand the spectral analysis preparation

This progression matches the actual data flow!

