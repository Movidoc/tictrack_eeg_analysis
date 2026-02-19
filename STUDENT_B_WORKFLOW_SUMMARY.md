# Student B's Code Analysis - Data Import & Processing Workflow

## Overview
Student B's work in `Exploratory_Analysis_MS/` takes Student A's preprocessing functions and combines them to analyze tics and their associated neural signatures across experimental phases.

---

## 📊 Data Sources & Import

### 1. **EEG Data (.vhdr files)**
   - **Location**: `PATIENT FILES/EEG PATIENT FILES/`
   - **Format**: BrainVision format (`.vhdr`, `.eeg`, `.vmrk`)
   - **Patients**: 5 subjects (DS26, BB28, BC29, MM30, SC31)
   - **Sampling Rates**: 250 Hz or higher
   - **Import Function**: `load_data()` → uses MNE's `read_raw_brainvision()`

### 2. **Behavioral Annotations (Excel)**
   - **Location**: `PATIENT FILES/EXCEL PATIENT FILES/`
   - **Content**: Manual tic onset/offset times marked by experimenters
   - **Format**: Excel sheets with frame-based timing
   - **Import Function**: `extract_tics_from_excel()` → uses pandas to read Excel
   - **Key Parameter**: `fps` (frames per second) - varies per subject (25-30 fps)

### 3. **Experimental Protocol Configuration** (your new config.py)
   - **Trigger Map**: Maps BrainVision decimal markers to experimental events
   - **Example TTLs**:
     - `S 9` / `S 10`: Start/End of Spontaneous Tic phase
     - `S 11` / `S 12`: Start/End of Imitated Tic phase
     - `S 13` / `S 14`: Start/End of Retention (Suppression) phase
     - `S 21` (D key): Tic onset mark by subject
     - `S 22` (F key): Tic offset mark by subject
     - `S 24` (T key): Spontaneous tic during instructed phase

---

## 🔄 Data Processing Pipeline

### **Stage 1: EEG Preprocessing** (`_02_Movidoc_tictrack_prepro_TTL_extraction_patients.py`)

```
Raw EEG (.vhdr)
    ↓
load_data()
    ├─ Read BrainVision file
    └─ Extract subject ID from filename
    ↓
extract_stimuli()
    ├─ Extract TTL events from annotations
    └─ Get event codes (S 9, S 10, S 21, etc.)
    ↓
preprocess_data()
    ├─ High-pass filter (1 Hz)
    ├─ Low-pass filter (100 Hz) 
    ├─ Montage application (standard_1020 or standard_1005)
    ├─ Automatic bad channel detection (Ransac)
    └─ Output: Filtered, montaged raw data
    ↓
apply_rest_reference()
    └─ Apply REST (Reference Electrode Standardization)
    ↓
recalibrate_from_first_event()
    ├─ Find first behavioral event (S 2)
    └─ Crop & realign EEG timeline
    ↓
collect_ttl_with_phases()
    ├─ Extract all TTL timestamps & assign phases
    ├─ Map each TTL to experimental phase (spontaneous, imitated, etc.)
    └─ Output: List of {ttl_name, time, phase}
```

**Key Functions Called in Notebook:**
```python
results = run_full_pipeline_for_patient(
    vhdr_path=patient["vhdr"],
    excel_path=patient["excel"],
    fps=patient["fps"],
    min_absence_frames=patient["min_absence_frames"],
    montage_name=patient["montage"],
    excel_phase_times=patient["excel_phase_times"]
)
```

---

### **Stage 2: Tic Extraction & Alignment** (`_03_` + `_04_Movidoc_full_pipeline_analysis_patients.py`)

```
Excel Tic Times (frame-based)
    ↓
extract_tics_from_excel()
    ├─ Read Excel annotations
    ├─ Convert frames → seconds using fps
    ├─ Filter tics (min 30 frames absence before/after)
    └─ Output: List of (start_sec, end_sec) tuples
    ↓
assign_phase_to_tics()
    ├─ Map each tic to experimental phase
    └─ Output: List of {start, end, phase}
    ↓
realign_excel_to_eeg()
    ├─ Calculate linear drift between Excel & EEG timelines
    │   (using 6 reference point pairs)
    ├─ Fit linear model: EEG_time = slope * Excel_time + intercept
    └─ Apply correction to all tic times
```

---

### **Stage 3: Timeline Integration** (`build_merged_ttl_tics()`)

Creates a **unified, time-sorted timeline** containing:

```python
merged_ttl_tics = [
    {'start_1': 12.345},        # Tic 1 start time
    {'D': 12.350},              # Subject pressed D (tic onset)
    {'end_1': 18.720},          # Tic 1 end time
    {'F': 18.728},              # Subject pressed F (tic offset)
    {'start_2': 25.100},        # Tic 2 start time
    ...
]
```

**Three Information Types:**
- **Tics from Excel**: `start_1`, `start_2`, `end_1`, `end_2`, ...
- **Subject Key Presses**: `D` (onset), `F` (offset), `T` (spontaneous), `S` (suppressed)
- **Phase Boundaries**: `start_spont`, `end_spont`, `start_imit`, `end_imit`, `start_ret`, `end_ret`

---

### **Stage 4: Analysis & Epoch Extraction**

```
merged_ttl_tics
    ↓
analyse_merged_ttl_tics_imitated()
    ├─ Categorize tics as "real" or "imitated"
    ├─ Check alignment with subject key presses
    ├─ Output: Analysis list of validated tics
    └─ File: function_per_phase_analyse_merged_ttl.py
    ↓
extract_pre_tic_epochs()
    ├─ Extract EEG windows before tic onset (e.g., -2 to 0 seconds)
    ├─ Create MNE Epochs object
    └─ Output: Ready for TFR (time-frequency) analysis
    ↓
tfr_array_morlet()
    ├─ Compute wavelet transforms (Morlet)
    └─ Generate spectrograms for each ROI
```

---

## 📁 Key Data Structures

### `results` dict from `run_full_pipeline_for_patient()`
```python
{
    "subject": "DS26",
    "montage": "standard_1020",
    "raw_cropped": MNE.Raw,                    # Preprocessed EEG data
    "ttl": [                                    # Extracted TTL events
        {"ttl_name": "Stimulus/S  9", 
         "time": 10.234, 
         "phase": "spontaneous_tics"}
    ],
    "tics_original": [                         # From Excel
        {"start": 12.1, "end": 18.5, "phase": "spontaneous_tics"}
    ],
    "tics_corrected": [                        # Aligned to EEG
        {"start": 12.123, "end": 18.507, "phase": "spontaneous_tics"}
    ],
    "phases_dict": {
        "spontaneous_tics": (10.2, 450.1)
    },
    "linear_drift_params": {
        "slope": 1.00234,
        "intercept": -0.056
    },
    "stim2_time_original": 0.234,
    "bad_channels": ["Fp1", "Oz"]
}
```

### `merged_ttl_tics` - Unified Timeline
```python
[
    {"start_spont": 10.234},
    {"D": 12.350},
    {"start_1": 12.345},
    {"F": 18.728},
    {"end_1": 18.720},
    ...
]
```

---

## 🎯 Current Workflow in Student B's Notebook

**Notebook: `Initial_Analysis_MS_1.ipynb`**

```python
# 1. Load patients config
patients = [
    {
        "montage": "standard_1020",
        "vhdr": ".../MOVIDOCTicTrack000010.vhdr",
        "excel": ".../DS26_annotations_binary-table_cutted.xlsx",
        "fps": 30,
        "excel_phase_times": [13.728, 50.028, 190.080, 349.272, ...]
    },
    # ... 5 patients total
]

# 2. Run pipeline for each patient
for patient in patients:
    results = run_full_pipeline_for_patient(
        vhdr_path=patient["vhdr"],
        excel_path=patient["excel"],
        # ... other params
    )
    
    # 3. Analyze tics per phase
    analysis_list = analyse_merged_ttl_tics(
        merged_ttl_tics=results["merged_ttl_tics"],
        phase_start_key="start_spont",
        phase_end_key="end_spont"
    )
    
    # 4. Extract pre-tic epochs
    pre_tic_epochs = extract_pre_tic_epochs(
        raw_cropped=results["raw_cropped"],
        urge_times=analysis_list  # Tic onset times
    )
    
    # 5. Time-Frequency Analysis (TFR)
    tfr_array = tfr_array_morlet(
        pre_tic_epochs,
        freqs=np.logspace(0, 2, 50),  # 1-100 Hz
        n_cycles=7
    )
    
    # 6. Visualization & Analysis
    plot_spectrograms_by_roi(tfr_array, subject=patient)
```

---

## 🔗 Import Chain

```
Initial_Analysis_MS_1.ipynb
├── imports from _02_Movidoc_tictrack_prepro_TTL_extraction_patients.py
│   ├── load_data()
│   ├── extract_stimuli()
│   ├── preprocess_data()
│   ├── apply_rest_reference()
│   ├── recalibrate_from_first_event()
│   └── collect_ttl_with_phases()
├
├── imports from _03_Movidoc_tictrack_tic_extraction_patients.py
│   └── extract_tics_from_excel()
│
└── imports from _04_Movidoc_full_pipeline_analysis_patients.py
    ├── run_full_pipeline_for_patient()    ← Main integration function
    ├── assign_phase_to_tics()
    ├── extract_eeg_phase_times_from_ttl()
    ├── realign_excel_to_eeg()
    ├── build_merged_ttl_tics()
    ├── analyse_merged_ttl_tics()
    └── extract_pre_tic_epochs()
```

---

## ⚠️ Issues to Address (for BIDS Migration)

1. **Hard-coded paths**: Fixed to `/Users/Tysia/Desktop/...` instead of config-based
2. **No BIDS structure**: Files should map to `sub-XX/ses-01/eeg/` format
3. **No metadata**: Missing BIDS `.json` sidecar files
4. **Linear drift assumption**: Assumes single linear relationship across entire session
5. **Montage hardcoded**: Should come from BIDS dataset_description
6. **Phase timing manual**: The `excel_phase_times` list is hardcoded per subject

---

## Next Steps for Your Refactoring

1. Move paths to your `config.py`
2. Create BIDS-compliant loaders
3. Validate drift correction across subjects
4. Create metadata tracking (participants.tsv, etc.)
5. Refactor to remove code duplication
6. Add unit tests for alignment functions

