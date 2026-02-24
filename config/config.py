"""
================================================================================
PROJECT: TicTrack EEG - Premonitory Urge & Tic Suppression
AUTHOR: Lizbeth Mondragon Gonzalez (@LizbethMG)
MODULE:  config.py
PHASE:   1 - Infrastructure & Experimental Mapping
--------------------------------------------------------------------------------
DESCRIPTION:
    Centralizes all experimental constants. Maps the FULL TRIG_MAP from 
    task.py to BrainVision decimal markers (Stimulus/S <num>).

NOMENCLATURE / DICTIONARY FOR THE WHOLE PROJECT
(BrainVision TTL / Trigger Reference)

====================================================================
GLOBAL
====================================================================
system (can occur anytime)
    100 system:t0                      Software time-zero anchor (first event only)
    51  system:window_closed           Window closed unexpectedly
start_experiment (instruction screen)
    1   phase_start:start_experiment   Start screen displayed
    25  key_press:right                Subject pressed ➡ to continue

====================================================================
PHASE_KP_INS  - Instruction screen (phase0)
====================================================================
    2   phase_start:phase0             Entered motor sham instruction screen
    25  key_press:right                Subject pressed ➡ to continue
====================================================================
*PHASE_KP*  - Motor sham / keypress block (phase0a)
====================================================================
    3   phase_start:phase0a            Motor sham block begins
    21  key_press:d                    Subject pressed D (motor control press)
    31  visual_feedback:d              Visual confirmation displayed (may repeat)
NOTE:
    No behavioral tic markers expected in this phase, but in the video yes. 
====================================================================
PHASE_EC_INS  - Instruction screen (phase1a)
====================================================================
    4   phase_start:phase1a            Entered eyes-closed instruction screen
    25  key_press:right                Subject pressed ➡ to continue
====================================================================
*PHASE_EC*  - Eyes Closed Resting State (phase1b)
====================================================================
    5   phase_start:phase1b            Eyes-closed resting block begins
    41  tone_feedback:tone_start       Start beep played
    42  tone_feedback:tone_end         End beep played
NOTE:
    No behavioral tic markers expected in this phase, but in the video yes. 
====================================================================
PHASE_EO_INS  - Instruction screen (phase1c)
====================================================================
    6   phase_start:phase1c            Entered eyes-open instruction screen
    25  key_press:right                Subject pressed ➡ to continue
====================================================================
PHASE_EO  - Eyes Open + Fixation Cross (phase1d)
====================================================================
    7   phase_start:phase1d            Eyes-open resting block begins
    41  tone_feedback:tone_start       Start beep played
    42  tone_feedback:tone_end         End beep played

NOTE:
    No behavioral tic markers expected in this phase, but in the video yes.
====================================================================
PHASE_FREE_INS  - Instruction screen (phase2a)
====================================================================
    8   phase_start:phase2a            Entered free-tic instruction screen
    25  key_press:right                Subject pressed ➡ to continue
====================================================================
PHASE_FREE  - Free Tic Observation (phase2b)
====================================================================
    9   phase_start:phase2b            Free tic observation block begins
    21  key_press:d                    Subject marks tic/urge beginning
    22  key_press:f                    Subject marks tic end
    31  visual_feedback:d              Visual confirmation (may repeat)
    32  visual_feedback:f              Visual confirmation (may repeat)
====================================================================
PHASE_MIM_INS  - Instruction screen (phase3a)
====================================================================
    10  phase_start:phase3a            Entered mimicry instruction screen
    25  key_press:right                Subject pressed ➡ to continue
====================================================================
PHASE_MIM  - Mimicry Block (phase3b)
====================================================================
    11  phase_start:phase3b            Mimicry block begins
    21  key_press:d                    Start voluntary mimic tic
    22  key_press:f                    End voluntary mimic tic
    24  key_press:t                    Spontaneous tic occurred
    31  visual_feedback:d              Visual confirmation (may repeat)
    32  visual_feedback:f              Visual confirmation (may repeat)
    34  visual_feedback:t              Visual confirmation (may repeat)

INTERPRETATION:
    21 → 22 = one voluntary mimic tic.
    24 = spontaneous tic intrusion during mimicry.
====================================================================
PHASE_SUP_INS  - Instruction screen (phase4a)
====================================================================
    12  phase_start:phase4a            Entered suppression instruction screen
    25  key_press:right                Subject pressed ➡ to continue
====================================================================
PHASE_SUP  - Suppression Block (phase4b)
====================================================================
    13  phase_start:phase4b            Suppression block begins
    23  key_press:s                    Intention to suppress tic
    22  key_press:f                    End suppression attempt
    24  key_press:t                    Tic occurred (failed suppression)
    33  visual_feedback:s              Visual confirmation (may repeat)
    32  visual_feedback:f              Visual confirmation (may repeat)
    34  visual_feedback:t              Visual confirmation (may repeat)
INTERPRETATION:
    23 → 22 defines one suppression attempt.
    24 indicates breakthrough tic.
====================================================================
end_experiment (end screen; ESC exits)
====================================================================
    14  phase_start:end_experiment     End screen displayed
    26  key_press:esc                  Experiment terminated
====================================================================
PRIMARY ANALYSIS TTLs (recommended for EEG processing)
====================================================================

Block segmentation:
    3   PHASE_KP
    5   PHASE_EC
    7   PHASE_EO
    9   PHASE_FREE
    11  PHASE_MIM
    13  PHASE_SUP

================================================================================

INPUTS:
    - None (Static configuration)

OUTPUTS:
    - TTL_MAP: Mapping for annotation renaming.
    - PIPE_PARAMS: Global constants for filters and epoch windows.
================================================================================
"""

from pathlib import Path
import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict

# --- 1. DATA STRUCTURE DEFINITION ---
@dataclass
class SubjectConfig:
    sub_id: str
    vhdr_path: Path
    montage: str = "standard_1020"
    eog_chs: List[str] = field(default_factory=lambda: ["VEOG", "HEOG"])
    ecg_ch: Optional[str] = "ECG"
    rename_chs: Dict[str, str] = field(default_factory=dict)
    bads: List[str] = field(default_factory=list)
    notes: str = ""

# --- 2. PATH MANAGEMENT ---

BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = BASE_DIR / "dataset"
DERIVATIVES_DIR = DATASET_DIR / "derivatives"
PREPROC_DIR = DERIVATIVES_DIR / "preproc"

# --- 3. TRIGGER DEFINITION (Integers) ---

TTL_LABELS = {
    # Phases
    1:  "START_EXPERIMENT",
    2:  "PHASE_KP_INS", 3:  "PHASE_KP",
    4:  "PHASE_EC_INS", 5:  "PHASE_EC",
    6:  "PHASE_EO_INS", 7:  "PHASE_EO",
    8:  "PHASE_FREE_INS", 9:  "PHASE_FREE",
    10: "PHASE_MIM_INS", 11: "PHASE_MIM",
    12: "PHASE_SUP_INS", 13: "PHASE_SUP",
    14: "END_EXPERIMENT",

    # Keys
    21: "KEY_D",
    22: "KEY_F",
    23: "KEY_S",
    24: "KEY_T",
    25: "KEY_RIGHT",
    26: "KEY_ESC",

    # Visual feedback
    31: "FB_D",
    32: "FB_F",
    33: "FB_S",
    34: "FB_T",

    # Audio feedback
    41: "TONE_START",
    42: "TONE_END",

    # System events
    51:  "WINDOW_CLOSED",
    100: "T0_SYNC",
}

# --- 4. BRAINVISION STRING MAPPING (The FIX for MNE) ---
# This converts 1 into "Stimulus/S  1" and 21 into "Stimulus/S 21"

def _format_bv_label(code: int) -> str:
    if code < 10:
        return f"Stimulus/S  {code}"  # Two spaces for single digits
    else:
        return f"Stimulus/S {code}"   # One space for double/triple digits

TTL_MAP = {_format_bv_label(k): v for k, v in TTL_LABELS.items()}

# --- 5. PIPELINE PARAMETERS ---

PIPE_PARAMS = {
    "sampling_rate": 1000,      # Hz
    "l_freq": 1.0,              # High-pass filter
    "h_freq": 100.0,            # Low-pass filter
    "notch_freq": 50.0,         # Line noise
    "epoch_windows": {
        "t_tic": [-2.0, 1.0],
        "t_urge": [-3.0, 0.0],
        "t_mimic": [-2.0, 1.0],
        "t_kp": [-1.0, 1.0]
    },
    "baseline": (-1.5, -1.0),   # Baseline period for TFR
}

# --- 6. PATIENT REGISTRY ---

PATIENTS = {
    "sub-001": SubjectConfig(
        sub_id="sub-001",
        vhdr_path=Path(r"/mnt/c/.../sub-001/sub-001_task-tictrack.vhdr"),
        montage="standard_1020",
        eog_chs=["VEOG", "HEOG"],
        ecg_ch="ECG",
        rename_chs={"FP1": "Fp1"},   # example if needed
        bads=[],
        notes="",
    ),

    "sub-002": SubjectConfig(
        sub_id="sub-002",
        vhdr_path=Path(r"/mnt/c/.../sub-002/sub-002_task-tictrack.vhdr"),
        eog_chs=["VEOG"],
        bads=["Fp1"],                # example manual QC
        notes="No HEOG recorded",
    ),
}

if __name__ == "__main__":
    print(f"[INFO] Configuration file loaded successfully.")
    print(f"[INFO] Dataset dir: {DATASET_DIR}")
    print(f"[INFO] Mapped {len(TTL_LABELS)} distinct triggers.")