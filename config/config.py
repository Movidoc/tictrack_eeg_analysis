"""
================================================================================
PROJECT: TicTrack EEG - Premonitory Urge & Tic Suppression
MODULE:  config.py
PHASE:   1 - Infrastructure & Experimental Mapping
--------------------------------------------------------------------------------
DESCRIPTION:
    Centralizes all experimental constants. Maps the FULL TRIG_MAP from 
    task.py to BrainVision decimal markers (Stimulus/S <num>).

NOMENCLATURE / DICTIONARY FOR THE WHOLE PROJECT

Phase Label	Description
PHASE_EC	Eyes Closed
PHASE_EO	Eyes Open
PHASE_KP	Motor Sham (Keypress control)
PHASE_FREE	Free Tic observation
PHASE_SUP	Suppression Block
PHASE_MIM	Mimicry Block

    - PHASE_: Block boundaries. Used to slice the continuous data.
    - EVT_:   Motor/Behavioral events. These are the "onsets" for epoching.
    - FB_:    Feedback triggers. Used to identify Visual Evoked Potentials (VEPs) 
              or Auditory responses that could contaminate the signal.
    - SYS_:   Technical markers for synchronization and data integrity.

IMPORTANCE OF ALL TTLs:
    - S 31-34 (Visual Feedback): Crucial for "Cleaning". We must ensure 
      pre-tic activity isn't confounded by the screen change.
    - S 41-42 (Audio): Helps identify sensory processing blocks.
    - S 100 (Sync): The anchor for aligning external video timestamps.

INPUTS:
    - None (Static configuration)

OUTPUTS:
    - TTL_MAP: Mapping for annotation renaming.
    - PIPE_PARAMS: Global constants for filters and epoch windows.
================================================================================
"""

from pathlib import Path
import os

# --- PATH MANAGEMENT ---
BASE_DIR = Path(os.getcwd())
RAW_DATA_DIR = Path("/mnt/c/Users/lizbe/Documents/EEG_Data")
DERIVATIVES_DIR = BASE_DIR / "derivatives"
REPORTS_DIR = DERIVATIVES_DIR / "reports"

# Create folders if they don't exist
for folder in [DERIVATIVES_DIR, REPORTS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# --- EXHAUSTIVE TRIGGER MAPPING (1:1 with task.py) ---
TTL_MAP = {
    # Phase Transitions (Structural context)
    # PHASE_EC	Eyes Closed
    'Stimulus/S  1': 'SYS_START_EXP',
    'Stimulus/S  2': 'PHASE_0_MOTOR_BASE',
    'Stimulus/S  3': 'PHASE_0A_INSTR',
    'Stimulus/S  4': 'PHASE_1A_EC',        # Eyes Closed (Resting)
    'Stimulus/S  5': 'PHASE_1B_EO',        # Eyes Open (Resting)
    'Stimulus/S  6': 'PHASE_1C_FREE',      # Spontaneous Tic observation 1
    'Stimulus/S  7': 'PHASE_1D_FREE',      # Spontaneous Tic observation 2
    'Stimulus/S  8': 'PHASE_2A_FREE',      # Spontaneous Tic observation 3
    'Stimulus/S  9': 'PHASE_2B_FREE',      # Spontaneous Tic observation 4
    'Stimulus/S 10': 'PHASE_3A_MIMIC',     # Voluntary Mimicry
    'Stimulus/S 11': 'PHASE_3B_MIMIC',
    'Stimulus/S 12': 'PHASE_4A_SUPPRESS',  # Intentional Suppression
    'Stimulus/S 13': 'PHASE_4B_SUPPRESS',
    'Stimulus/S 14': 'SYS_END_EXP',

    # Behavioral Events (Analysis targets)
    'Stimulus/S 21': 'EVT_KEY_D_SUP_ON',   # Suppression attempt started
    'Stimulus/S 22': 'EVT_KEY_F_SUP_OFF',  # Suppression attempt ended
    'Stimulus/S 23': 'EVT_KEY_S_URGE',     # Moment of Urge awareness
    'Stimulus/S 24': 'EVT_KEY_T_SPONT',    # Subject-perceived spontaneous tic
    'Stimulus/S 25': 'EVT_KEY_RIGHT',
    'Stimulus/S 26': 'EVT_KEY_ESC',

    # Visual Feedback (Artifact identification)
    'Stimulus/S 31': 'FB_VIS_D_TEXT',      # Screen displays "Fin de tic supprimé"
    'Stimulus/S 32': 'FB_VIS_F_TEXT',
    'Stimulus/S 33': 'FB_VIS_S_TEXT',      # Screen displays "Urge Awareness"
    'Stimulus/S 34': 'FB_VIS_T_TEXT',

    # Audio/System
    'Stimulus/S 41': 'FB_TONE_START',
    'Stimulus/S 42': 'FB_TONE_END',
    'Stimulus/S 51': 'SYS_WIN_CLOSED',
    'Stimulus/S 100': 'SYS_T0_SYNC'        # Critical sync pulse
}

# --- PIPELINE PARAMETERS ---
PIPE_PARAMS = {
    "sampling_rate": 1000,      # Hz
    "l_freq": 1.0,              # High-pass filter
    "h_freq": 100.0,            # Low-pass filter
    "notch_freq": 50.0,         # Line noise
    
    # Epoching windows (in seconds)
    "t_tic": [-2.0, 1.0],       # 2s before tic, 1s after
    "t_urge": [-3.0, 0.0],      # 3s before urge tag
    "t_mimic": [-2.0, 1.0],     
    "t_kp": [-1.0, 1.0],        # Keypress control
    
    "baseline": (-1.5, -1.0),   # Baseline period for TFR
}


print(f"[INFO] Config validated with {len(TTL_MAP)} TTL markers.")