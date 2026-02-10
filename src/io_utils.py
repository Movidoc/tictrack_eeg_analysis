"""
PROJECT: TicTrack EEG Analysis
MODULE:  io_utils.py
AUTHOR:  LizbethMG
PURPOSE: Robust loading of BrainVision (.vhdr) data and integration 
         of external Video-TIC annotation files.
"""

import mne
import pandas as pd
from config import RAW_DATA_DIR

def load_raw_eeg(subject_id):
    """
    Loads the BrainVision file for a given subject.
    Expects file naming: sub-01_task-tic_raw.vhdr
    """
    file_path = RAW_DATA_DIR / f"sub-{subject_id}" / f"sub-{subject_id}_raw.vhdr"
    
    if not file_path.exists():
        raise FileNotFoundError(f"Could not find EEG file at: {file_path}")
        
    raw = mne.io.read_raw_brainvision(file_path, preload=True)
    
    # Standardize channel types
    # (Optional: Add logic here to set EOG channels if you have them)
    raw.set_channel_types({'HEOG': 'eog', 'VEOG': 'eog'}) if 'HEOG' in raw.ch_names else None

    return raw

def load_video_annotations(subject_id):
    """
    Loads the .tsv file containing tic onsets from video coding.
    """
    tsv_path = RAW_DATA_DIR / f"sub-{subject_id}" / f"sub-{subject_id}_tics.tsv"
    
    if not tsv_path.exists():
        print(f"⚠️ Warning: No video tic file found for subject {subject_id}")
        return None
        
    df_tics = pd.read_csv(tsv_path, sep='\t')
    return df_tics