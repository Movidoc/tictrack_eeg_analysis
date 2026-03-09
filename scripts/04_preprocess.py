
#  ---------------------------------------------- 
# Project : 02_preprocess.py
# Author  : Martyna
# Goal    : Run full preprocessing pipeline for all patients
#  ---------------------------------------------- 

from __future__ import annotations

import sys
import os
import argparse
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mne
from config.config import PATIENTS, PIPE_PARAMS, PREPROC_DIR, DATASET_DIR, ICA_EXCLUSIONS
from src.preproc import (
    preprocess_raw,
    Ransac_bad_channel_detection,
    rejection_threshold_std,
    apply_ICA,
    apply_rest_reference
)
# Dataset constants (1 session, 1 run)
TASK = "tictrack"
SES = "01"
RUN = "01"


def bids_eeg_dir(sub: str) -> Path:
    return DATASET_DIR / sub / f"ses-{SES}" / "excel"

def bids_vhdr_path(sub: str) -> Path:
    return DATASET_DIR / sub / f"ses-{SES}" / "eeg"/ f"{sub}_task-{TASK}.vhdr"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run preprocessing pipeline for TicTrack EEG patients"
    )
    parser.add_argument(
        "--sub",
        nargs="*",
        default=None,
        help="Subjects to process, e.g. --sub sub-BB28 sub-BC29. Default: all patients",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # --- Select patients ---
    if args.sub:
        subjects = {k: v for k, v in PATIENTS.items() if k in args.sub}
        missing = [s for s in args.sub if s not in PATIENTS]
        if missing:
            raise RuntimeError(f"Subjects not found in config: {missing}")
    else:
        subjects = PATIENTS


    for sub_id, cfg in subjects.items():
        print(f"\n{'='*60}")
        print(f"[START] Processing {sub_id}")
        print(f"{'='*60}")

        # --- Output folders ---
        out_dir = PREPROC_DIR / sub_id
        out_dir.mkdir(parents=True, exist_ok=True)

        plots_dir = out_dir / "preprocessing"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # --- 1. Load recalibrated raw EEG ---
        fif_path = PREPROC_DIR / sub_id / "realign" / f"{sub_id}_ses-01_task-tictrack_aligned_raw.fif"
        print(f"\n{'='*60}")
        print(f"[1/6] Loading EEG: {fif_path}")
        print(f"\n{'='*60}")
        raw = mne.io.read_raw_fif(fif_path, preload=True, verbose="ERROR")
    

        # --- 2. Preprocess (filter + montage) ---
        print(f"\n{'='*60}")
        print(f"[2/6] Filtering and applying montage...")
        print(f"\n{'='*60}")
        raw = preprocess_raw(raw, sub_id, cfg.montage, plots_dir)

        # --- 3. RANSAC bad channel detection ---
        print(f"\n{'='*60}")
        print(f"[3/6] Detecting bad channels with RANSAC...")
        print(f"\n{'='*60}")
        raw, bad_channels =  Ransac_bad_channel_detection(raw, sub_id, plots_dir)

        # --- 4. Epoch rejection (FASTER) ---
        print(f"\n{'='*60}")
        print(f"[4/6] Rejecting bad epochs (FASTER)...")
        print(f"\n{'='*60}")
        epochs_temp = rejection_threshold_std(raw, sub_id, plots_dir)

        # --- 5. ICA ---
        print(f"\n{'='*60}")
        print(f"[5/6] Applying ICA...")
        print(f"\n{'='*60}")
        raw = apply_ICA(epochs_temp, raw, sub_id, ICA_EXCLUSIONS, plots_dir)

        # --- 6. Re-reference to REST ---
        print(f"\n{'='*60}")
        print(f"[6/6] Re-referencing to REST...")
        print(f"\n{'='*60}")
        raw = apply_rest_reference(raw, sub_id, plots_dir)

        # --- Save preprocessed raw ---
        out_path = out_dir / f"{sub_id}_ses-01_task-tictrack_preprocessed-raw.fif"
        raw.save(out_path, overwrite=True)
        print(f"[OK] Saved preprocessed data: {out_path}")
        print(f"[OK] Saved plots to: {plots_dir}")

    print("\n[DONE] Preprocessing complete.")


if __name__ == "__main__":
    main()

