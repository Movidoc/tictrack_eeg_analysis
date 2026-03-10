#  ---------------------------------------------- 
# Project : 04_preprocess.py
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
from src.extract_ttl_events import plot_raw, build_phases_dict
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


    for sub, cfg in subjects.items():
        print(f"\n{'='*60}")
        print(f"[START] Processing {sub}")
        print(f"{'='*60}")

        # --- Output folders ---
        out_dir = PREPROC_DIR / sub
        out_dir.mkdir(parents=True, exist_ok=True)

        plots_dir = out_dir / "preprocessing"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # --- 1. Load recalibrated raw EEG ---
        fif_path = PREPROC_DIR / sub / "realign" / f"{sub}_ses-01_task-tictrack_aligned_annotated_raw.fif"
        print(f"\n{'='*60}")
        print(f"[1/6] Loading EEG: {fif_path}")
        print(f"\n{'='*60}")
        raw = mne.io.read_raw_fif(fif_path, preload=True, verbose="ERROR")
    

        # --- 2. Preprocess (filter + montage) ---
        print(f"\n{'='*60}")
        print(f"[2/6] Filtering and applying montage...")
        print(f"\n{'='*60}")
        raw = preprocess_raw(raw, sub, cfg.montage, plots_dir)

        # --- 3. RANSAC bad channel detection ---
        print(f"\n{'='*60}")
        print(f"[3/6] Detecting bad channels with RANSAC...")
        print(f"\n{'='*60}")
        raw, bad_channels =  Ransac_bad_channel_detection(raw, sub, plots_dir)

        # --- 4. Epoch rejection (FASTER) ---
        print(f"\n{'='*60}")
        print(f"[4/6] Rejecting bad epochs (FASTER)...")
        print(f"\n{'='*60}")
        epochs_temp = rejection_threshold_std(raw, sub, plots_dir)

        # --- 5. ICA ---
        print(f"\n{'='*60}")
        print(f"[5/6] Applying ICA...")
        print(f"\n{'='*60}")
        raw = apply_ICA(epochs_temp, raw, sub, ICA_EXCLUSIONS, plots_dir)

        # --- 6. Re-reference to REST ---
        print(f"\n{'='*60}")
        print(f"[6/6] Re-referencing to REST...")
        print(f"\n{'='*60}")
        raw = apply_rest_reference(raw, sub, plots_dir)

        # --- Save preprocessed raw ---
        out_path = out_dir /"preprocessing" / f"{sub}_ses-01_task-tictrack_preprocessed_raw.fif"
        raw.save(out_path, overwrite=True)
        print(f"[OK] Saved preprocessed data: {out_path}")
        print(f"[OK] Saved plots to: {plots_dir}")

        # --- 7. Plot raw data ---
        # --- Output folder ---
        plot_dir = PREPROC_DIR / sub / "preprocessing"/"plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"[5/5] Plotting annotated raw by phase...")
        print(f"\n{'='*60}")
        fif_path = PREPROC_DIR / sub / "preprocessing" / f"{sub}_ses-01_task-tictrack_preprocessed_raw.fif"
        raw_proc = mne.io.read_raw_fif(fif_path, preload=True, verbose="ERROR")

        phases_dict = build_phases_dict(raw)
        plot_raw(
            raw         = raw_proc,
            phases_dict = phases_dict,
            sub_id      = sub,
            plot_dir    = plot_dir,
            window_sec    = 30.0,
            n_channels  = 20,
        )
        print(f"[OK] Saved raw data plots: {plot_dir.name}")

    print("\n[DONE] Preprocessing complete.")


if __name__ == "__main__":
    main()

