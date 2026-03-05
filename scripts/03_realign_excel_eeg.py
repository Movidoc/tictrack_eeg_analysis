# -----------------------------------------------------------
# Function: realingn the Excel file times with EEG recording times (based on the green led)
# Author: Martyna (structure) & Indira (code)
# Goal: to create a new Excel file with the realigned times
# ------------------------------------------------------------

from __future__ import annotations

import argparse
from pathlib import Path
import re
from collections import Counter
import pandas as pd
import mne
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mne
from config.config import PATIENTS, PIPE_PARAMS, PREPROC_DIR, DATASET_DIR, ICA_EXCLUSIONS
from src.align_eeg_excel import recalibrate_from_first_event


# Dataset constants (1 session, 1 run)
TASK = "tictrack"
SES = "01"
RUN = "01"

# BrainVision "Stimulus" marker strings look like: "Stimulus/S  3"
BV_STIM_RE = re.compile(r"^Stimulus/S\s+\d+$")


def bids_eeg_dir(sub: str) -> Path:
    return DATASET_DIR / sub / f"ses-{SES}" / "excel"

def bids_vhdr_path(sub: str) -> Path:
    return DATASET_DIR / sub / f"ses-{SES}" / "eeg"/ f"{sub}_task-{TASK}.vhdr"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract BrainVision TTL markers to BIDS-like events.tsv"
    )
    parser.add_argument(
        "--sub",
        nargs="*",
        default=None,
        help="Subjects to process, e.g. --sub sub-028 sub-029. Default: all dataset/sub-*",
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
        print(f"[START] Aligning {sub}")
        print(f"{'='*60}")

        # --- Output folder ---
        # out_dir = PREPROC_DIR / sub_id 
        # out_dir.mkdir(parents=True, exist_ok=True)

        realign_dir = PREPROC_DIR /sub / "realign"
        realign_dir.mkdir(parents=True, exist_ok=True)

        # --- 1. Load preprocessed EEG ---
        print(f"\n{'='*60}")
        print(f"[1/6] Loading preprocessed EEG: ")
        print(f"\n{'='*60}")
        vhdr = bids_vhdr_path(sub)
        if not vhdr.exists():
            print(f"[SKIP] Missing vhdr for {sub}: {vhdr}")
            continue

        # Load without preload (fast). We only need annotations.
        raw = mne.io.read_raw_brainvision(vhdr, preload=False, verbose="ERROR")

        # preproc_path = out_dir / f"{sub_id}_ses-01_task-tictrack_preprocessed-raw.fif"
        # raw = mne.io.read_raw_fif(preproc_path, preload=True, verbose="ERROR")

        # --- 2. Recalibrate EEG to S2 ---
        print(f"\n{'='*60}")
        print(f"[2/6] Recalibrating EEG to Stimulus/S  2...")
        print(f"\n{'='*60}")
        raw_cropped = recalibrate_from_first_event(raw, target_stim="Stimulus/S  2")
        # --- Save recalibrated raw ---
        recal_path = realign_dir / f"{sub}_ses-01_task-tictrack_aligned_raw.fif"
        raw_cropped.save(recal_path, overwrite=True)
        print(f"[OK] Saved recalibrated raw: {recal_path}")



if __name__ == "__main__":
    main()
