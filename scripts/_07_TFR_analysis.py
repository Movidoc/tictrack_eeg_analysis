# ------------------------------------------------ #
# Function: Time_Frequency Analysis
# Author: Martyna
# ------------------------------------------------ #

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import mne
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.config import (
    PREPROC_DIR, PATIENTS, PHASES_TTL,
    TFR_PARAMS
)
from src.time_frequency_analysis import tfr_per_ROI_normalized, plot_trf_roi

TASK   = "tictrack"
SES    = "01"
RUN = "01"
PHASES = ["PHASE_FREE", "PHASE_MIM", "PHASE_SUP"]

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
    args   = parse_args()

    if args.sub:
        subjects = {k: v for k, v in PATIENTS.items() if k in args.sub}
        missing  = [s for s in args.sub if s not in PATIENTS]
        if missing:
            raise RuntimeError(f"Subjects not found in config: {missing}")
    else:
        subjects = PATIENTS

    for sub, cfg in subjects.items():

        # --- 1. Load preprocessed epochs data 
        print(f"\n{'='*60}")
        print(f"[1/5] Loading epochs data ...")
        print(f"{'='*60}")
        # -------------- Pre-tic epochs -----------
        epochs_fif_path = PREPROC_DIR / sub / "tics_manual"/ "tic" / f"{sub}_ses-01_task-tictrack_tic_epo.fif"
        tic_epochs = mne.read_epochs(epochs_fif_path, preload = True, verbose = "ERROR")

        if tic_epochs.metadata is None:
            print("[ERROR] Epochs must contain metadata with phase and tic_type")
        phases = tic_epochs.metadata["phase"].unique()
        tic_types = tic_epochs.metadata["tic_type"].unique()

        print(f"[INFO] Phases: {phases}")
        print(f"[INFO] Tic types: {tic_types}")

        # ------------- No-tic epochs-----------
        no_epochs_fif = PREPROC_DIR / sub / "tics_manual" /"no_tic" / f"{sub}_ses-01_task-tictrack_no_tic_epo.fif"
        no_tic_epochs = mne.read_epochs(no_epochs_fif, preload = True, verbose = "ERROR")

        print(f"[INFO] Loaded {len(tic_epochs)} tic epochs")
        print(f"[INFO] Loaded {len(no_tic_epochs)} no-tic epochs")

        # -- 2. Time-Frequency Analysis 
        print(f"\n{'='*60}")
        print(f"[2/5] Time-Frequency Analysis ...")
        print(f"{'='*60}")

        # # choose the phase for no_tic epochs 
        # baseline_phase = args.baseline_phase
        # if baseline_phase == "ALL":
        #     random_epochs = no_tic_epochs
        # else:
        #     if no_tic_epochs.metadata is None:
        #         print("[ERROR] No metadata found in no_tic_epochs")

        #     sel = no_tic_epochs.metadata["phase"] == baseline_phase
        #     random_epochs = no_tic_epochs[sel]

        #     print(f"[INFO] Baseline: {baseline_phase} ({len(random_epochs)} epochs)")

        print(f"\n{'='*60}")
        print(f"[2/2] Plotting TFR Analysis results...")
        print(f"{'='*60}")

        for phase in phases:
            sel = no_tic_epochs.metadata["phase"] == phase
            random_epochs = no_tic_epochs[sel]
            print(f"[INFO] Baseline: {phase} ({len(random_epochs)} no-tic epochs)")

            for tic in tic_types: 
                print(f"\n--- {phase} | {tic} ---")
                # choose the tic type for each phase
                select = (
                    (tic_epochs.metadata["phase"] == phase) & 
                    (tic_epochs.metadata["tic_type"] == tic)
                )
                tic_epochs_sel = tic_epochs[select]
                if len(tic_epochs_sel) == 0:
                    print(f"  [SKIP] No epochs for {phase} | {tic}, skipping.")
                    continue
                print(f"  {phase} | {tic}: tmin={tic_epochs_sel.tmin}, tmax={tic_epochs_sel.tmax}, n={len(tic_epochs_sel)}")
                print(f"  pre_tic tmin={tic_epochs_sel.tmin}, tmax={tic_epochs_sel.tmax}")
                print(f"  random  tmin={random_epochs.tmin},  tmax={random_epochs.tmax}")

                # Compute TFR
                roi_tfr, freqs, times, n  = tfr_per_ROI_normalized(
                    patient = cfg, pre_tic_epochs = tic_epochs_sel, 
                    random_epochs = random_epochs, epoch_type='pre_tic', freqs=TFR_PARAMS["freqs"], normalization =TFR_PARAMS["normalization"] )

            # -- 3. Plotting the TFR results
                fig = plot_trf_roi(roi_tfr, freqs, times, n, epoch_type='pre_tic', vmin=None, vmax=None)

                # save the ouputs 
                out_dir = PREPROC_DIR / sub / "tfr" / phase
                out_dir.mkdir(parents=True, exist_ok=True)

                fname = f"{sub}_{phase}_{tic}_tfr.png"
                fig.savefig(out_dir / fname, dpi=150)
                plt.close(fig)

                # --- 4. Plot each epochs seperately 
                print(f"\n{'='*60}")
                print(f"[4/5] Plotting each epoch separately  ...")
                print(f"{'='*60}")
                for i in range(len(tic_epochs_sel)):

                    # -- Time-Frequency--
                    sin_tfr, sin_freqs, sin_times, sin_n  = tfr_per_ROI_normalized(
                    patient = cfg, pre_tic_epochs = tic_epochs_sel[i], 
                    random_epochs = random_epochs, epoch_type='pre_tic', freqs=TFR_PARAMS["freqs"], normalization =TFR_PARAMS["normalization"] )

                    # -- Plot each epoch ---
                    fig = plot_trf_roi(sin_tfr, sin_freqs, sin_times, sin_n, epoch_type='pre_tic', vmin=None, vmax=None)

                    # --- output dir ---
                    out_dir = PREPROC_DIR / sub / "tfr" / 'single_epoch'/ phase
                    out_dir.mkdir(parents=True, exist_ok=True)

                    fname = f"{sub}_{phase}_{tic}_{i}_tfr.png"
                    fig.savefig(out_dir / fname, dpi=150)
                    plt.close(fig)


        # -- 4. Analysis for no_tic epochs ---
        roi_tfr_nt, freqs_nt, times_nt, n_nt  = tfr_per_ROI_normalized(
            patient = cfg, pre_tic_epochs = random_epochs, 
            random_epochs = random_epochs, epoch_type='random', freqs=TFR_PARAMS["freqs"], normalization =TFR_PARAMS["normalization"] )
        fig_nt = plot_trf_roi(roi_tfr_nt, freqs_nt, times_nt, n_nt, epoch_type='random', vmin=None, vmax=None)

        # save the ouputs 
        out_dir_nt = PREPROC_DIR / sub / "tfr" / "no_tic"
        out_dir_nt.mkdir(parents=True, exist_ok=True)

        fname_nt = f"{sub}_baseline_tfr.png"
        fig_nt.savefig(out_dir_nt / fname_nt, dpi=150)
        plt.close(fig_nt)




        print(f"[OK] Saved → {fname}")
    print(f"[OK] Saved → {fname_nt}")


    print("\n[DONE] All data saved.")

            


if __name__ == "__main__":
    main()
       
