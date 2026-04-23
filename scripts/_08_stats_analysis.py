# ---------------------------------------------- #
# Function: Statistical analysis of TFR results
# Author: Martyna Siatka
# -------------------------------------------- #
""""
For each condition and each ROI we will do the statistical test to find the clusters that show statistically significance difference across time and frequency
"""

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
    TFR_PARAMS, STAT_PARAMS,
)
from src.time_frequency_analysis import tfr_per_ROI_normalized
from src.stats_analysis import cluster_stats, plot_cluster_results, between_cluster_stats


TASK   = "tictrack"
SES    = "01"
RUN = "01"
PHASES = ["PHASE_FREE", "PHASE_MIM", "PHASE_SUP"]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Statistical analysis of each phase and between different conditions"
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
        tfr_data = {}

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

        for phase in phases:
            tfr_data[phase] = {}
            sel = no_tic_epochs.metadata["phase"] == phase
            random_epochs = no_tic_epochs[sel]
            print(f"[INFO] Baseline: {phase} ({len(random_epochs)} no-tic epochs)")
            if len(random_epochs) < 5:
                print(f"[WARN] Only {len(random_epochs)} no-tic epochs for {phase}, "
                    f"falling back to PHASE_FREE baseline.")
                sel = no_tic_epochs.metadata["phase"] == "PHASE_FREE"
                random_epochs = no_tic_epochs[sel]
                print(f"[INFO] Fallback baseline: PHASE_FREE ({len(random_epochs)} no-tic epochs)")

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
                # print(f"  {phase} | {tic}: tmin={tic_epochs_sel.tmin}, tmax={tic_epochs_sel.tmax}, n={len(tic_epochs_sel)}")
                # print(f"  pre_tic tmin={tic_epochs_sel.tmin}, tmax={tic_epochs_sel.tmax}")
                # print(f"  random  tmin={random_epochs.tmin},  tmax={random_epochs.tmax}")

                # Compute TFR
                roi_tfr, freqs, times, n , X = tfr_per_ROI_normalized(
                    patient = cfg, pre_tic_epochs = tic_epochs_sel, 
                    random_epochs = random_epochs, epoch_type='pre_tic', freqs=TFR_PARAMS["freqs"], normalization =TFR_PARAMS["normalization"] )
                """
                roi_tfr : dictionary of each ROI with (freqs, times) 
                X : dictionary of each ROI with (epochs, freqs, times)
                """
                tfr_data[phase][tic] = {
                    "roi_tfr": roi_tfr,
                    "X": X,
                    "freqs": freqs,
                    "times": times,
                    "n": n
                }

                # ---3. Permutation Cluster 1 Sample Test
                print(f"\n{'='*60}")
                print(f"[3/5] Permutation Cluster 1 Sample Test and Plotting for {phase} & {tic}...")
                print(f"{'='*60}")

                cluster_results = cluster_stats(X, n_permutations=STAT_PARAMS['n_permutations'], tail= STAT_PARAMS['tail'], threshold = STAT_PARAMS['threshold'], correction =STAT_PARAMS['correction'])
                fig = plot_cluster_results(roi_tfr, cluster_results, freqs, times, n, phase, tic, sub)

                # ----------- Output folder ---------
                out_dir = PREPROC_DIR / sub / "stats" / 'cluster_1sample'/ phase
                out_dir.mkdir(parents=True, exist_ok=True)
                fname = f"{sub}_{phase}_{tic}tfr.png"
                fig.savefig(out_dir / fname, dpi=150)
                plt.close(fig)

            print(f"[OK] Saved → {fname}")

            # --- 4. Permutation cluster test ---
            print(f"\n{'='*60}")
            print(f"[4/5] Permutation Cluster Test and Plotting for {phase}...")
            print(f"{'='*60}")
            if phase == "PHASE_MIM":
                cond1, cond2 = "mimicked", "expressed"
            elif phase == "PHASE_SUP":
                cond1, cond2 = "suppressed", "expressed"
            else:
                continue

            if cond1 not in tfr_data[phase] or cond2 not in tfr_data[phase]:
                continue

            X1 = tfr_data[phase][cond1]["X"]
            X2 = tfr_data[phase][cond2]["X"]
            comparison =  f"{cond1} vs {cond2}"
            print(f"[INFO] Comparing {cond1} vs {cond2}")


            between_cluster_results = between_cluster_stats(X1, X2, n_permutations=STAT_PARAMS['n_permutations'], tail= STAT_PARAMS['tail'], threshold = STAT_PARAMS['threshold'], correction =STAT_PARAMS['correction'])
            fig2 = plot_cluster_results(roi_tfr, between_cluster_results, freqs, times, n, phase, comparison, sub)

            # ----- output folder -----
            out_dir2 = PREPROC_DIR / sub / "stats" / 'between_cluster'/ phase
            out_dir2.mkdir(parents=True, exist_ok=True)
            fname2 = f"{sub}_{phase}_{comparison}tfr.png"
            fig2.savefig(out_dir2 / fname2, dpi=150)
            plt.close(fig2)


        print(f"[OK] Saved → {fname2}")

    print("\n[DONE] All data saved.")

            


if __name__ == "__main__":
    main()
       
