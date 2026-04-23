# -------------------------------------------------------------- #
# Function: Create epochs from the start of the tics
# Author: Martyna 
# Goal: Create random and pre-tic epochs from the start times of the tics
# -------------------------------------------------------------- #

"""
We exctract pre-tic epochs and random epochs used as a baseline. To change the duration of each epoch we can change the parameters in the config file. We plot each epoch for inspection. 
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import matplotlib.pyplot as plt
import pandas as pd
import mne
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mne
from config.config import PATIENTS, PREPROC_DIR, EPOCH_EXT_PARAMS, CHANNELS_32, CHANNELS_64, ROI_LIST_32, ROI_LIST_64, ROI_COLORS, ANNOTATION_COLORS, ANNOTATION_COLORS_DEFAULT
from src.epoch_creation import extract_random_epochs_in_phase, extract_pre_tic_epochs

PHASES = ["spontaneous", "imitated", "imitated_real", "suppressed", "suppressed_real"]
PHASE_NAME_MAP = {
    "spontaneous":     "PHASE_FREE",
    "imitated":        "PHASE_MIM",
    "imitated_real":   "PHASE_MIM",
    "suppressed":      "PHASE_SUP",
    "suppressed_real": "PHASE_SUP",
}

# Dataset constants (1 session, 1 run)
TASK = "tictrack"
SES = "01"
RUN = "01"

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


def add_annotations(raw: mne.io.BaseRaw, df: pd.DataFrame):
    # --- Remove FB_ events ---
    df_clean = df[~df["event"].str.startswith("FB_")].copy()
    print(f"[INFO] Annotations after removing FB_: {len(df_clean)} (removed {len(df) - len(df_clean)})")

    # TTLs already in the raw data -- remove them 
    print(f"[INFO] Clearing {len(raw.annotations)} existing raw annotations (Stimulus/S markers)")
    raw.set_annotations(mne.Annotations([], [], []))

    # --- Build and merge annotations ---
    new_annotations = mne.Annotations(
        onset       = df_clean["time"].values,
        duration    = np.zeros(len(df_clean)),
        description = df_clean["event"].values.astype(str),
    )
    raw.set_annotations(raw.annotations + new_annotations)
    print(f"[INFO] Total annotations on raw: {len(raw.annotations)}")

    return raw


    
def plot_raw_epochs_with_roi(raw, epochs, patient, patient_id, phase_name, label,
                                  out_dir: 'Path', 
                                  scalings =dict(eeg=40e-6) ):

    n_epochs = len(epochs)
    if n_epochs == 0:
        print(f"[SKIP] No epochs to plot for {label}")
        return
    
    # Select channels and ROIs based on montage
    if patient.montage == "standard_1020":
        channels_to_use = CHANNELS_32
        roi_lists = ROI_LIST_32
    elif patient.montage == "standard_1005":
        channels_to_use = CHANNELS_64
        roi_lists = ROI_LIST_64
    else:
        raise ValueError(f"Unknown montage: {patient['montage']}")
    
    # Pick ROI channels
    available_channels = [ch for ch in channels_to_use if ch in epochs.ch_names]
    if not available_channels:
        print(f"No ROI channels available in epochs")
        return
    
    epochs_roi = epochs.copy().pick(available_channels)
    
    # Create channel-to-color mapping
    channel_colors = {}
    for roi_name, roi_channels in roi_lists.items():
        for ch in roi_channels:
            if ch in available_channels:
                channel_colors[ch] = ROI_COLORS[roi_name]

    # add colors for annotations
    events_from_annot, event_id_from_annot = mne.events_from_annotations(raw)
    annotation_colors = {}
    for desc, int_id in event_id_from_annot.items():
        if desc in ANNOTATION_COLORS:
            annotation_colors[int_id] = ANNOTATION_COLORS[desc]
        elif desc.startswith("start_"):
            annotation_colors[int_id] = "green"
        elif desc.startswith("end_"):
            annotation_colors[int_id] = "red"
        else:
            annotation_colors[int_id] = ANNOTATION_COLORS_DEFAULT

    
    # Plot each epoch
    for idx in range(len(epochs_roi)):
        events_from_annot, event_id_from_annot = mne.events_from_annotations(raw)
        # Create the plot
        fig = epochs_roi[idx].plot(
            n_epochs=1,
            scalings=20e-6,
            n_channels=len(available_channels),
            title=f"{patient_id}  |  {phase_name}  |  {label}  |  epoch {idx + 1}/{n_epochs}",
            show=False, 
            events = events_from_annot,
            event_id = event_id_from_annot,
            event_color = annotation_colors, 
        )
        
        # Color the channel labels by ROI
        # Get the axes with channel labels
        ax = fig.axes[0]
    
        # Iterate through y-tick labels (channel names) and color them
        for tick_label in ax.get_yticklabels():
            ch_name = tick_label.get_text()
            if ch_name in channel_colors:
                tick_label.set_color(channel_colors[ch_name])
                tick_label.set_weight('bold')
        
        # Add the line at the tic onset t=0
        ax.axvline(x=0, color="red", lw=1.2, linestyle="--")

        # Add legend for ROI colors
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=roi_name) 
                          for roi_name, color in ROI_COLORS.items()]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
        
        # --- Save figures
        out_fig = out_dir / f"{patient_id}_ses-{SES}_task-{TASK}_{phase_name}_{label}_epoch{idx + 1}.png"
        fig.savefig(out_fig, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Saved → {out_fig.name}")


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
        print(f"[START] Creating epochs for {sub}")
        print(f"{'='*60}")


        # ---- 1. Load the racalibrated data ---
        print(f"\n{'='*60}")
        print(f"[1/3] Loading preprocessed raw EEG: ")
        print(f"\n{'='*60}")

        fif_path = PREPROC_DIR / sub / "preprocessing" / f"{sub}_ses-01_task-tictrack_preprocessed_raw.fif"
        raw = mne.io.read_raw_fif(fif_path, preload=True, verbose="ERROR")

        # --- 2. Load the start tic data (tsv) ---
        print(f"\n{'='*60}")
        print(f"[2/3] Loading tic summary (tsv)... ")
        print(f"\n{'='*60}")

        tsv_path = PREPROC_DIR / sub / "tics" / f"{sub}_ses-{SES}_task-{TASK}_tics_summary.tsv"
        if not tsv_path.exists():
            print(f"[SKIP] Missing tic summary: {tsv_path}")
            continue
        df_summary = pd.read_csv(tsv_path, sep="\t")

        merged_path = PREPROC_DIR / sub / "tics" / f"{sub}_ses-{SES}_task-{TASK}_merged_events.tsv"
        if not merged_path.exists():
            print(f"[SKIP] Missing merged events: {merged_path}")
            continue
        df_merged = pd.read_csv(merged_path, sep="\t")

        # --- 3. Create epochs and plot each one ---
        print(f"\n{'='*60}")
        print(f"[3/3] Creating and plotting the epochs ... ")
        print(f"\n{'='*60}")

        for phase in PHASES:
            # --- Output folder ----
            epochs_dir = PREPROC_DIR / sub / "epochs"/ phase
            epochs_dir.mkdir(parents=True, exist_ok=True)

            # --- Output folder for random epochs ---
            rand_epochs_dir = PREPROC_DIR / sub / "epochs"/ "random"/phase
            rand_epochs_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n  → Phase: {phase}")
            df_phase = df_summary[df_summary["phase"] == phase].dropna(subset=["time"])
            if df_phase.empty:
                print(f"  [SKIP] No tics for phase: {phase}")
                continue
            urge_times = df_phase["time"].values
            print(f"{len(urge_times)} tic times found")

            #---- pre-tic epochs ---- #
            pre_tic_epochs = extract_pre_tic_epochs(raw_cropped = raw, urge_times = urge_times, pre_seconds = EPOCH_EXT_PARAMS.get("pre_seconds"), post_seconds = EPOCH_EXT_PARAMS.get("post_seconds"))
            print(f"Pre-tic epochs: {len(pre_tic_epochs)} valid")

            out_epo = epochs_dir / f"{sub}_ses-{SES}_task-{TASK}_{phase}_pretic_epo.fif"
            pre_tic_epochs.save(out_epo, overwrite=True)
            print(f"[OK] Saved → {out_epo.name}")

            # plot each pre-tic epoch 
            plot_raw_epochs_with_roi(
                raw = raw, 
                epochs = pre_tic_epochs,
                patient = cfg,
                patient_id = sub,
                phase_name = phase, 
                label = 'pre-tic',
                out_dir = epochs_dir,
                scalings = 'auto')
            
            # --- random epochs --- #
            phase_key = PHASE_NAME_MAP.get(phase)
            if phase_key is None:
                print(f"  [SKIP] No phase mapping for: {phase}")
                continue

            phase_start_vals = df_merged[df_merged["event"] == f"start_{phase_key}"]["time"].values
            phase_end_vals   = df_merged[df_merged["event"] == f"end_{phase_key}"]["time"].values

            phase_start = phase_start_vals[0]   # ← scalar not array
            phase_end   = phase_end_vals[0]      # ← scalar not array
            print(f"  Phase boundaries: {phase_start:.2f}s → {phase_end:.2f}s")
            random_epochs = extract_random_epochs_in_phase(
                raw_cropped    = raw,
                start_time     = phase_start,   
                end_time       = phase_end,    
                n_epochs       = EPOCH_EXT_PARAMS["random_n_epochs"],
                epoch_duration = EPOCH_EXT_PARAMS["random_epoch_duration"],
                event_id       = 1,
                seed           = None,
            )

            print(f"Random epochs: {len(random_epochs)} valid")

            out_rand = rand_epochs_dir / f"{sub}_ses-{SES}_task-{TASK}_{phase}_random_epo.fif"
            random_epochs.save(out_rand, overwrite=True)
            print(f"[OK] Saved → {out_rand.name}")
            plot_raw_epochs_with_roi(
                raw        = raw,
                epochs     = random_epochs,
                patient    = cfg,
                patient_id = sub,
                phase_name = phase,
                label      = "random",
                out_dir    = rand_epochs_dir,
            )



        print("\n[DONE] Epoch creation complete.")


if __name__ == "__main__":
    main()











