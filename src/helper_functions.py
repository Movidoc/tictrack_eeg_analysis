# ------------------------------------------- #
# Function: Helper functions  
# Author: Martyna
# Goal : Function to exctract TTL events from raw data and get a summary of each and its time, plot raw data for inspection
# -------------------------------------------- #
import mne
import pandas as pd
import re
from config.config import PHASES_TTL, TTL_MAP, ANNOTATION_COLORS, ANNOTATION_COLORS_DEFAULT
from pathlib import Path
import matplotlib.pyplot as plt


BV_STIM_RE = re.compile(r"^Stimulus/S\s+\d+$")

def extract_ttl_events(raw) :
    """
    Convert MNE annotations to a tidy TTL events table.
    Further add phase labels to each event 
    Why annotations?
    - BrainVision markers are loaded by MNE as raw.annotations (onset/duration/description).
    We keep only Stimulus TTL-like descriptions and map them using TTL_MAP.
    """
    ann = raw.annotations
    rows = []

    # Build phase intervals from annotations
    phase_intervals = {}
    for phase_name, t in PHASES_TTL.items():
        start = [onset for onset, desc in zip(ann.onset, ann.description) if desc == t["start"]]
        end   = [onset for onset, desc in zip(ann.onset, ann.description) if desc == t["end"]]
        if start and end:
            phase_intervals[phase_name] = (start[0], end[0])
        else:
            phase_intervals[phase_name] = None

    for onset, duration, desc in zip(ann.onset, ann.duration, ann.description):
        if not BV_STIM_RE.match(desc):
            continue

        trial_type = TTL_MAP.get(desc, "UNKNOWN")

        # find which phase this TTL belongs to
        phase = None
        for phase_name, interval in phase_intervals.items():
            if interval is None:
                continue
            start_t, end_t = interval
               # last phase includes end, all others exclude it
            if phase_name == list(phase_intervals.keys())[-1]:
                if start_t <= onset <= end_t:
                    phase = phase_name
                    break
            else:
                if start_t <= onset < end_t:
                    phase = phase_name
                    break

        rows.append(
            {
                "onset": float(onset),
                "duration": float(duration) if duration is not None else 0.0,
                "trial_type": trial_type,  # interpreted label
                "value": desc,             # raw BrainVision marker string
                "phase": phase,           # new column with phase label
            }
        )

    df = pd.DataFrame(rows).sort_values("onset").reset_index(drop=True)
    return df

def build_phases_dict(raw: mne.io.BaseRaw) -> dict:
    """
    Build phases_dict from mapped annotation names (post-alignment).
    Looks for start_PHASE_XXX and end_PHASE_XXX annotations.
    """
    phases_dict = {}
    main_phases = ["PHASE_FREE", "PHASE_MIM", "PHASE_SUP"]

    for phase_name in main_phases:
        start = [
            onset for onset, desc in zip(raw.annotations.onset, raw.annotations.description)
            if desc == f"start_{phase_name}"
        ]
        end = [
            onset for onset, desc in zip(raw.annotations.onset, raw.annotations.description)
            if desc == f"end_{phase_name}"
        ]

        if start and end:
            phases_dict[phase_name] = (start[0], end[0])
            print(f"[OK] Phase '{phase_name}': {start[0]:.2f}s → {end[0]:.2f}s")
        else:
            print(f"[WARN] Missing annotations for phase '{phase_name}'")
            phases_dict[phase_name] = None

    return phases_dict

def get_annotation_colors(raw: mne.io.BaseRaw) -> dict:
    """
    Consistent annotation colors when plotting.
    Returns dict with integer event IDs as keys (required by raw.plot).
    """
    # Convert annotations to events array + id mapping
    events, event_id = mne.events_from_annotations(raw, verbose=False)

    colors = {}
    for desc, int_id in event_id.items():
        if desc in ANNOTATION_COLORS:
            colors[int_id] = ANNOTATION_COLORS[desc]
        elif desc.startswith("start_"):
            colors[int_id] = "red"
        elif desc.startswith("end_"):
            colors[int_id] = "green"
        else:
            colors[int_id] = ANNOTATION_COLORS_DEFAULT

    return events, event_id, colors

def plot_raw(
    raw: mne.io.BaseRaw,
    phases_dict: dict,
    sub_id: str,
    plot_dir: Path,
    window_sec: float = 60.0,
    n_channels: int = 20,
):
    """
    For each phase, plot consecutive 60s windows from phase start to phase end.
    One PNG per window.
    """
    main_phases = ["PHASE_FREE", "PHASE_MIM", "PHASE_SUP"]
    phases_to_plot = {k: v for k, v in phases_dict.items() if k in main_phases and v is not None}

    if not phases_to_plot:
        print(f"[SKIP] No valid phases found for {sub_id}")
        return

    for phase_name, (phase_start, phase_end) in phases_to_plot.items():

        # --- Output folder ---
        phase_plot_dir = plot_dir / phase_name
        phase_plot_dir.mkdir(parents=True, exist_ok=True)

        # --- Compute windows for this phase ---
        windows = []
        t = phase_start
        while t < phase_end:
            w_end = min(t + window_sec, phase_end)
            windows.append((t, w_end))
            t += window_sec

        print(f"→ {phase_name}: {len(windows)} windows")

        events, event_id, annotation_colors = get_annotation_colors(raw)
        #remove temporarily 
        orig_annotations = raw.annotations.copy()
        raw.set_annotations(mne.Annotations([], [], []))


        for win_idx, (t_start, t_end) in enumerate(windows, start=1):

            fig = raw.plot(
                start      = t_start,
                duration   = window_sec,
                n_channels = n_channels,
                show       = False,
                events = events, 
                event_id = event_id,
                event_color = annotation_colors, 
                title      = f"{sub_id}  |  {phase_name}  |  window {win_idx}/{len(windows)}  |  t={t_start:.1f}s → {t_end:.1f}s",
            )

        
            for ax in fig.axes:
                for text in ax.texts:
                    label = text.get_text()
                    if label in event_id:
                        int_id = event_id[label]
                        text.set_color(annotation_colors.get(int_id, ANNOTATION_COLORS_DEFAULT))

            for ax in fig.axes:
                for text in ax.texts:
                    text.set_fontsize(5)
                    text.set_rotation(90)



            plt.tight_layout()
            out_fig = phase_plot_dir / f"{sub_id}_ses-01_{phase_name}_window{win_idx:02d}.png"
            fig.savefig(out_fig, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"[OK] Saved → {out_fig.name}")
        raw.set_annotations(orig_annotations)

import pandas as pd
import numpy as np

def compute_start_to_key_d_delays(df, max_delay=3.0, same_phase=True):
    """
    Pair each Excel start_* event with the closest following KEY_D event.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns: event, time, phase, source.

    max_delay : float
        Maximum allowed delay in seconds between start_* and KEY_D.
        Use this to avoid pairing unrelated events.

    same_phase : bool
        If True, only match KEY_D events from the same phase.

    Returns
    -------
    pairs : pandas.DataFrame
        One row per matched start_* event.
    """

    df = df.copy()
    df["time"] = pd.to_numeric(df["time"], errors="coerce")

    # Excel start events only: start_24, start_25, etc.
    starts = df[
        df["event"].astype(str).str.match(r"^start_\d+$")
        & (df["source"] == "Excel")
    ].copy()

    # EEG KEY_D events
    key_d = df[df["event"] == "KEY_D"].copy()

    results = []

    for _, start_row in starts.iterrows():
        start_time = start_row["time"]
        start_phase = start_row["phase"]
        start_event = start_row["event"]

        candidates = key_d[key_d["time"] >= start_time].copy()

        if same_phase:
            candidates = candidates[candidates["phase"] == start_phase]

        if candidates.empty:
            continue

        candidates["delay_s"] = candidates["time"] - start_time

        # keep only close events
        candidates = candidates[candidates["delay_s"] <= max_delay]

        if candidates.empty:
            continue

        # nearest following KEY_D
        best = candidates.sort_values("delay_s").iloc[0]

        results.append({
            "start_event": start_event,
            "phase": start_phase,
            "start_time": start_time,
            "key_d_time": best["time"],
            "delay_s": best["delay_s"],
        })

    return pd.DataFrame(results)


def plot_alpha_spectrum_eo_ec_no_tic_by_roi(
    no_tic_epochs: mne.Epochs,
    patient,
    patient_id: str,
    out_dir: Path,
    fmin: float = 2.0,
    fmax: float = 30.0,
    alpha_band: tuple[float, float] = (8.0, 12.0),
):
    """
    Plot power spectrum for no-tic epochs during eyes open and eyes closed,
    separately for each ROI.

    For each ROI, the function:
    - selects ROI channels
    - selects PHASE_EO and PHASE_EC no-tic epochs
    - computes Welch PSD
    - averages across epochs and ROI channels
    - plots EO vs EC spectrum
    - prints mean alpha power difference
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if no_tic_epochs.metadata is None:
        print("[SKIP] no_tic_epochs has no metadata.")
        return

    if "phase" not in no_tic_epochs.metadata.columns:
        print("[SKIP] no_tic_epochs metadata does not contain a 'phase' column.")
        return

    # Select ROI list based on montage
    if patient.montage == "standard_1020":
        roi_lists = ROI_LIST_32
    elif patient.montage == "standard_1005":
        roi_lists = ROI_LIST_64
    else:
        raise ValueError(f"Unknown montage: {patient.montage}")

    # Select no-tic EO and EC epochs once
    eo_epochs_all = no_tic_epochs[
        no_tic_epochs.metadata["phase"].values == "PHASE_EO"
    ]

    ec_epochs_all = no_tic_epochs[
        no_tic_epochs.metadata["phase"].values == "PHASE_EC"
    ]

    if len(eo_epochs_all) == 0:
        print(f"[SKIP] No PHASE_EO no-tic epochs for {patient_id}.")
        return

    if len(ec_epochs_all) == 0:
        print(f"[SKIP] No PHASE_EC no-tic epochs for {patient_id}.")
        return

    print(f"[INFO] {patient_id}: EO no-tic epochs = {len(eo_epochs_all)}")
    print(f"[INFO] {patient_id}: EC no-tic epochs = {len(ec_epochs_all)}")

    results = []

    for roi_name, roi_channels in roi_lists.items():

        available_roi_channels = [
            ch for ch in roi_channels
            if ch in no_tic_epochs.ch_names
        ]

        if len(available_roi_channels) == 0:
            print(f"[SKIP] {patient_id} | {roi_name}: no available channels.")
            continue

        print(f"\n[INFO] Processing ROI: {roi_name}")
        print(f"[INFO] Channels used: {available_roi_channels}")

        eo_epochs = eo_epochs_all.copy().pick(available_roi_channels)
        ec_epochs = ec_epochs_all.copy().pick(available_roi_channels)

        # Compute PSD
        psd_eo = eo_epochs.compute_psd(
            method="welch",
            fmin=fmin,
            fmax=fmax,
            picks=available_roi_channels,
            verbose=False,
        )

        psd_ec = ec_epochs.compute_psd(
            method="welch",
            fmin=fmin,
            fmax=fmax,
            picks=available_roi_channels,
            verbose=False,
        )

        freqs = psd_eo.freqs

        # Shape: epochs x channels x freqs
        power_eo = psd_eo.get_data()
        power_ec = psd_ec.get_data()

        # Average across epochs and channels
        mean_eo = power_eo.mean(axis=(0, 1))
        mean_ec = power_ec.mean(axis=(0, 1))

        # Log-transform for visualization
        mean_eo_log = np.log10(mean_eo)
        mean_ec_log = np.log10(mean_ec)

        # Alpha band summary
        alpha_mask = (freqs >= alpha_band[0]) & (freqs <= alpha_band[1])

        alpha_eo = mean_eo_log[alpha_mask].mean()
        alpha_ec = mean_ec_log[alpha_mask].mean()
        alpha_diff = alpha_ec - alpha_eo

        print(f"[RESULT] {patient_id} | {roi_name} | alpha EO: {alpha_eo:.4f}")
        print(f"[RESULT] {patient_id} | {roi_name} | alpha EC: {alpha_ec:.4f}")
        print(f"[RESULT] {patient_id} | {roi_name} | EC - EO: {alpha_diff:.4f}")

        results.append({
            "subject": patient_id,
            "roi": roi_name,
            "n_eo_epochs": len(eo_epochs),
            "n_ec_epochs": len(ec_epochs),
            "channels": ", ".join(available_roi_channels),
            "alpha_eo_log10": alpha_eo,
            "alpha_ec_log10": alpha_ec,
            "alpha_ec_minus_eo_log10": alpha_diff,
        })

        # Plot
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(freqs, mean_eo_log, label=f"Eyes open no-tic, n={len(eo_epochs)}")
        ax.plot(freqs, mean_ec_log, label=f"Eyes closed no-tic, n={len(ec_epochs)}")

        ax.axvspan(
            alpha_band[0],
            alpha_band[1],
            alpha=0.2,
            label="Alpha band 8-12 Hz"
        )

        ax.set_title(f"{patient_id}: {roi_name} spectrum during no-tic epochs")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Log10 power")
        ax.legend()
        ax.grid(True, alpha=0.3)

        safe_roi_name = str(roi_name).replace(" ", "_").replace("/", "-")

        fname = (
            f"{patient_id}_ses-{SES}_task-{TASK}"
            f"_no_tic_EO_vs_EC_{safe_roi_name}_alpha_spectrum.png"
        )

        fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"[OK] Saved ROI spectrum plot -> {fname}")

    # Save summary CSV
    if results:
        results_df = pd.DataFrame(results)

        csv_name = (
            f"{patient_id}_ses-{SES}_task-{TASK}"
            f"_no_tic_EO_vs_EC_alpha_by_roi.csv"
        )

        results_df.to_csv(out_dir / csv_name, index=False)
        print(f"\n[OK] Saved ROI alpha summary -> {csv_name}")