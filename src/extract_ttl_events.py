# ------------------------------------------- #
# Function: Helper functions  
# Author: Martyna
# Goal : Function to exctract TTL events from raw data and get a summary of each and its time, plot raw data for inspection
# -------------------------------------------- #
import mne
import pandas as pd
import re
from config.config import PHASES_TTL, TTL_MAP
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

        for win_idx, (t_start, t_end) in enumerate(windows, start=1):

            fig = raw.plot(
                start      = t_start,
                duration   = window_sec,
                n_channels = n_channels,
                show       = False,
                title      = f"{sub_id}  |  {phase_name}  |  window {win_idx}/{len(windows)}  |  t={t_start:.1f}s → {t_end:.1f}s",
            )
            for ax in fig.axes:
                for text in ax.texts:
                    text.set_fontsize(7)
                    text.set_rotation(90)  # vertical so they don't overlap horizontally



            plt.tight_layout()
            out_fig = phase_plot_dir / f"{sub_id}_ses-01_{phase_name}_window{win_idx:02d}.png"
            fig.savefig(out_fig, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"[OK] Saved → {out_fig.name}")