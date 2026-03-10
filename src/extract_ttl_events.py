# ------------------------------------------- #
# Function: Extract_ttl_events 
# Author: Martyna
# Goal : Function to exctract TTL events from raw data and get a summary of each and its time
# -------------------------------------------- #
import mne
import pandas as pd
import re
from config.config import PHASES_TTL, TTL_MAP

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
