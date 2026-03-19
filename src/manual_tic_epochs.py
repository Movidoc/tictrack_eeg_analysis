# --------------------------------------------- #
# Function: Exctract events for tic epochs
# Author: Martyna
# ---------------------------------------------- #
"""
After the inspection of the events TTLs & excel annotations the start and end of each tic was manually annotated. We will use the start of the tic as the center of the epoch. However, we will study the signal precedeing the tic onset. 
For the ICA we will use the gaps in between the tics to make sure we do not reject muscle or eye signal present during the tic. We find artefacts in the signal that does not capture the tic signal.
"""
import numpy as np
import mne
from config.config import PHASES_TTL, EPOCH_EXT_PARAMS



def no_tic_gaps(raw, tics_df, phase_boundaries  = None, epoch_duration = 2.0, min_gap = 2.0):
    """
    We only extract the in-between tics gaps from EYES_OPEN, EYES_CLOSED, and PHASE_SUP phases.That way we will be able to capture eye blink artefacts and muscle artefacts.  
    """
    used_phases = ["PHASE_EC","PHASE_EO","PHASE_FREE"]
    epochs_onsets = []
    for phase in used_phases:
        if phase not in PHASES_TTL:
            print(f"Phase {phase} not found in phases_ttl, skipping.")
            continue

            
        print(f"  phase_boundaries keys: {list(phase_boundaries.keys())}")
        print(f"  used_phases: {used_phases}")

        phase_start = phase_boundaries[phase]["start"]
        phase_end = phase_boundaries[phase]["end"]

        # get tics within this phase, sorted by start time
        phase_tics = tics_df[tics_df["phase"] == phase]
    
        tic_intervals = list(zip(phase_tics["start"], phase_tics["end"])) # (start, end)

        boundaries = [(phase_start, phase_start)] + tic_intervals + [(phase_end, phase_end)] 

        for i in range(len(boundaries) - 1):
            gap_start = boundaries[i][1] # end of the current tic 
            gap_end = boundaries[i+1][0] # start of the next tic
            gap_duration = gap_end - gap_start 

            # do not consider epochs shorter than 2s
            if gap_duration < min_gap:
                print(f"  Skipping gap of {gap_duration:.2f}s in {phase} (too short)")
                continue

            # split the long gaps into epoch_duration interval 
            n_epochs = int(gap_duration // epoch_duration) 
            
            if n_epochs==0:
                epoch_onsets.append(gap_start)
            else:
                for j in range(n_epochs):
                    epoch_onset = gap_start +j*epoch_duration
                    epochs_onsets.append(epoch_onset)
                    
    # build an events array
    events = []
    for t in epochs_onsets:
        sample_idx = int(t * raw.info['sfreq'])
        events.append([sample_idx, 0, 2])
    events = np.array(events, dtype=int)

    # create epochs 
    epochs = mne.Epochs(
    raw, 
    events,
    event_id={"gap": 2},
    tmin=0, 
    tmax=epoch_duration,
    baseline=None, 
    preload=True
    )
    return epochs 

def create_tic_epochs(raw, tics_df, phase_boundaries  = None):
    """
    From the excel file with manually annotated tics we create epochs surrounded around the strat of the tic. 
    """
    used_phases = ["PHASE_FREE", "PHASE_MIM", "PHASE_SUP"]
    epochs_onsets = []
    epochs_phase = []
    epochs_type = []
    epochs_annot_type = []

    for phase in used_phases:
        if phase not in phase_boundaries:
            print(f"  [SKIP] {phase} not found in phase_boundaries.")
            continue

        phase_tics = tics_df[tics_df["phase"]== phase]
        print(f"phase={phase}, n_tics={len(phase_tics)}")  # ← add this
        
        for _, column in phase_tics.iterrows():
            epochs_onsets.append(column["start"])
            epochs_phase.append(phase)
            epochs_type.append(column['tic_type'])
            epochs_annot_type.append(column["annot_type"])

        
        print(f"[OK] {len(phase_tics)} tics found in the {phase} ")
    
    # build events array 
    sfreq  = raw.info["sfreq"]
    events = np.array(
        [[int(t * sfreq), 0, 1] for t in epochs_onsets],
        dtype=int
    )

    # create epochs from the events 
    epochs = mne.Epochs(
        raw,
        events, 
        event_id={"tic":1},
        tmin = EPOCH_EXT_PARAMS["pre_seconds"],
        tmax = EPOCH_EXT_PARAMS["post_seconds"],
        baseline = None, 
        preload = True
    )
    return epochs, epochs_phase, epochs_type,  epochs_annot_type











