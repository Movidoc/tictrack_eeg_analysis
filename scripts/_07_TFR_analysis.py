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

        # --- 1. Load preproccessed data 