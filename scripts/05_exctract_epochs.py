# ❀ ---------------------------------------------- ❀
# Project : TicTrack EEG - Premonitory Urge & Tic Suppression
# Author  : Martyna
# Module  : scripts/05_summarize_tics.py
# Goal    : Summarize tics from merged events file
#           using phase-specific analysis functions
# ❀ ---------------------------------------------- ❀

from __future__ import annotations

import sys
import os
import argparse
import pandas as pd
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.config import PATIENTS, PREPROC_DIR
from src.tic_exctraction import (
    analyse_merged_ttl_tics_spontaneous,
    analyse_merged_ttl_tics_imitated,
    analyse_merged_ttl_tics_suppressed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize tics from merged events file"
    )
    parser.add_argument(
        "--sub",
        nargs="*",
        default=None,
        help="Subjects to process, e.g. --sub sub-BB28 sub-BC29. Default: all patients",
    )
    return parser.parse_args()

def build_merged_ttl_tics_from_tsv(df: pd.DataFrame) -> list:
    """
    Convert merged events .tsv dataframe to list of dicts
    format expected by the analysis functions.
    Excludes FB_ events (visual feedback).
    """
    merged_ttl_tics = []
    for _, row in df.iterrows():
        if str(row["event"]).startswith("FB_"):  # skip visual feedback events
            continue
        merged_ttl_tics.append({row["event"]: row["time"]})
    return merged_ttl_tics


def results_to_df(results: list, sub_id: str, phase: str) -> pd.DataFrame:
    """
    Convert results list to a tidy dataframe.
    """
    rows = []
    for r in results:
        time = (
            r.get("start_time") or
            r.get("D_time") or
            r.get("S_time") or
            r.get("T_time") or
            r.get("F_time")
        )
        rows.append({
            "subject": sub_id,
            "phase":   phase,
            "type":    r["type"],
            "time":    round(float(time), 3) if time is not None else None,
        })
    return pd.DataFrame(rows)


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

    for sub_id, cfg in subjects.items():
        print(f"\n{'='*60}")
        print(f"[START] Summarizing tics for {sub_id}")
        print(f"{'='*60}")

        # --- Output folder ---
        out_dir = PREPROC_DIR / sub_id / "tics"
        out_dir.mkdir(parents=True, exist_ok=True)

        # --- 1. Load merged events tsv ---
        tsv_path = out_dir / f"{sub_id}_ses-01_task-tictrack_merged_events.tsv"
        print(f"[1/4] Loading merged events: {tsv_path}")
        df = pd.read_csv(tsv_path, sep="\t")

        # --- 2. Convert to merged_ttl_tics format ---
        print(f"[2/4] Converting to merged_ttl_tics format...")
        merged_ttl_tics = build_merged_ttl_tics_from_tsv(df)
        print("merged_ttl_tics", merged_ttl_tics)

        # --- 3. Analyse each phase ---
        all_results = []

        # Spontaneous phase
        print(f"[3/4] Analysing phases...")
        print(f"  → Spontaneous tics...")
        spont_results = analyse_merged_ttl_tics_spontaneous(
            merged_ttl_tics,
            phase_start_key="start_PHASE_FREE",
            phase_end_key="end_PHASE_FREE",
        )
        all_results.append(results_to_df(spont_results, sub_id, "spontaneous"))
        print("all_results:", all_results)

        # Imitated phase
        print(f"  → Imitated tics...")
        imitated_tics, real_tics_imitated = analyse_merged_ttl_tics_imitated(
            merged_ttl_tics,
            phase_start_key="start_PHASE_MIM",
            phase_end_key="end_PHASE_MIM",
        )
        all_results.append(results_to_df(imitated_tics, sub_id, "imitated"))
        all_results.append(results_to_df(real_tics_imitated, sub_id, "imitated_real"))

        # Suppressed phase
        print(f"  → Suppressed tics...")
        suppressed_tics, real_tics_suppressed = analyse_merged_ttl_tics_suppressed(
            merged_ttl_tics,
            phase_start_key="start_PHASE_SUP",
            phase_end_key="end_PHASE_SUP",
        )
        all_results.append(results_to_df(suppressed_tics, sub_id, "suppressed"))
        all_results.append(results_to_df(real_tics_suppressed, sub_id, "suppressed_real"))

        # --- 4. Save summary ---
        print(f"[4/4] Saving summary...")
        df_summary = pd.concat(all_results, ignore_index=True)
        out_path = out_dir / f"{sub_id}_ses-01_task-tictrack_tics_summary.tsv"
        df_summary.to_csv(out_path, sep="\t", index=False)
        print(f"[OK] Saved: {out_path}")
        print(f"[OK] Total tics: {len(df_summary)}")
        print(f"\nSummary by phase and type:")
        print(df_summary.groupby(["phase", "type"]).size().to_string())

    print("\n[DONE] Tic summary complete.")


if __name__ == "__main__":
    main()