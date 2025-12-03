# ===================================================================
# File : _03_Movidoc_tictrack_tic_extraction_patients.py
# Purpose : Extract tics start & end times from the Excel annotations
# Author  : Indira
# ===================================================================



# ============================================================
# Libraries
# ============================================================

import pandas as pd
import numpy as np



# ============================================================
# Define the function
# ============================================================


# ==============================================================
# Function : extract_tics_from_excel
# Purpose : extract tic beginning & end from a single Excel file
# ==============================================================

def extract_tics_from_excel(excel_file, fps, min_absence_frames=30):

    """
    ----------
    Purpose
    ----------
    Extract tic intervals from a specified time window (phase).

    ----------
    Parameters
    ----------
    excel_file : str
        Path to the Excel annotation file.
    # phase_start_s : float
        # Start time of the phase to analyze (seconds).
    # phase_end_s : float
        # End time of the phase to analyze (seconds).
    fps : float
        Frequency of the chosen time column
    min_absence_frames : int
        Minimum number of consecutive "Absence = 1" frames before AND after a tic.

    ----------
    Returns
    ----------
    tics : list of tuples
        [(start_time_s, end_time_s), ...] only inside the selected phase.
    """

    # load the Excel
    df = pd.read_excel(excel_file)


    # check the required columns
    if fps == 30:
        time_col = "Time (30 fps) s"
    elif fps == 25:
        time_col = "Time (25 fps) ms"
    else:
        raise ValueError("fps must be 30 or 25")

    if time_col not in df.columns:
        raise ValueError(f"Missing required column: {time_col}")
    times = df[time_col].values
    if fps == 25:
        times = times / 1000.0

    if "Absence" not in df.columns:
        raise ValueError("Missing required column 'Absence'.")
    absence = df["Absence"].values

    # define explicitly the columns corresponding to the tics bodyparts
    movement_cols = ["Epaules", "Main - Bras", "Tête", "Yeux", "Visage", "Bassin - Tronc", "Jambes - Pieds", "Phonique", "Vocaux"]
    # convert into a numpy array
    movement_data = df[movement_cols].values
    # count the number of active bodyparts columns per frame
    movement_sum = movement_data.sum(axis=1)

    invalid_mask = (absence == 1) & (movement_sum > 0)
    if np.any(invalid_mask):
        bad_indices = np.where(invalid_mask)[0]
        raise ValueError(f"Annotation error: Absence=1 but movement>0 at rows: {bad_indices.tolist()}")

    tics = []
    n = len(df)
    i = 0


    # -------------------- main detection loop --------------------
    print(f"Checkpoint : Total frames to process: {n}")


    # iterate through all frames until the end
    while i < n:
        # print(f"Checkpoint : Processing frame {i}/{n}")
        
        # to check the indices that seems to be a problem
        # if i > 10140:
        #     print(i)
        

        # -------------------- TIC CONDITION 1 : 30 frames of "Absence" before --------------------
        before_ok = False
        before_ok = np.all((absence[i-min_absence_frames:i] == 1) & (movement_sum[i-min_absence_frames:i] == 0)) & (i >= min_absence_frames)
        # print(f"Checkpoint : i={i} before_ok={before_ok} absence[i]={absence[i]} movement_sum[i]={movement_sum[i]}")


        if before_ok and absence[i] == 0 and movement_sum[i] > 0:
            
            start_idx = i
            start_time = times[start_idx]

            # search the end while in the condition 2
            j = i
            after_len = 0

            while after_len < min_absence_frames:
                
                while j < n and (absence[j] == 0 and movement_sum[j] > 0):
                    j += 1 # move forward frame by frame

                end_idx = j - 1
                end_time_candidate = times[end_idx]


                # -------------------- TIC CONDITION 3 : 30 frames of "Absence" after --------------------
                
                # measure how many consecutive frames are pure absence
                k = j
                while k < n and (absence[k] == 1) and (movement_sum[k] == 0):
                    k += 1
                after_len = k - j
                # print(f"  j={j}, k={k}, after_len={after_len}")

                # to check the indices that seems to be a problem
                # if i > 10140:
                #     print(f"J value : {j}\nK value : {k}")
                
                # verify if the lenght is enough
                after_ok = after_len >= min_absence_frames

                if after_ok:
                    # add & save the tic interval as a (start, end) pair
                    tics.append((start_time, end_time_candidate))
                    # print(f"Candidate selected line : {i}")
                    i = k
                    break # and restart detection from there
                else :
                    j = k
        else :
            i += 1
            # print(f"Skipping frame {i}")
    
    # # filter tics inside phase
    # tics_in_phase = []
    # for start, end in tics:
    #     # keep the tic if it intersects the phase
    #     if end >= phase_start_s and start <= phase_end_s:
    #         tics_in_phase.append((start, end))

    # # return all detected tic intervals
    # return tics_in_phase

    return tics

#########################################################################



# === code with functions ===

# if __name__ == "__main__":
    
#     # Test on 1 Excel file only

#     # Definition of the phase limits ALWAYS in seconds (even if in milliseconds in the Excel file)
#     phase_start_DS26 = 349.270
#     phase_end_DS26 = 980.730
#     # phase_start_BB28 = 302600 / 1000.0
#     # phase_end_BB28 = 929960 / 1000.0
#     # phase_start_BC29 = 358.215
#     # phase_end_BC29 = 1088.67
#     # phase_start_MM30 = 323.994
#     # phase_end_MM30 = 995.049
#     # phase_start_SC31 = 345.972
#     # phase_end_SC31 = 970.167

#     # Load the data
#     xlsx_file = "C:\\Users\\indira.lavocat\\MOVIDOC\\PATIENT FILES\\EXCEL PATIENT FILES\\DS26_annotations_binary-table_cutted.xlsx"
#     fps = 30 # 30, or 25 (for BB28 only)
#     min_absence_frames = 30 # 30 ou 25 (for BB28 only)

#     try:
#         # tics = extract_tics_from_excel(excel_file=xlsx_file, phase_start_s=phase_start_MM30, phase_end_s=phase_end_MM30, fps=fps, min_absence_frames=min_absence_frames)
#         tics = extract_tics_from_excel(excel_file=xlsx_file, fps=fps, min_absence_frames=min_absence_frames)
#         print("\nTics detected inside selected phase:")
#         if not tics:
#             print(" No tics detected in this phase.")
#         else:
#             for start, end in tics:
#                 print(f" - Tic from {start:.3f}s to {end:.3f}s")
#     except Exception as e:
#         print(f"Error: {e}")

#########################################################################