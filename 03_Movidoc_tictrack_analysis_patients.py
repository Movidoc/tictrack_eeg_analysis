# ===================================================================
# File : 03_Movidoc_tictrack_analysis_patients.py
# Purpose : Extract tics start & end times from the Excel annotations
# Author  : Indira
# ===================================================================


import pandas as pd
import numpy as np


def extract_tics_from_excel(excel_file, phase_start_s, phase_end_s, min_absence_frames=30):

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
    phase_start_s : float
        Start time of the phase to analyze (seconds).
    phase_end_s : float
        End time of the phase to analyze (seconds).
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
    # print(df.columns)


    # check the required columns
    time_col = "Time (30 fps) s"
    if time_col not in df.columns:
        raise ValueError(f"Missing required column: {time_col}")
    if "Absence" not in df.columns:
        raise ValueError("Missing required column 'Absence'.")
    times = df[time_col].values
    absence = df["Absence"].values

    # define explicitly the columns corresponding to the tics bodyparts
    movement_cols = ["Epaules", "Main - Bras", "Tête", "Yeux", "Visage", "Bassin - Tronc", "Jambes - Pieds", "Phonique", "Vocaux"]
    # convert into a numpy array
    movement_data = df[movement_cols].values
    # count the number of active bodyparts columns per frame
    movement_sum = movement_data.sum(axis=1)

    # tic frames : Absence = 0 & movement activity = 1+
    # is_tic_frame = (absence == 0) & (movement_sum > 0)

    tics = []
    n = len(df)
    i = 0


    # main detection loop
    # iterate through all frames until the end
    while i < n:

        # -------------------- TIC CONDITION 1 : 30 frames of "Absence" before --------------------
        if i >= min_absence_frames:
            before_ok = np.all((absence[i-min_absence_frames:i] == 1) & (movement_sum[i-min_absence_frames:i] == 0))
        else:
            before_ok = False

            i += 1 # otherwise, move to the next frame
            continue # and skip the rest of this iteration

        # -------------------- TIC CONDITION 2 : frame with "Absence"=0 and BodyParts=1 --------------------
        if before_ok and absence[i] == 0 and movement_sum[i] > 0:
            start_idx = i
            start_time = times[start_idx]

            # search the end while in the condition 2
            j = i
            while j < n and (absence[j] == 0 and movement_sum[j] > 0):
                j += 1 # move forward frame by frame

            end_idx = j - 1
            end_time_candidate = times[end_idx]

            # -------------------- TIC CONDITION 3 : 30 frames of "Absence" after --------------------
            # if j + min_absence_frames <= n:
            #     after_ok = np.all((absence[j:j+min_absence_frames] == 1) & (movement_sum[j:j+min_absence_frames] == 0))
            # else:
            #    after_ok = False

            while j < n and not ((absence[j] == 1) and (movement_sum[j] == 0)):
                j += 1
            
            # measure how many consecutive frames are pure absence
            after_len = 0
            k = j
            while k < n and (absence[k] == 1) and (movement_sum[k] == 0):
                k += 1
            after_len = k - j

            # verify if the lenght is enough
            after_ok = after_len >= min_absence_frames

            
            if after_ok:
                # add & save the tic interval as a (start, end) pair
                tics.append((start_time, end_time_candidate))
                # i = end_idx + min_absence_frames
                i = k
                continue # and restart detection from there

        # if no tic in this frame, just move to next frame
        i += 1
    
    # filter tics inside phase
    tics_in_phase = []
    for start, end in tics:
        # keep the tic if it intersects the phase
        if end >= phase_start_s and start <= phase_end_s:
            tics_in_phase.append((start, end))

    # return all detected tic intervals
    return tics_in_phase



# small test if run directly (recommended!)
if __name__ == "__main__":
    # Filename of your real Excel annotations file
    test_file = "SC31_annotations_binary-table_cutted.xlsx"

    phase_start = 345.972
    phase_end = 970.167
    
    try:
        tics = extract_tics_from_excel(excel_file=test_file, phase_start_s=phase_start, phase_end_s=phase_end, min_absence_frames=30)
        print("\nTics detected inside selected phase:")
        if not tics:
            print(" No tics detected in this phase.")
        else:
            for start, end in tics:
                print(f" - Tic from {start:.3f}s to {end:.3f}s")
    except Exception as e:
        print(f"Error: {e}")








#########################################################################
# import pandas as pd
# import numpy as np


# def extract_tics_from_excel(excel_file, phase_start_s, phase_end_s, min_absence_frames=30):

#     """
#     ----------
#     Purpose
#     ----------
#     Extract tic intervals from a specified time window (phase).

#     ----------
#     Parameters
#     ----------
#     excel_file : str
#         Path to the Excel annotation file.
#     phase_start_s : float
#         Start time of the phase to analyze (seconds).
#     phase_end_s : float
#         End time of the phase to analyze (seconds).
#     min_absence_frames : int
#         Minimum number of consecutive "Absence = 1" frames before AND after a tic.

#     ----------
#     Returns
#     ----------
#     tics : list of tuples
#         [(start_time_s, end_time_s), ...] only inside the selected phase.
#     """

#     # load the Excel
#     df = pd.read_excel(excel_file)

#     # check the required columns
#     time_col = "Time (30 fps) s"
#     if time_col not in df.columns:
#         raise ValueError(f"Missing required column: {time_col}")
#     if "Absence" not in df.columns:
#         raise ValueError("Missing required column 'Absence'.")
    
#     # limit the analysis to the selected phase only
#     # df = df[(df[time_col] >= phase_start_s) & (df[time_col] <= phase_end_s)].reset_index(drop=True)

#     times = df[time_col].values
#     absence = df["Absence"].values

#     # define explicitly the columns corresponding to the tics bodyparts
#     movement_cols = ["Epaules", "Main - Bras", "Tête", "Yeux", "Visage", "Bassin - Tronc", "Jambes - Pieds", "Phonique", "Vocaux"]
#     # convert into a numpy array
#     movement_data = df[movement_cols].values
#     # count the number of active bodyparts columns per frame
#     movement_sum = movement_data.sum(axis=1)

#     # tic frames : Absence = 0 & movement activity = 1+
#     is_tic_frame = (absence == 0) & (movement_sum > 0)

#     # tics = []
#     tics_full = []
#     n = len(df)
#     i = 0


#     # main detection loop
#     # iterate through all frames until the end
#     while i < n:

#         # -------------------------- TIC START --------------------------
#         # check if the current frame is part of a tic
#         if is_tic_frame[i]:

#             # if there are enough frames before 'i' & all of them correspond to an absence (= 1),
#             if i >= min_absence_frames and np.all(absence[i-min_absence_frames:i] == 1):
#                 start_time = times[i] # then validate this as the beginning of a tic & store the start time of the tic
#             else:
#                 i += 1 # otherwise, move to the next frame
#                 continue # and skip the rest of this iteration

#             # -------------------------- TIC END --------------------------
#             j = i # initialize j to search for the end
#             while j < n and is_tic_frame[j]: # increase j while still in tic frames
#                 j += 1 # move forward frame by frame

#             # after the tic ends (at frame j), check if there are enough absence frames afterwards
#             if j + min_absence_frames < n and np.all(absence[j:j+min_absence_frames] == 1):
#                 end_time = times[j-1] # store the end time (last tic frame)
#             else:
#                 i = j # if no valid absence block after, skip to j
#                 continue # and restart detection from there

#             # save the tic interval as a (start, end) pair
#             # tics.append((start_time, end_time))
#             tics_full.append((start_time, end_time))

#             # continue scanning after the end of the detected tic
#             i = j

#         else:
#             i += 1 # if no tic in this frame, just move to next frame
    
#     # filter tics inside phase
#     tics_in_phase = []
#     for start, end in tics_full:
#         # Keep tic if it intersects the phase
#         if end >= phase_start_s and start <= phase_end_s:
#             tics_in_phase.append((start, end))

#     # return all detected tic intervals
#     return tics_in_phase



# # small test if run directly (recommended!)
# if __name__ == "__main__":
#     # Filename of your real Excel annotations file
#     test_file = "SC31_annotations_binary-table_cutted.xlsx"

#     phase_start = 345.972
#     phase_end = 970.167
    
#     try:
#         tics = extract_tics_from_excel(excel_file=test_file, phase_start_s=phase_start, phase_end_s=phase_end, min_absence_frames=30)
#         print("\nTics detected inside selected phase:")
#         for start, end in tics:
#             print(f" - Tic from {start:.3f}s to {end:.3f}s")
#     except Exception as e:
#         print(f"Error: {e}")