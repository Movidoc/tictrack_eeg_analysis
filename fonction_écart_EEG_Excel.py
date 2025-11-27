# ===================================================================================================
# Function : recalibrate_excel_to_eeg
# Purpose : readjust the timestamps of the extracted tics from the Excel based on the EEG referencial
# ===================================================================================================

def recalibrate_excel_to_eeg(tics_info, stim2_eeg_time):

    """
    ----------
    Purpose
    ----------
    Readjust the timestamps of the extracted tics from the Excel (t=0 at Stimulus/S  2) based on the EEG referencial

    ----------
    Parameters
    ----------
    tics_info : list of dict
        list of the tics with their timestamps (start, end) based on the Excel
    stim2_eeg_time : float
        Timestamp of the Stimulus/S 2 in the readjusted EEG signal (should be ~0.0 après recalage)
    
    ----------
    Returns
    -------
    tics_info : list of dict
        List of the tics with their readjusted timestamps based on the EEG referencial
    """
    
    # Stimulus/S 2 in the Excel corresponds to t=0 (as the video has been cutted at this moment)
    excel_stim2_time = 0.0
    
    # calculate the offset between the EEG & Excel files
    offset = stim2_eeg_time - excel_stim2_time
    
    print(f"\n===== RECALAGE EXCEL → EEG =====")
    print(f"Stimulus/S 2 dans EEG : {stim2_eeg_time:.3f}s")
    print(f"Stimulus/S 2 dans Excel : {excel_stim2_time:.3f}s")
    print(f"Offset appliqué aux tics : {offset:.3f}s")
    
    # Apply the offset to all the tics
    for tic in tics_info:
        tic["start"] += offset
        tic["end"] += offset
    
    print(f"✓ {len(tics_info)} tics recalés sur le référentiel EEG\n")
    
    return tics_info