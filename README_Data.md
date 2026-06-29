README Data

For the data the main file is tictrack_eeg_analysis_Lizbeth/dataset

1. Each folder with names of the subject (eg. sub-BB28) contains two folders with Excel and EEG data 
- ses-01/excel folder:
    a. sub-BB28_task-tictrack.xlsx - file with manual annotations of the videos 
- ses-01/eeg folder:
    a. sub-BB28_task-tictrack.eeg - raw signal 
    b. sub-BB28_task-tictrack.vmrk - raw signal 
    c. sub-BB28_task-tictrack.vhdr - information about the number of channels, reference, impedance, and other EEG parameters, [IMPORTANT] to keep the name formating consistent between patients i renamed .eeg and .vmrk as above, however, then in the .vhdr file those names have to be update 
    " Brain Vision Data Exchange Header File Version 1.0
        Data created by the Vision Recorder

        [Common Infos]
        Codepage=UTF-8
        DataFile= sub-BB28_task-tictrack.eeg
        MarkerFile= sub-BB28_task-tictrack.vmrk
        "

2. Data obtained from the preprocessing and data analysis are in the derivatives/preproc/sub-BB28
- sub-BB28/raw:
    a. sub-BB28_ses-01_task-tictrack_run-01_events_qc - file with the events (TTLs) obtained from the raw signal 
    b. sub-BB28_ses-01_task-tictrack_run-01_events - file with the events (TTLs) and their timestep - used to see where to cut the EEG data for realignement 
- sub-BB28/realign:
    a. sub-BB28_ses-01_task-tictrack_aligned_raw.fif - realigned EEG signal 
    b. sub-BB28_ses-01_task-tictrack_aligned_annotated_raw.fif - realigned EEG signal with realigned annotations (correct file)
    c. sub-BB28_ses-01_task-tictrack_run-01_events_recalibrated_with_phases - file with TTLs events with timestep from realigned data and LED timestep from the videos - used to ensure the realignment was correct
    d. plots folder - contains images of the raligned data (30s) with annotations for each phase
- sub-BB28/tics:
    a. sub-BB28_ses-01_task-tictrack_merged_events - file with the merged EEG events (TTLs) and EXCEL events (tics), timestep, and phase
    b. sub-BB28_ses-01_task-tictrack_start_diff - file with the delay between the reported urge and start of the follwing tic 
- sub-BB28/tics_manual:
    a. sub-BB28_ses-01_task-tictrack_tic_epoch_manual - manually created file with the start and end of each events, the start either means the reported premonitory urge or the start of the tic, and the end is the end of the tic, this file is used to create epochs for the tic/urge events
    b. tic folder with raw data of each tic event, file with the summary of each start time (sub-BB28_ses-01_task-tictrack_run-01_tic_epochs) and PREPROCESSED epochs .fif (sub-BB28_ses-01_task-tictrack_tic_epo.fif)
    c. no_tic folder - same as b.
- sub-BB28/preprocessing:
    a. sub-BB28_butterfly_before_RANSAC.png - butterfly plot of the raw data before RANSAC, used just for the control
    b. sub-BB28_butterfly_after_RANSAC.png - butterfly plot of the raw data after RANSAC, used to detect bad channels 
    c. sub-BB28_FASTER_drop_log.png - shows how many epochs were rejected by FASTER algorithm
    d. sub-BB28_ica_components.png - shows the ICA components, used to detect bad components
    e. sub-BB28_ica_property_x.png - shows the properties of the ICA components, used to detect bad components for the exclusion 
    f. sub-BB28_ica_sources.png - shows the sources of the ICA components, used to detect bad components for the exclusion
    g. sub-BB28_raw_rest_reference & sub-BB28_raw - raw and preprocessed data just for the visual inspection to see the effect of the preprocessing steps
    h. sub-BB28_ses-01_task-tictrack_preprocessed_raw.fif - preprocessed raw data after all the preprocessing steps
    i. plots folder - contains images of the preprocessed data (30s) with annotations for each phase
- sub-BB28/epochs:
    a. contains folders for each phase with the epochs of each phase, after preprocessing - used for visual inspection
- sub-BB28/tfr:
    a. contains folders for each phase with the time-frequency representations of each phase and each tic type (eg. expressed or imitated)
    b. no_tic folder contains the time-frequency representations of the no tic epochs (baseline) - used for controlling if the baseline doesnt contain any signal 
    c. single_epoch folder contains the time-frequency representations of single epochs, used for visual inspection of the tfrs
-sub-BB28/stats:
    a. cluster_1sample - folder that includes folders for each phase with the results of the cluster-based permutation test for each phase, used to see if there are significant differences between tic and baseline, includes both the csv of the significant clusters data and the TFR images of the significant clusters
    b. between_cluster - includes TFR images of the significant clusters obtained from the between cluster permutation test comparing different tic types across phases
-sub-BB28/spectra:
    a. no_tic_EO_EC_by_roi - contains power spectra of the no tic epochs (baseline) for eyes open and eyes closed conditions, separated by regions of interest (ROIs), used to observe alpha-band modulation (increase in alpha power during eyes closed condition) 
    b. file with quantitive data of the power spectra 

3. Demographic and summary data across patients are in the derivatives/summary_data
    a. Patient_summary_data - contains information about the summary of the preprocessing data (nb of electodes, channels exclusion, ICA component exclusion, nb of epochs for ICA and analysis)
    b. D_start_diff - contains the delay between the reported urge and start of the follwing tic for all patients, used to see if there are differences in this delay across patients and tic types, quantative data 
    c. delay_histogram - histogram of the delay between the reported urge and start of the follwing tic for all patients
    d. Demographic_data - summary of the demographic data 
 