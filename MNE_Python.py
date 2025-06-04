# import numpy as np
# import mne
# # Loading Data
# sample_data_folder = mne.datasets.sample.data_path()
# sample_data_raw_file = (
#     sample_data_folder / "MEG" / "sample" / "sample_audvis_filt-0-40_raw.fif"
#     #chemin du fichier ?
# )
# raw = mne.io.read_raw_fif(sample_data_raw_file)
# print(raw)
# print(raw.info)
# raw.compute_psd(fmax=50).plot(picks="data", exclude="bads", amplitude=False)
# raw.plot(duration=5, n_channels=30)

# Comment test to Git

# Libraries
import os
import mne
import pandas as pd
import matplotlib.pyplot as plt


# 1. Define the path to the .vhdr file
FolderPath = "C:\\Users\\indira.lavocat\\MOVIDOC\\EEG\\Sujets\\IndiraLAVOCAT" # need to adapt the last folder to suit the subject
# Looking for the .vhdr file in the folder
for file in os.listdir(FolderPath):
    if file.endswith(".vhdr"):
        FilePath = os.path.join(FolderPath, file)
        break
print(FilePath)


# 2. Load the data
raw = mne.io.read_raw_brainvision(FilePath, preload=True)
raw.info
print(raw.ch_names)
print(raw.info['description']) # gives a note about the channels when there is one

# Create an event dicionnary
events, event_id = mne.events_from_annotations(raw)
print("Events list (stimulus) :")
print(event_id)

# Display the tab of events
print("Events (sample, previous_id, event_id) :")
print(events)

# Convert the timestamps into seconds
events_times_sec = events[:, 0] / raw.info['sfreq'] # converts the timestamps into seconds
for time, eid in zip(events_times_sec, events[:, 2]): # links each time in seconds to its event ID
    print(f"Stimulus {eid} à {time:.3f} s") # formats the number with 3 decimal

# Alternative to display the stimulus name (and not its ID) with its timestamp in seconds
id_to_name = {v: k for k, v in event_id.items()}
for time, eid in zip(events_times_sec, events[:, 2]):
    name = id_to_name.get(eid, f"ID {eid}")
    print(f"{name} à {time:.3f} s")


##########


# 3. Define the montage ???
raw.set_montage("easycap-M1", on_missing = "ignore")
fig1 = raw.plot_sensors(show_names=True)
#raw.set_montage("standard_1020") # to prevent error during the topography step

# 4. Plot the data
raw.plot(duration=5, n_channels=30)
raw.compute_psd(fmax=50).plot(picks="data", exclude="bads", amplitude=False)

# 5. Extract the events (from the .vmrk annotations)
events, event_id = mne.events_from_annotations(raw)
print("Événements détectés :", event_id)

# 4. Filter the data
raw.filter(l_freq=1., h_freq=30.) # for ERPs, [1-30] Hz band-pass filter

# 6. Seperation in epochs
tmin = -0.2  # 200 ms before the event
tmax = 0.8   # 800 ms after the event
epochs = mne.Epochs(raw, events, event_id=event_id,
                    tmin=tmin, tmax=tmax, baseline=(None, 0),
                    preload=True)
epochs.plot_drop_log()

# 7. Define an automatic reject of the artifacts (optional)
# epochs.plot_drop_log()
# epochs.drop_bad()

# 8. Averaging (ERP)
evoked = epochs.average()
evoked.plot() # evoked does not accept any "title"
fig = evoked.plot_image(picks='eeg')
fig.suptitle("ERP (moyenne des epochs)")

# 9. Topography
evoked.plot_topomap(times=[0.1, 0.2, 0.3], ch_type='eeg')

plt.show()



# 10.1.a Save the analysis in .fif (average epochs)
# evoked.save("MOVIDOC_TicTrack_000002_erp-ave.fif") # saves the average epochs
# => To re-load it :
# from mne import read_evokeds
# evoked = read_evokeds("MOVIDOC_TicTrack_000002_erp-ave.fif", condition=0)

# 10.1.b Save the analysis in .fif (complete epochs)
# epochs.save("MOVIDOC_TicTrack_000002_epochs.fif", overwrite=True) # saves the complete epochs
# To re-load it :
# from mne import read_epochs
# epochs = read_epochs("MOVIDOC_TicTrack_000002_epochs.fif")

# 10.2 Save the analysis in .csv
# import pandas as pd
# => Convert the ERP (evoked) into a DataFrame
# df = pd.DataFrame(evoked.data.T, columns=evoked.ch_names)
# df.insert(0, "time", evoked.times)  # Ajoute la colonne temps en 1ère position
# => Save as a .csv
# df.to_csv("MOVIDOC_TicTrack_000002_erp.csv", index=False)