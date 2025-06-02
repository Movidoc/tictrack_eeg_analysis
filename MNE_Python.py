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


# Libraries
import mne
import matplotlib.pyplot as plt

# 1. Define the path to the .vhdr file
vhdr_file = "C:\\Users\\indira.lavocat\\MOVIDOC\\EEG\\Indira Test\\MOVIDOCTicTrack000005.vhdr"

# 2. Load the data
raw = mne.io.read_raw_brainvision(vhdr_file, preload=True)
print(raw)
print(raw.info)

# 3. Quick plot
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