import pandas as pd

def events_to_dataframe(events, sfreq):
    """
    Convert MNE events array into a labeled DataFrame.

    Parameters
    ----------
    events : ndarray
        MNE events array (n_events x 3)
    sfreq : float
        Sampling frequency

    Returns
    -------
    df : pandas.DataFrame
        Columns:
            - sample
            - time_sec
            - ttl
            - label
    How to use
    raw = mne.io.read_raw_brainvision(vhdr_path)
    events, _ = mne.events_from_annotations(raw)
    df_events = events_to_dataframe(events, raw.info["sfreq"])
    print(df_events.head())

    """
    df = pd.DataFrame({
        "sample": events[:, 0],
        "time_sec": events[:, 0] / sfreq,
        "ttl": events[:, 2],
    })

    df["label"] = df["ttl"].map(TTL_LABELS)



    return df