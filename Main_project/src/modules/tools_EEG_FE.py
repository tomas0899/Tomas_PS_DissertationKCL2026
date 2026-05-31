import os
from pathlib import Path
from datetime import datetime, timedelta
from scipy.stats import skew, kurtosis
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.io as sio
from scipy.signal import welch
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import welch
import glob
from scipy.signal import iirnotch, tf2sos
from scipy.signal import butter, sosfiltfilt, iirnotch, tf2sos
from pathlib import Path

# FEATURE EXTRACTION
# Function #1
def parse_timestamp_2_1(val):
    """Accept Unix float, datetime string, or repeated timestamp arrays."""

    if isinstance(val, np.ndarray):
        flat = val.ravel()
        cleaned = [str(x).strip() for x in flat if str(x).strip() != ""]

        if len(cleaned) == 0:
            return None

        unique_vals = list(dict.fromkeys(cleaned))

        if len(unique_vals) == 1:
            val = unique_vals[0]
        else:
            raise ValueError(f"Timestamp array has multiple different values: {unique_vals}")

    try:
        return float(val)
    except (ValueError, TypeError):
        pass

    val = str(val).strip()
    dt = datetime.strptime(val, "%Y-%m-%d %H:%M:%S.%f")
    return dt.replace(tzinfo=timezone.utc).timestamp()
#=================================================================================
#=================================================================================
#=================================================================================
# 
# Function #2

def clean_onsets_2_2(x):
    if isinstance(x, (list, np.ndarray)):
        return [i for i in x if not pd.isna(i)]
    elif pd.isna(x):
        return []
    else:
        return [x]
#=================================================================================
#=================================================================================
#=================================================================================
# 
# Function #3
def create_eeg_windows_2_3(df, window_sec=10):
    """
    Create window-level metadata from EEG recordings stored in .npz files.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with at least:
        - file_path
        - file_name
        - fs (sampling frequency)
        - seizure_onsets_clean

    window_sec : int or float, optional (default=10)
        Window size in seconds.

    Returns
    -------
    df_windows : pandas.DataFrame
        One row per window with metadata.
    """

    rows_windows = []

    for idx, row in df.iterrows():
        # Load data
        file_path = row["file_path"]
        data = np.load(file_path, allow_pickle=True)

        X = data["X"]  # shape: (channels, samples)
        fs = row["fs"]

        # Compute window size in samples
        window_size = int(window_sec * fs)

        # Total samples
        N = X.shape[1]

        # Number of full windows
        n_windows = N // window_size  # ignore incomplete last window

        seizure_onsets = row["seizure_onsets_clean"]

        for w in range(n_windows):
            start = w * window_size
            end = start + window_size

            rows_windows.append({
                "file_name": row["file_name"],
                "window_id": w,
                "start_sample": start,
                "end_sample": end,
                "fs": fs,
                "n_channels": X.shape[0],
                "window_sec": window_sec,
                "seizure_onsets": seizure_onsets
            })

    df_windows = pd.DataFrame(rows_windows)

    return df_windows
#=================================================================================
#=================================================================================
#=================================================================================
# 
# Function #4

def get_windows_with_seizures_2_4(df_windows, filter_seizures=True):
    """
    Optionally filter windows that belong to recordings with seizures.

    Parameters
    ----------
    df_windows : pandas.DataFrame
        DataFrame containing all EEG windows

    filter_seizures : bool (default=True)
        - True  → return only windows from recordings that have seizures
        - False → return full dataset without filtering

    Returns
    -------
    pandas.DataFrame
        Filtered or unfiltered DataFrame (copy, original is not modified)
    """

    # If filtering is disabled, return a copy of the original DataFrame
    if not filter_seizures:
        return df_windows.copy()

    # Create a boolean mask to identify rows with valid seizure information
    mask = df_windows["seizure_onsets"].apply(
        lambda x: (
            not pd.isna(x).all()  # case: list/array → check if at least one valid value exists
            if isinstance(x, (list, np.ndarray))
            else not pd.isna(x)   # case: single value → check if not NaN
        )
    )

    # Apply the mask to filter the DataFrame
    df_seizures = df_windows[mask].copy()

    return df_seizures
#=================================================================================
#=================================================================================
#=================================================================================
# 
# Function #5



# Labeling:
# step by step
import pandas as pd
import numpy as np

# 1. Helper functions
LABEL_MAP = {
    "interictal": 0,
    "preictal": 1,
    "seizure": 2
}
def clean_onsets(x):
    if isinstance(x, (list, np.ndarray)):
        return [i for i in x if not pd.isna(i)]
    elif pd.isna(x):
        return []
    else:
        return [x]


def overlaps(a_start, a_end, b_start, b_end):
    """
    Return True if two time intervals overlap.
    """
    return (a_start < b_end) and (a_end > b_start)
#window:   |----------|
#seizure:       |----------|
# returns True in that case, because it overlaps

def is_in_gap(window_start, window_end, preictal_end, seizure_start):
    """
    Return True if the window overlaps the gap between
    the preictal interval and the seizure interval.
    detects if a window is in between the space of preictal ending and seizure start
    """
    return overlaps(window_start, window_end, preictal_end, seizure_start)
#onset -10 min    onset -5 min        onset        onset +5 min
#|--- preictal ---|---- gap ----|--- seizure ---|

def get_seizure_intervals(onset, preictal_range_min, ictal_range_min):
    """
    Given one seizure onset, create the preictal and seizure intervals.
    E.g.:
    onset = "2026-04-27 12:00:00"
    preictal_range_min = (-10, -5)
    ictal_range_min = (0, 5)
    returns:
    preictal: 11:50 to 11:55
    seizure:  12:00 to 12:05
    """
    onset = pd.to_datetime(onset)

    preictal_start = onset + pd.Timedelta(minutes=preictal_range_min[0])
    preictal_end = onset + pd.Timedelta(minutes=preictal_range_min[1])

    seizure_start = onset + pd.Timedelta(minutes=ictal_range_min[0])
    seizure_end = onset + pd.Timedelta(minutes=ictal_range_min[1])

    return preictal_start, preictal_end, seizure_start, seizure_end
# 2. Prepare dataframe
def initialize_labeled_dataframe_2_5_1(df_windows):
    """
    Create a copy of df_windows and add empty columns
    needed for labeling.
    df_windows original
        ↓
    initialize_labeled_dataframe()
        ↓
    df_labeled with empty columns
    """
    # 2.1 creates a copy to preserve original dataframe
    df_labeled = df_windows.copy()
    # 2.2 add new columns
    df_labeled["window_start_time"] = pd.NaT
    df_labeled["window_end_time"] = pd.NaT
    df_labeled["class_label"] = np.nan
    df_labeled["label_name"] = pd.NA
    df_labeled["excluded_reason"] = pd.NA

    return df_labeled
# 3. Calculate real time per window:
def compute_window_times(row, recording_start):
    """
    Compute the real datetime start and end of one EEG window.

    Parameters
    ----------
    row : pd.Series
        One row from df_windows.

    recording_start : datetime-like
        Start time of the recording.

    Returns
    -------
    window_start_time : pd.Timestamp
    window_end_time : pd.Timestamp
    -----
    start_sample / fs = seconds from beginning of recording
    end_sample / fs   = seconds from end of recording

    E.g:
    start_sample = 2070
    end_sample = 4140
    fs = 207
    
    start_sec = 10 segundos
    end_sec   = 20 segundos
    if recording started:
    2026-04-27 12:00:00
    then:
    window_start_time = 2026-04-27 12:00:10
    window_end_time   = 2026-04-27 12:00:20
    """

    recording_start = pd.to_datetime(recording_start)

    fs = row["fs"]

    start_sec = row["start_sample"] / fs
    end_sec = row["end_sample"] / fs

    window_start_time = recording_start + pd.Timedelta(seconds=start_sec)
    window_end_time = recording_start + pd.Timedelta(seconds=end_sec)

    return window_start_time, window_end_time
# 4. Search for corresponding recording
def get_matching_recording(row, df_recordings):
    """
    Find the recording-level metadata corresponding to one window.
        row["file_name"]
            ↓
    search for file name in df_recording
            ↓
    returns corresponding row
    """

    file_name = row["file_name"]

    matching_rec = df_recordings[
        df_recordings["file_name"] == file_name
    ]

    if matching_rec.empty:
        raise ValueError(
            f"No matching recording found in df_recordings for file_name: {file_name}"
        )

    rec = matching_rec.iloc[0]

    return rec
# 5. get real time per window
def get_window_datetime_info(row, df_recordings):
    """
    Get the real start and end datetime of one EEG window.
        row de df_windows
            ↓
    search for metadata in df_recordings
            ↓
    extract start_time from recording
            ↓
    convert start_sample/end_sample into real datetime
    """

    rec = get_matching_recording(row, df_recordings)

    window_start_time, window_end_time = compute_window_times(
        row=row,
        recording_start=rec["start_time"]
    )

    return window_start_time, window_end_time
# 6. Labels one window at a time
def label_single_window(
    window_start_time,
    window_end_time,
    seizure_onsets,
    preictal_range_min=(-10, -5),
    ictal_range_min=(0, 5),
    include_gap_as_interictal=True
):
    """
    Label one EEG window as interictal, preictal, seizure,
    or mark it for exclusion if it falls in the periictal gap.
    
    One window + seizure_onsets
            ↓
    overlap with seizure?
            ↓
    yes → seizure
    
    elif:
            ↓
    overlap with preictal?
            ↓
    yes → preictal
    
    else:
            ↓
    ¿overlap with gap?
            ↓
    yes and include_gap_as_interictal=False → exclude
    
    if nothing:
            ↓
    interictal
    """

    # Default label
    assigned_label = LABEL_MAP["interictal"]
    assigned_name = "interictal"
    excluded_reason = pd.NA

    # Clean seizure onsets
    seizure_onsets = clean_onsets(seizure_onsets)

    # If no seizure onsets exist, keep interictal
    if len(seizure_onsets) == 0:
        return assigned_label, assigned_name, excluded_reason

    overlaps_gap = False

    for onset in seizure_onsets:

        preictal_start, preictal_end, seizure_start, seizure_end = (
            get_seizure_intervals(
                onset=onset,
                preictal_range_min=preictal_range_min,
                ictal_range_min=ictal_range_min
            )
        )

        # 1. Seizure has highest priority
        if overlaps(
            window_start_time,
            window_end_time,
            seizure_start,
            seizure_end
        ):
            assigned_label = LABEL_MAP["seizure"]
            assigned_name = "seizure"
            excluded_reason = pd.NA
            break

        # 2. Preictal has second priority
        elif overlaps(
            window_start_time,
            window_end_time,
            preictal_start,
            preictal_end
        ):
            assigned_label = LABEL_MAP["preictal"]
            assigned_name = "preictal"
            excluded_reason = pd.NA

        # 3. Optional gap detection
        elif preictal_end < seizure_start:
            if is_in_gap(
                window_start_time,
                window_end_time,
                preictal_end,
                seizure_start
            ):
                overlaps_gap = True

    # Exclude gap windows if requested
    if overlaps_gap and not include_gap_as_interictal:
        assigned_label = np.nan
        assigned_name = pd.NA
        excluded_reason = "periictal_gap"

    return assigned_label, assigned_name, excluded_reason
# 7. loop through all windows
def apply_window_labeling_2_5_2(
    df_labeled,
    df_recordings,
    preictal_range_min=(-10, -5),
    ictal_range_min=(0, 5),
    include_gap_as_interictal=True
):
    """
    Apply window labeling row by row to the full dataframe.
    """

    for idx, row in df_labeled.iterrows():

        # 1. Compute real datetime of the window
        window_start_time, window_end_time = get_window_datetime_info(
            row=row,
            df_recordings=df_recordings
        )

        # 2. Label one window
        assigned_label, assigned_name, excluded_reason = label_single_window(
            window_start_time=window_start_time,
            window_end_time=window_end_time,
            seizure_onsets=row["seizure_onsets"],
            preictal_range_min=preictal_range_min,
            ictal_range_min=ictal_range_min,
            include_gap_as_interictal=include_gap_as_interictal
        )

        # 3. Save results
        df_labeled.at[idx, "window_start_time"] = window_start_time
        df_labeled.at[idx, "window_end_time"] = window_end_time
        df_labeled.at[idx, "class_label"] = assigned_label
        df_labeled.at[idx, "label_name"] = assigned_name
        df_labeled.at[idx, "excluded_reason"] = excluded_reason

    return df_labeled
def filter_preictal_seizure_2_5_3(
    df_labeled,
    keep_only_preictal_seizure=True
):
    """
    Optionally keep only preictal and seizure windows.

    Parameters
    ----------
    df_labeled : pd.DataFrame
        DataFrame with labeled windows.

    keep_only_preictal_seizure : bool
        If True, keep only labels 1 (preictal) and 2 (seizure).
        If False, return the full dataframe unchanged.

    Returns
    -------
    df_final : pd.DataFrame
    """

    if not keep_only_preictal_seizure:
        # Still ensure correct dtype
        df_labeled["class_label"] = df_labeled["class_label"].astype("Int64")
        return df_labeled

    df_final = df_labeled[
        df_labeled["class_label"].isin([1, 2])
    ].copy()

    df_final["class_label"] = df_final["class_label"].astype(int)

    return df_final

#=================================================================================
#=================================================================================
#=================================================================================
# 
# Function #6


def extract_time_features(window, channel_names=None):
    """
    Extract per-channel time-domain features from one EEG window.

    Parameters
    ----------
    window : np.ndarray
        Shape (C, window_samples)
    channel_names : list or None
        Optional list of channel names

    Returns
    -------
    features : dict
        Flat dictionary with one feature per channel
    """
    n_channels = window.shape[0]
    features = {}

    for ch in range(n_channels):
        x = window[ch, :]

        if channel_names is not None and ch < len(channel_names):
            ch_name = str(channel_names[ch])
        else:
            ch_name = f"ch{ch+1}"

        ch_name = ch_name.replace(" ", "_").replace("-", "_")

        std_x = np.std(x)

        features[f"mean_{ch_name}"] = np.mean(x)
        features[f"std_{ch_name}"] = std_x
        features[f"var_{ch_name}"] = np.var(x)
        features[f"rms_{ch_name}"] = np.sqrt(np.mean(x**2))
        features[f"ptp_{ch_name}"] = np.ptp(x)
        features[f"line_length_{ch_name}"] = np.sum(np.abs(np.diff(x)))

        if std_x < 1e-6:
            features[f"skew_{ch_name}"] = np.nan
            features[f"kurtosis_{ch_name}"] = np.nan
        else:
            features[f"skew_{ch_name}"] = skew(x, bias=False)
            features[f"kurtosis_{ch_name}"] = kurtosis(x, bias=False)

    return features

def extract_frequency_features(window, fs, channel_names=None):
    """
    Extract per-channel frequency-domain features from one EEG window.

    Parameters
    ----------
    window : np.ndarray
        Shape (C, window_samples)
    fs : float
        Sampling frequency in Hz
    channel_names : list or None
        Optional list of channel names

    Returns
    -------
    features : dict
        Flat dictionary with one feature per channel
    """
    bands = {
        "delta": (0.5, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "beta":  (13.0, 30.0),
        "gamma": (30.0, 40.0),
    }

    n_channels = window.shape[0]
    features = {}

    for ch in range(n_channels):
        x = window[ch, :]

        if channel_names is not None and ch < len(channel_names):
            ch_name = str(channel_names[ch])
        else:
            ch_name = f"ch{ch+1}"

        ch_name = ch_name.replace(" ", "_").replace("-", "_")

        if len(x) < 2:
            for band_name in bands:
                features[f"{band_name}_power_{ch_name}"] = np.nan
            features[f"peak_frequency_{ch_name}"] = np.nan
            continue

        nperseg = min(len(x), int(fs * 2))
        noverlap = nperseg // 2

        freqs, psd = welch(
            x,
            fs=fs,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            detrend="constant",
            scaling="density"
        )

        for band_name, (f_low, f_high) in bands.items():
            mask = (freqs >= f_low) & (freqs < f_high)

            if np.any(mask):
                band_power = np.trapezoid(psd[mask], freqs[mask])
            else:
                band_power = np.nan

            features[f"{band_name}_power_{ch_name}"] = band_power

        eeg_mask = (freqs >= 0.5) & (freqs <= 40.0)
        if np.any(eeg_mask):
            peak_idx = np.argmax(psd[eeg_mask])
            peak_freq = freqs[eeg_mask][peak_idx]
        else:
            peak_freq = np.nan

        features[f"peak_frequency_{ch_name}"] = peak_freq

    return features

#=================================================================================
#=================================================================================
#=================================================================================
# 
# Function #7

def extract_features_from_row_cached_2_7(row, npz_base_path, file_cache):
    """
    Extract metadata + per-channel features for a single EEG window row,
    using a cache so repeated NPZ files are not reloaded.

    Parameters
    ----------
    row : pd.Series
        One row from df_final. Must contain:
        - file_name
        - start_sample
        - end_sample

    npz_base_path : str
        Directory containing the preprocessed NPZ files.

    file_cache : dict
        Dictionary used to store already loaded NPZ content by file name.

    Returns
    -------
    full_row : dict
        Dictionary containing original metadata plus extracted features.
    """

    # Get file name from the dataframe row
    file_name = row["file_name"]

    # Load the NPZ only once per file and store it in the cache
    if file_name not in file_cache:
        npz_path = os.path.join(npz_base_path, file_name)
        npz_data = np.load(npz_path, allow_pickle=True)

        file_cache[file_name] = {
            "X": npz_data["X"],
            "fs": float(npz_data["fs"]),
            "channel_names": npz_data["channel_names"]
        }

    # Retrieve cached data for the current file
    X = file_cache[file_name]["X"]
    fs = file_cache[file_name]["fs"]
    channel_names = file_cache[file_name]["channel_names"]

    # Read window boundaries from the dataframe row
    start_sample = int(row["start_sample"])
    end_sample = int(row["end_sample"])

    # Slice the current window from the recording
    window = X[:, start_sample:end_sample]

    # Extract time-domain features
    time_features = extract_time_features(
        window,
        channel_names=channel_names
    )

    # Extract frequency-domain features
    freq_features = extract_frequency_features(
        window,
        fs=fs,
        channel_names=channel_names
    )

    # Merge all extracted features
    all_features = {**time_features, **freq_features}

    # Merge original metadata and extracted features
    full_row = {**row.to_dict(), **all_features}

    return full_row