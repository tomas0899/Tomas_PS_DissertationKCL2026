import os
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from pathlib import Path
import sys
import json
import pandas as pd
import numpy as np
#==========================
#==========================
#==========================
# 0.1 Load modules and json config
# Get current file location
current_file = Path(__file__).resolve()

# Go up until you find the project root (where "src" exists)
for parent in current_file.parents:
    if (parent / "src").exists():
        project_root = parent
        break
# Add to PYTHONPATH if not already there
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Now import works
from src.modules import tools_EEG_FE as TEEG_FE
#==========================
#==========================
#==========================
# 0.2 Load JSON config


if len(sys.argv) > 1:
    config_path = Path(sys.argv[1])
else:
    config_path = project_root / "configs" / "config_XB47Y_labeling.json"

with open(config_path, "r") as f:
    config = json.load(f)

print(f"Loaded config from: {config_path.resolve()}")
print(f"Patient ID: {config['patient_id']}")
#==========================
#==========================
#==========================
# 0.3 Load FILES

files = Path(config["paths"]["input_npz_dir"])
file_pattern = config["file_selection"]["file_pattern"]

base_files = sorted(files.glob(file_pattern))

print(f"Input folder: {files.resolve()}")
print(f"File pattern: {file_pattern}")
print(f"Found {len(base_files)} .npz files")

rows = []


def parse_timestamp(val):
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
    return dt.replace(tzinfo=timezone.utc).timestamp() #==========================
#==========================
#==========================
# 1. load metadata
for base_NPZ_path in base_files:
    meta = {"file_name": base_NPZ_path.name, "file_path": str(base_NPZ_path.resolve()), "load_error": None}

    try:
        with np.load(base_NPZ_path, allow_pickle=True) as data:
            keys = set(data.files)
            # store data from key into meta df, convert them into float or str first
            
            meta["fs"]         = float(data["fs"]) if "fs" in keys else None
            meta["source_file"]= str(data["source_file"]) if "source_file" in keys else None
            meta["T0"] = parse_timestamp(data["T0"]) if "T0" in keys else None
            meta["TF"] = parse_timestamp(data["TF"]) if "TF" in keys else None
            meta["is_normalized"] = "mu" in keys and "sigma" in keys

            meta["channel_names"]  = list(data["channel_names"]) if "channel_names" in keys else []
            meta["seizure_onsets"] = list(data["seizure_onsets"]) if "seizure_onsets" in keys else []

            # Read X shape only — avoids loading the full array into memory
            if "X" in keys:
                shape = data["X"].shape
                meta["n_channels"] = int(shape[0]) if len(shape) == 2 else None
                meta["n_samples"]  = int(shape[1]) if len(shape) == 2 else None
            else:
                meta["n_channels"] = None
                meta["n_samples"]  = None

    except Exception as e:
        meta["load_error"] = str(e)
        print(f"Failed to load {base_NPZ_path.name}: {e}")

    rows.append(meta)

print(f"\nLoaded metadata from {len(rows)} file(s)")
#==========================
#==========================
#==========================
# 2. Temporal sanity check

from datetime import datetime

for meta in rows:
    T0 = meta.get("T0")
    TF = meta.get("TF") 

    meta["start_time"] = datetime.fromtimestamp(T0) if T0 is not None else None
    meta["end_time"]   = datetime.fromtimestamp(TF) if TF is not None else None
    meta["duration_s"] = round(TF - T0, 3) if (T0 is not None and TF is not None) else None

    # Cross-check: does n_samples / fs match TF - T0?
    fs, n = meta.get("fs"), meta.get("n_samples")
    if fs and n and meta["duration_s"] is not None:
        expected = round(n / fs, 3)
        meta["duration_check_ok"] = abs(expected - meta["duration_s"]) < 1.0
    else:
        meta["duration_check_ok"] = None

bad = [m for m in rows if m["duration_check_ok"] is False]
print(len(bad), "files with duration mismatch")

#==========================
#==========================
#==========================
# 3. General Dataframe for every file
COLUMNS = [
    "file_name", "file_path",
    "start_time", "end_time", "duration_s",
    "fs", "n_channels", "n_samples",
    "channel_names", "seizure_onsets",
    "is_normalized", "source_file",
    "duration_check_ok", "load_error",
]

df = pd.DataFrame(rows, columns=COLUMNS)
df = df.sort_values("start_time", na_position="last").reset_index(drop=True)

print(df.shape)
df.head()

print("── Sanity Report ──────────────────────────────────────")
print(f"  Total recordings   : {len(df)}")
print(f"  Load errors        : {df['load_error'].notna().sum()}")
print(f"  Missing T0/TF      : {df['start_time'].isna().sum()}")
print(f"  Duration check ✗   : {(df['duration_check_ok'] == False).sum()}")
print(f"  Missing fs         : {df['fs'].isna().sum()}")
print(f"  Normalized files   : {df['is_normalized'].sum()}")

# Check for overlapping recordings
valid = df.dropna(subset=["start_time", "end_time"]).sort_values("start_time")
overlaps = 0
for i in range(len(valid) - 1):
    if valid.iloc[i]["end_time"] > valid.iloc[i + 1]["start_time"]:
        overlaps += 1
print(f"  Overlapping pairs  : {overlaps}")
print("───────────────────────────────────────────────────────")
#==========================
#==========================
#==========================
# 4. Clean onset: solution to problems with "nan"
def clean_onsets(x):
    if isinstance(x, (list, np.ndarray)):
        return [i for i in x if not pd.isna(i)]
    elif pd.isna(x):
        return []
    else:
        return [x]

df["seizure_onsets_clean"] = df["seizure_onsets"].apply(clean_onsets)
#==========================
#==========================
#==========================
# 5. Windowing
# Create an empty list that will store one dictionary per EEG window
window_sec = config["windowing"]["window_sec"]

print(f"Window size: {window_sec} seconds")

df_windows = TEEG_FE.create_eeg_windows_2_3(
    df,
    window_sec=window_sec
)
# Convert the list of dictionaries into a new dataframe
# Each row now represents one 10-second window


print(df_windows.head())
print(df_windows.shape) # rows vs. columns
df_windows[df_windows["seizure_onsets"].apply(lambda x: not pd.isna(x).all() if isinstance(x, (list, np.ndarray)) else not pd.isna(x))].head(10)
#==========================
#==========================
#==========================
# 6. Labeling
df_labeled = TEEG_FE.initialize_labeled_dataframe_2_5_1(df_windows)

preictal_range_min = tuple(config["labeling"]["preictal_range_min"])
ictal_range_min = tuple(config["labeling"]["ictal_range_min"])
include_gap_as_interictal = config["labeling"]["include_gap_as_interictal"]

print(f"Preictal range: {preictal_range_min} minutes")
print(f"Ictal range: {ictal_range_min} minutes")
print(f"Include gap as interictal: {include_gap_as_interictal}")

df_labeled = TEEG_FE.apply_window_labeling_2_5_2(
    df_labeled=df_labeled,
    df_recordings=df,
    preictal_range_min=preictal_range_min,
    ictal_range_min=ictal_range_min,
    include_gap_as_interictal=include_gap_as_interictal
)
#==========================
#==========================
#==========================
# 7. (Optional) Filtering only 
keep_only_preictal_seizure = config["filtering"]["keep_only_preictal_seizure"]

print(f"Keep only preictal + seizure: {keep_only_preictal_seizure}")

df_final = TEEG_FE.filter_preictal_seizure_2_5_3(
    df_labeled,
    keep_only_preictal_seizure=keep_only_preictal_seizure
)
df_final.head()
df_final.shape
#==========================
#==========================
#==========================
# 8. Save df(s) as pickle
output_dir = Path(config["paths"]["output_dir"])
output_dir.mkdir(parents=True, exist_ok=True)

df_labeled_filename = config["output_files"]["df_labeled_filename"]
df_final_filename = config["output_files"]["df_final_filename"]

path_labeled = output_dir / df_labeled_filename
path_final = output_dir / df_final_filename

print(f"Output directory: {output_dir.resolve()}")
print(f"df_labeled filename: {df_labeled_filename}")
print(f"df_final filename: {df_final_filename}")

df_labeled.to_pickle(path_labeled)
df_final.to_pickle(path_final)

print(f"df_labeled saved in: {path_labeled.resolve()}")
print(f"df_final saved in: {path_final.resolve()}")