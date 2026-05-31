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
    config_path = project_root / "configs" / "config_XB47Y_feature_extraction.json"

with open(config_path, "r") as f:
    config = json.load(f)

print(f"Loaded config from: {config_path.resolve()}")
print(f"Patient ID: {config['patient_id']}")
#==========================
#==========================
#==========================
# 0.2 Load df Pickles

input_pickle_dir = Path(config["paths"]["input_pickle_dir"])

df_labeled_all_path = input_pickle_dir / config["input_files"]["df_labeled_all"]
df_ictalVspreictal_path = input_pickle_dir / config["input_files"]["df_ictalVspreictal"]

df_labeled_all = pd.read_pickle(df_labeled_all_path)
df_final_ictalVspreictal = pd.read_pickle(df_ictalVspreictal_path)

print(f"Loaded df_labeled_all from: {df_labeled_all_path.resolve()}")
print(f"Loaded df_final_ictalVspreictal from: {df_ictalVspreictal_path.resolve()}")

print("df_labeled_all shape:", df_labeled_all.shape)
print("df_final_ictalVspreictal shape:", df_final_ictalVspreictal.shape)

print(df_labeled_all.shape)
print(df_final_ictalVspreictal.shape)
#==========================
#==========================
#==========================
# 0.3 load path with all the preprocessed npz
npz_base_path = Path(config["paths"]["input_npz_dir"])

print(f"NPZ base path: {npz_base_path.resolve()}")
#==========================
#==========================
#==========================
# 1. df only including ictal vs. preictal
if config["pipeline_steps"]["run_features_ictalVspreictal"]:

    rows_with_features = []
    file_cache = {}

    for idx, row in df_final_ictalVspreictal.iterrows():
        full_row = TEEG_FE.extract_features_from_row_cached_2_7(
            row,
            npz_base_path=npz_base_path,
            file_cache=file_cache
        )
        rows_with_features.append(full_row)

    df_features_ictalVspreictal = pd.DataFrame(rows_with_features)

    print("df_features_ictalVspreictal shape:", df_features_ictalVspreictal.shape)
    print("Number of cached files:", len(file_cache))
    print(df_features_ictalVspreictal.head())

else:
    df_features_ictalVspreictal = None
    print("Skipped ictal vs preictal feature extraction.")
#==========================
#==========================
#==========================
# 2. df for all the windows
if config["pipeline_steps"]["run_features_all"]:

    rows_with_features = []
    file_cache = {}

    for idx, row in df_labeled_all.iterrows():
        full_row = TEEG_FE.extract_features_from_row_cached_2_7(
            row,
            npz_base_path=npz_base_path,
            file_cache=file_cache
        )
        rows_with_features.append(full_row)

    df_features_all = pd.DataFrame(rows_with_features)

    print("df_features_all shape:", df_features_all.shape)
    print("Number of cached files:", len(file_cache))
    print(df_features_all.head())

else:
    df_features_all = None
    print("Skipped all-windows feature extraction.")
#==========================
#==========================
#==========================
# 3. save both df as pickles files

if config["pipeline_steps"]["save_outputs"]:

    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    df_features_all_filename = config["output_files"]["df_features_all"]
    df_features_ictalVspreictal_filename = config["output_files"]["df_features_ictalVspreictal"]

    path_all = output_dir / df_features_all_filename
    path_ictalVspreictal = output_dir / df_features_ictalVspreictal_filename

    print(f"Output directory: {output_dir.resolve()}")
    print(f"df_features_all filename: {df_features_all_filename}")
    print(f"df_features_ictalVspreictal filename: {df_features_ictalVspreictal_filename}")
    if df_features_all is not None:
        df_features_all.to_pickle(path_all)
        print(f"df_features_all saved in: {path_all.resolve()}")

    if df_features_ictalVspreictal is not None:
        df_features_ictalVspreictal.to_pickle(path_ictalVspreictal)
        print(f"df_features_ictalVspreictal saved in: {path_ictalVspreictal.resolve()}")