from pathlib import Path
import sys
import json
import pandas as pd
import numpy as np
import os
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
from src.modules import tools_EEG_Preprocess as TEEG_PR


# ==========================================================
# 0.2 JSON CONFIG
# Usage:
# python run_normalization.py configs/config_XB47Y_global_normalization.json

if len(sys.argv) > 1:
    config_path = Path(sys.argv[1])
else:
    config_path = project_root / "configs" / "config_XB47Y_global_normalization.json"

# Check config exists
if not config_path.exists():
    raise FileNotFoundError(f"Config file not found: {config_path}")

# Load config
with open(config_path, "r") as f:
    config = json.load(f)
# ==========================================================
# 0.3 EXTRACT VALUES FROM CONFIG


# --------------------------
# Patient
# --------------------------
patient_id = config["patient_id"]

# --------------------------
# Paths
# --------------------------
input_npz_dir = Path(config["paths"]["input_npz_dir"])
example_npz_file = Path(config["paths"]["example_npz_file"])
normalized_output_dir = Path(config["paths"]["normalized_output_dir"])

# --------------------------
# File selection
# --------------------------
file_pattern = config["file_selection"]["file_pattern"]

# --------------------------
# Normalization parameters
# --------------------------
expected_n_channels = config["normalization"]["expected_n_channels"]
eps = config["normalization"]["eps"]
allow_pickle = config["normalization"]["allow_pickle"]
metadata_keys = config["normalization"]["metadata_keys"]

# --------------------------
# Pipeline steps
# --------------------------
inspect_example_file = config["pipeline_steps"]["inspect_example_file"]
compute_stats = config["pipeline_steps"]["compute_global_channel_stats"]
apply_normalization = config["pipeline_steps"]["apply_global_normalization"]
run_sanity_check = config["pipeline_steps"]["run_sanity_check"]


#==========================
#==========================
#==========================
# 1. load files
# Load example file from JSON config
if inspect_example_file:
    data = np.load(example_npz_file, allow_pickle=allow_pickle)

    print("Keys found:", list(data.keys()))
    
# expected: key found ['X', 'mu', 'sigma', 'fs', 'channel_names', 'source_file', 'seizure_onsets', 'T0', 'TF']
#==========================
#==========================
#==========================
# 2. file inspection
if inspect_example_file:
    X = data["X"]

    print("Type of data:", X.dtype)
    print("Shape:", X.shape)
    print(f"  → Channels (C): {X.shape[0]}")
    print(f"  → Samples (N): {X.shape[1]}")
    print()
    print("Min value:", np.nanmin(X))
    print("Max value:", np.nanmax(X))
    print("NaN?:", np.any(np.isnan(X)))
    print("Inf?:", np.any(np.isinf(X)))

    if X.shape[0] != expected_n_channels:
        raise ValueError(
            f"Expected {expected_n_channels} channels, but found {X.shape[0]}"
        )


# Find all .npz files from JSON config
all_files = sorted(input_npz_dir.glob(file_pattern))

print(f"Files found: {len(all_files)}")

if len(all_files) == 0:
    raise FileNotFoundError(
        f"No files found in {input_npz_dir} using pattern {file_pattern}"
    )

#print(f"Files found: {len(all_files)}")

#==========================
#==========================
#==========================
# 3. Calculate mean and std per channel
if compute_stats:
    ch_mean, ch_std, ch_count, skipped = TEEG_PR_PR.compute_global_channel_stats_1_15(all_files)

    print("Global channel statistics computed")
    print("Channel means:", ch_mean)
    print("Channel stds:", ch_std)
    print("Channel counts:", ch_count)
    print("Skipped files:", len(skipped))

#==========================
#==========================
#==========================
# 4. Data normalization per recording + save it in a new npz
if apply_normalization:
    normalized_output_dir.mkdir(parents=True, exist_ok=True)

    stats_npz_path, summary = TEEG_PR.apply_global_channel_normalization_1_16(
        all_files=all_files,
        ch_mean=ch_mean,
        ch_std=ch_std,
        ch_count=ch_count,
        output_dir=normalized_output_dir
    )

    print("Normalization completed")
    print("Stats saved at:", stats_npz_path)
    print("Summary:", summary)
#=====================
#==========================
#==========================
# 5. Sanity Check
if run_sanity_check:
    df_check = TEEG_PR.sanity_check_global_zscore_npz_1_14(
        original_dir=input_npz_dir,
        normalized_dir=normalized_output_dir
    )

    print("Sanity check completed")
    print(df_check.head())

#=====================
#==========================
#==========================
# 6. PDF per seizure:
# --------------------------
# Seizure visualization parameters
# --------------------------
#=====================


# --------------------------
# Seizure visualization parameters
# --------------------------
seizure_viz_config = config["seizure_visualization"]

run_visualization = seizure_viz_config["run_visualization"]
visualization_input_dir_key = seizure_viz_config["visualization_input_dir"]
file_suffix = seizure_viz_config["file_suffix"]

channel_idx_1 = seizure_viz_config["channel_idx_1"]
channel_idx_2 = seizure_viz_config["channel_idx_2"]
window_sec = seizure_viz_config["window_sec"]
n_windows = seizure_viz_config["n_windows"]
pre_onset_sec = seizure_viz_config["pre_onset_sec"]
vertical_offset_uv = seizure_viz_config["vertical_offset_uv"]

viz_output_dir = Path(config["paths"]["viz_output_dir"])

# Decide which directory to use as input for visualization
if visualization_input_dir_key == "normalized_output_dir":
    visualization_input_dir = normalized_output_dir
elif visualization_input_dir_key == "input_npz_dir":
    visualization_input_dir = input_npz_dir
else:
    visualization_input_dir = Path(visualization_input_dir_key)


if run_visualization:
    viz_output_dir.mkdir(parents=True, exist_ok=True)

    print("\nRunning seizure visualization")
    print("Visualization input directory:", visualization_input_dir)
    print("Visualization output directory:", viz_output_dir)

    files_to_visualize = sorted(visualization_input_dir.glob(f"*{file_suffix}"))

    print(f"Files selected for visualization: {len(files_to_visualize)}")

    if len(files_to_visualize) == 0:
        raise FileNotFoundError(
            f"No files found in {visualization_input_dir} ending with {file_suffix}"
        )

    for npz_path in files_to_visualize:
        print(f"\nProcessing: {npz_path.name}")

        TEEG_PR.visualize_seizure_windows_from_npz_1_10VNormal(
            npz_path=str(npz_path),
            channel_idx_1=channel_idx_1,
            channel_idx_2=channel_idx_2,
            window_sec=window_sec,
            n_windows=n_windows,
            pre_onset_sec=pre_onset_sec,
            vertical_offset_uv=vertical_offset_uv,
            output_dir=str(viz_output_dir)
        )

    print("\nSeizure visualization completed")
print("\nPipeline summary:")
print("patient_id:", patient_id)
print("input_npz_dir:", input_npz_dir)
print("example_npz_file:", example_npz_file)
print("normalized_output_dir:", normalized_output_dir)
print("viz_output_dir:", viz_output_dir)
