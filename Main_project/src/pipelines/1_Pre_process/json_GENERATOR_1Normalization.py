import json
from pathlib import Path
from datetime import datetime

# ==========================================================
# JSON CONFIG GENERATOR FOR EEG GLOBAL NORMALIZATION PIPELINE
# ==========================================================
# This script creates a JSON configuration file for the EEG
# global channel normalization pipeline.
#
# The user only needs to define:
# - patient ID
# - input .npz directory
# - example .npz file
# - output root directory
# - normalization parameters
#
# The output names and folders are generated automatically
# using:
# - patient_id
# - input folder name
# - normalization type
# - expected number of channels
# - epsilon value
# - current date
# ==========================================================


# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def clean_number_for_name(x):
    """
    Convert numbers into filename-safe strings.

    Examples:
    2      -> 2
    0.5    -> 0p5
    1e-8   -> 1e-8
    """
    if x is None:
        return "None"

    if isinstance(x, int):
        return str(x)

    if isinstance(x, float):
        if x.is_integer():
            return str(int(x))

        # Use scientific notation for very small numbers
        if abs(x) < 0.001:
            return f"{x:.0e}"

        return str(x).replace(".", "p")

    return str(x).replace(".", "p")


def build_experiment_id(
    patient_id,
    input_npz_dir,
    normalization_type,
    expected_n_channels,
    eps,
    date_code
):
    """
    Build a reproducible experiment ID based on input data
    and normalization parameters.
    """

    input_name = Path(input_npz_dir).name

    norm_code = normalization_type.upper()
    ch_code = f"CH{expected_n_channels}"
    eps_code = f"EPS{clean_number_for_name(eps)}"

    experiment_id = (
        f"{patient_id}_"
        f"IN-{input_name}_"
        f"{norm_code}_"
        f"{ch_code}_"
        f"{eps_code}_"
        f"{date_code}"
    )

    return experiment_id


# ==========================================================
# USER-DEFINED SETTINGS
# ==========================================================

# ----------------------------------------------------------
# PATIENT INFORMATION
# ----------------------------------------------------------
patient_id = "10OXG"

# ----------------------------------------------------------
# INPUT PATHS
# ----------------------------------------------------------
# Folder containing the original/non-normalized .npz files
input_npz_dir = "/home/tperezsanchez/Tomas_PS_DissertationKCL2026/Main_project/results/10OXG/Pre_processing/10OXG_IN-10OXG_AMP200_BP0p5-48Hz_NOTCH34p5Hz_NOZSCORE_20260612/npz"

# Example .npz file used only for inspection
example_npz_file = "/home/tperezsanchez/Tomas_PS_DissertationKCL2026/Main_project/results/10OXG/Pre_processing/10OXG_IN-10OXG_AMP200_BP0p5-48Hz_NOTCH34p5Hz_NOZSCORE_20260612/npz/10OXG_182_preproc_full.npz"

# ----------------------------------------------------------
# USER-DEFINED OUTPUT ROOT DIRECTORY
# ----------------------------------------------------------
# The user chooses ONLY this folder.
# The script will create a subfolder inside it using experiment_id.
output_root_dir = "/home/tperezsanchez/Tomas_PS_DissertationKCL2026/Main_project/results/10OXG/Pre_processing"

# ----------------------------------------------------------
# CONFIG OUTPUT DIRECTORY
# ----------------------------------------------------------
# Folder where the generated JSON config will be saved.
config_output_dir = "/home/tperezsanchez/Tomas_PS_DissertationKCL2026/Main_project/src/pipelines/1_Pre_process/configs/10OXG"


# ==========================================================
# FILE SELECTION PARAMETERS
# ==========================================================

file_selection = {
    "file_pattern": "*.npz"
}


# ==========================================================
# NORMALIZATION PARAMETERS
# ==========================================================

normalization = {
    # Type of normalization used in this pipeline
    "normalization_type": "GLOBALCH-NORM",

    # Expected number of EEG channels
    "expected_n_channels": 2,

    # Small value used to avoid division by zero
    "eps": 1e-8,

    # Whether np.load should allow pickle objects
    "allow_pickle": True,

    # Metadata keys that should be preserved in the normalized .npz files
    "metadata_keys": [
        "mu",
        "sigma",
        "fs",
        "channel_names",
        "source_file",
        "seizure_onsets",
        "T0",
        "TF"
    ]
}


# ==========================================================
# PIPELINE STEPS
# ==========================================================

pipeline_steps = {
    "inspect_example_file": True,
    "compute_global_channel_stats": True,
    "apply_global_normalization": True,
    "run_sanity_check": True
}

# ==========================================================
# SEIZURE VISUALIZATION PARAMETERS
# ==========================================================

seizure_visualization = {
    # Whether to generate one PDF/plot per seizure file
    "run_visualization": True,

    # Directory that will be used as input for visualization.
    # In this case, we visualize the normalized .npz files.
    "visualization_input_dir": "normalized_output_dir",

    # File suffix used to select files for visualization
    "file_suffix": "_preproc_full.npz",

    # Channels to plot
    "channel_idx_1": 0,
    "channel_idx_2": 1,

    # Windowing around seizure onset
    "window_sec": 10,
    "n_windows": 12,
    "pre_onset_sec": 60,

    # Vertical offset between channels
    "vertical_offset_uv": 20.0
}
# ==========================================================
# AUTOMATIC NAMING
# ==========================================================

date_code = datetime.now().strftime("%Y%m%d")
created_at = datetime.now().isoformat(timespec="seconds")

experiment_id = build_experiment_id(
    patient_id=patient_id,
    input_npz_dir=input_npz_dir,
    normalization_type=normalization["normalization_type"],
    expected_n_channels=normalization["expected_n_channels"],
    eps=normalization["eps"],
    date_code=date_code
)

# Main experiment output directory
experiment_output_dir = Path(output_root_dir) / experiment_id

# Specific output directory for normalized .npz files
normalized_output_dir = experiment_output_dir / "normalized_npz"

# Optional folder for logs or sanity-check outputs
logs_output_dir = experiment_output_dir / "logs"
viz_output_dir = experiment_output_dir / "seizure_visualizations"
# Create directories
experiment_output_dir.mkdir(parents=True, exist_ok=True)
normalized_output_dir.mkdir(parents=True, exist_ok=True)
logs_output_dir.mkdir(parents=True, exist_ok=True)
viz_output_dir.mkdir(parents=True, exist_ok=True)

# ==========================================================
# DEFINE CONFIG DICTIONARY
# ==========================================================

config = {
    # ------------------------------------------------------
    # EXPERIMENT INFORMATION
    # ------------------------------------------------------
    "experiment_id": experiment_id,
    "created_at": created_at,
    "date_code": date_code,

    # ------------------------------------------------------
    # PATIENT INFORMATION
    # ------------------------------------------------------
    "patient_id": patient_id,

    # ------------------------------------------------------
    # FILE AND FOLDER PATHS
    # ------------------------------------------------------
    "paths": {
        "input_npz_dir": str(input_npz_dir),
        "example_npz_file": str(example_npz_file),
    
        "output_root_dir": str(output_root_dir),
        "experiment_output_dir": str(experiment_output_dir),
    
        "normalized_output_dir": str(normalized_output_dir),
        "logs_output_dir": str(logs_output_dir),
        "viz_output_dir": str(viz_output_dir)
    },

    # ------------------------------------------------------
    # AUTOMATIC NAMING INFORMATION
    # ------------------------------------------------------
    "naming": {
        "input_name": Path(input_npz_dir).name,
        "normalization_code": normalization["normalization_type"],
        "channel_code": f"CH{normalization['expected_n_channels']}",
        "eps_code": f"EPS{clean_number_for_name(normalization['eps'])}",
        "date_code": date_code
    },

    # ------------------------------------------------------
    # FILE SELECTION PARAMETERS
    # ------------------------------------------------------
    "file_selection": file_selection,

    # ------------------------------------------------------
    # NORMALIZATION PARAMETERS
    # ------------------------------------------------------
    "normalization": normalization,

    # ------------------------------------------------------
    # PIPELINE STEPS
    # ------------------------------------------------------

    "pipeline_steps": pipeline_steps,

# ------------------------------------------------------
# SEIZURE VISUALIZATION PARAMETERS
# ------------------------------------------------------
    "seizure_visualization": seizure_visualization}

# ==========================================================
# SAVE JSON CONFIG FILE
# ==========================================================

config_output_dir = Path(config_output_dir)
config_output_dir.mkdir(parents=True, exist_ok=True)

config_path = config_output_dir / f"config_{experiment_id}.json"

with open(config_path, "w") as f:
    json.dump(config, f, indent=4)

print("Config saved at:")
print(config_path)

print("\nExperiment ID:")
print(experiment_id)

print("\nExperiment output directory:")
print(experiment_output_dir)

print("\nGenerated output paths:")
print(f"Normalized NPZ output dir: {normalized_output_dir}")
print(f"Logs output dir: {logs_output_dir}")
print(f"Seizure visualization output dir: {viz_output_dir}")