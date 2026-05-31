from pathlib import Path
from datetime import datetime
import json

# ============================================================
# JSON CONFIG GENERATOR FOR PCA ON EEG FEATURE DATAFRAME
# ============================================================
# Purpose:
#   Generate a JSON config file for a PCA pipeline.
#
# Input expected:
#   A .pkl dataframe containing:
#       - metadata columns
#       - EEG feature columns
#
# Output expected from the later PCA script:
#   - dataframe: metadata + PC columns
#   - explained variance CSV
#
# Important:
#   This script only creates the config JSON.
#   It does NOT run PCA.
# ============================================================


# ============================================================
# 1. USER FILL SECTION
# ============================================================

USER_INFO = {
    "patient_id": "JYXFE",
    "experiment_name": "features_to_pca",
    "version": "v01"
}

INPUTS = {
    # Main input dataframe
    "features_pickle": "/home/tperezsanchez/FoundationModel_EEG_Dissertation/Main_project/results/JYXFE/Feature_ext/Part2_features/JYXFE_IN-normalized_npz_FP-fullnpz_W10s_PRE8to5min_ICT0to3min_GAPasINT_FINAL-PREvsSEIZ_20260519_v01_FEAT-TIME-FREQ_20260519_v01/JYXFE_IN-normalized_npz_FP-fullnpz_W10s_PRE8to5min_ICT0to3min_GAPasINT_FINAL-PREvsSEIZ_20260519_v01_FEAT-TIME-FREQ_20260519_v01_df_features_ictalVspreictal.pkl",

    # Input type expected by the PCA pipeline
    "input_type": "pkl"
}

OUTPUTS = {
    # If None, PCA outputs will be saved in the same parent folder as the input pickle
    "output_base_dir": None,

    # Directory where this generated config JSON will be saved
    # If None, the config will be saved in the current working directory
    "config_output_dir": "/home/tperezsanchez/FoundationModel_EEG_Dissertation/Main_project/src/pipelines/2_Feature_ext/configs"
}

PARAMETERS = {
    # Metadata columns are kept and re-attached after PCA
    "metadata_cols": [
        "file_name",
        "window_id",
        "start_sample",
        "end_sample",
        "fs",
        "n_channels",
        "window_sec",
        "seizure_onsets",
        "window_start_time",
        "window_end_time",
        "class_label",
        "label_name",
        "excluded_reason"
    ],

    # Label column kept for traceability
    "target_col": "class_label",

    # Scaling before PCA
    "scaling": {
        "apply": True,
        "method": "StandardScaler"
    },

    # PCA settings
    # mode options:
    #   "variance_threshold" -> automatically choose enough PCs to reach threshold
    #   "fixed_n_components" -> use fixed_n_components
    "pca": {
        "mode": "variance_threshold",
        "explained_variance_threshold": 0.90,
        "fixed_n_components": None
    },

    # Safety checks for the later PCA script
    "checks": {
        "stop_if_nan": True,
        "stop_if_inf": True,
        "ignore_missing_metadata_cols": True,
        "stop_if_target_missing": True
    },

    # Extra outputs
    "save": {
        "pca_dataframe_pickle": True,
        "explained_variance_csv": True,
        "config_copy_json": False
    }
}


# ============================================================
# 2. AUTOMATIC NAMING SECTION
# ============================================================

features_pickle = Path(INPUTS["features_pickle"])
input_stem = features_pickle.stem

today = datetime.now().strftime("%Y%m%d")
patient_id = USER_INFO["patient_id"]
version = USER_INFO["version"]

pca_mode = PARAMETERS["pca"]["mode"]
variance_threshold = PARAMETERS["pca"]["explained_variance_threshold"]
fixed_n_components = PARAMETERS["pca"]["fixed_n_components"]

if pca_mode == "variance_threshold":
    pca_tag = f"PCA_VAR{int(variance_threshold * 100)}"
elif pca_mode == "fixed_n_components":
    pca_tag = f"PCA_N{fixed_n_components}"
else:
    raise ValueError("pca.mode must be 'variance_threshold' or 'fixed_n_components'.")

experiment_id = f"{patient_id}_{input_stem}_{pca_tag}_{today}_{version}"


# ============================================================
# 2.1 PCA OUTPUT PATHS
# ============================================================

if OUTPUTS["output_base_dir"] is None:
    output_dir = features_pickle.parent / experiment_id
else:
    output_dir = Path(OUTPUTS["output_base_dir"]) / experiment_id

output_paths = {
    "output_dir": str(output_dir),
    "pca_dataframe_pickle": str(output_dir / f"{experiment_id}_df_windows_pca.pkl"),
    "explained_variance_csv": str(output_dir / f"{experiment_id}_explained_variance.csv"),
    "config_copy_json": str(output_dir / f"{experiment_id}_config.json")
}


# ============================================================
# 2.2 GENERATED CONFIG OUTPUT PATH
# ============================================================

if OUTPUTS["config_output_dir"] is None:
    config_output_dir = Path.cwd()
else:
    config_output_dir = Path(OUTPUTS["config_output_dir"])

config_output_dir.mkdir(parents=True, exist_ok=True)

config_output_path = config_output_dir / f"config_{experiment_id}_config.json"


# Optional: add generated config path to the config itself
output_paths["generated_config_json"] = str(config_output_path)


# ============================================================
# 3. FINAL CONFIG STRUCTURE
# ============================================================

config = {
    "metadata": {
        "config_type": "PCA_feature_dataframe_config",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "description": "Config file for applying PCA to EEG feature dataframe."
    },

    "user_info": USER_INFO,

    "inputs": {
        "features_pickle": str(features_pickle),
        "input_type": INPUTS["input_type"]
    },

    "outputs": output_paths,

    "parameters": PARAMETERS
}


# ============================================================
# 4. SAVE CONFIG JSON
# ============================================================

with open(config_output_path, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=4)

print("Config JSON generated successfully.")
print(f"Experiment ID: {experiment_id}")
print(f"Config saved to: {config_output_path.resolve()}")
print()
print("Main output paths that the PCA pipeline should use:")
for key, value in output_paths.items():
    print(f"- {key}: {value}")