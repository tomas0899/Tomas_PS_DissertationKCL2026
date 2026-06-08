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
    "patient_id": "RQXZ1",
    "experiment_name": "features_to_pca",
    "version": "v01"
}

# ------------------------------------------------------------
# Project-level paths
# ------------------------------------------------------------
PROJECT_ROOT = Path(
    "/home/tperezsanchez/Tomas_PS_DissertationKCL2026/Main_project"
)

PATIENT_ID = USER_INFO["patient_id"]

# ------------------------------------------------------------
# Inputs
# ------------------------------------------------------------
INPUTS = {
    # Folder containing the feature dataframe
    "features_input_dir": str(
        PROJECT_ROOT
        / "results"
        / PATIENT_ID
        / "Feature_ext"
        / "Part2_features"
        / "RQXZ1_IN-normalized_npz_FP-fullnpz_W10s_PRE6to5min_ICT0to1min_GAPasINT_FINAL-PREvsSEIZ_20260510_v01_FEAT-TIME-FREQ_20260510_v01"
    ),

    # Feature dataframe filename
    "features_pickle_name": (
        "RQXZ1_IN-normalized_npz_FP-fullnpz_W10s_PRE6to5min_ICT0to1min_GAPasINT_FINAL-PREvsSEIZ_20260510_v01_FEAT-TIME-FREQ_20260510_v01_df_features_ictalVspreictal.pkl"
    ),

    # Input type expected by the PCA pipeline
    "input_type": "pkl"
}

# ------------------------------------------------------------
# Outputs
# ------------------------------------------------------------
OUTPUTS = {
    # Base folder where PCA experiment folders will be created
    "output_base_dir": str(
        PROJECT_ROOT
        / "results"
        / PATIENT_ID
        / "Feature_ext"
        / "Part3_PCA"
    ),

    # Folder where this generated config JSON will be saved
    "config_output_dir": str(
        PROJECT_ROOT
        / "src"
        / "pipelines"
        / "2_Feature_ext"
        / "configs"
        / PATIENT_ID
    )
}


# ============================================================
# 2. PARAMETERS
# ============================================================

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
# 3. AUTOMATIC INPUT PATH
# ============================================================

features_pickle = (
    Path(INPUTS["features_input_dir"])
    / INPUTS["features_pickle_name"]
)

if not features_pickle.exists():
    raise FileNotFoundError(f"Feature pickle not found: {features_pickle}")


# ============================================================
# 4. AUTOMATIC NAMING SECTION
# ============================================================

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
# 5. PCA OUTPUT PATHS
# ============================================================

output_dir = Path(OUTPUTS["output_base_dir"]) / experiment_id

output_paths = {
    "output_dir": str(output_dir),
    "pca_dataframe_pickle": str(output_dir / f"{experiment_id}_df_windows_pca.pkl"),
    "explained_variance_csv": str(output_dir / f"{experiment_id}_explained_variance.csv"),
    "config_copy_json": str(output_dir / f"{experiment_id}_config.json")
}


# ============================================================
# 6. GENERATED CONFIG OUTPUT PATH
# ============================================================

config_output_dir = Path(OUTPUTS["config_output_dir"])
config_output_dir.mkdir(parents=True, exist_ok=True)

config_output_path = config_output_dir / f"config_{experiment_id}.json"

# Add generated config path to the config itself
output_paths["generated_config_json"] = str(config_output_path)


# ============================================================
# 7. FINAL CONFIG STRUCTURE
# ============================================================

config = {
    "metadata": {
        "config_type": "PCA_feature_dataframe_config",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "description": "Config file for applying PCA to EEG feature dataframe."
    },

    "user_info": USER_INFO,

    "inputs": {
        "features_input_dir": INPUTS["features_input_dir"],
        "features_pickle_name": INPUTS["features_pickle_name"],
        "features_pickle": str(features_pickle),
        "input_type": INPUTS["input_type"]
    },

    "outputs": output_paths,

    "parameters": PARAMETERS
}


# ============================================================
# 8. SAVE CONFIG JSON
# ============================================================

with open(config_output_path, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=4)

print("Config JSON generated successfully.")
print(f"Experiment ID: {experiment_id}")
print(f"Config saved to: {config_output_path.resolve()}")
print()
print("Input path:")
print(f"- features_pickle: {features_pickle}")
print()
print("Main output paths that the PCA pipeline should use:")
for key, value in output_paths.items():
    print(f"- {key}: {value}")