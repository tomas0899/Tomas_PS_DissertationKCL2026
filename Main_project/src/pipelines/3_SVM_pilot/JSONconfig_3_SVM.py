# ============================================================
# JSON CONFIG GENERATOR - SVM PILOT
# ============================================================

from pathlib import Path
import json
from datetime import datetime


# ============================================================
# USER EDITABLE SECTION
# ============================================================
# Edit only this section for each new experiment.
# The rest of the script builds the config automatically.
# ============================================================

# Patient ID
patient_id = "XB47Y"

# Input dataframe path.
# This can be either:
# 1. PCA dataframe
# 2. Features dataframe
input_path = Path(
    "/home/tperezsanchez/FoundationModel_EEG_Dissertation/Main_project/results/XB47Y/Feature_ext/Part2_features/XB47Y_IN-normalized_npz_FP-fullnpz_W10s_PRE6to5min_ICT0to1min_GAPasINT_FINAL-PREvsSEIZ_20260504_v01_FEAT-TIME-FREQ_20260505_v01/df_windowsXB47Y_pca.pkl"
)

# Input data type for automatic naming.
# Options:
# "PCA"
# "FEATURES"
input_data_type = "PCA"

# Metric used to select the best model in grid search.
# Examples:
# "f1_macro"
# "balanced_accuracy"
# "accuracy"
# "recall_macro"
scoring = "f1_macro"

# Config version
version = "v01"


# ============================================================
# AUTOMATIC SECTION
# Do not edit unless you want to change the generator logic.
# ============================================================

# ============================================================
# 1. AUTOMATIC PROJECT PATH DETECTION
# ============================================================

current_file = Path(__file__).resolve()

project_root = None

for parent in current_file.parents:
    if (parent / "src").exists() and (parent / "results").exists():
        project_root = parent
        break

if project_root is None:
    raise RuntimeError(
        "Project root not found. Could not find a parent folder containing 'src' and 'results'."
    )


# ============================================================
# 2. CLEAN USER INPUTS
# ============================================================

input_data_type_clean = input_data_type.upper()

if input_data_type_clean not in ["PCA", "FEATURES"]:
    raise ValueError(
        "input_data_type must be either 'PCA' or 'FEATURES'."
    )

scoring_clean = scoring.replace("_", "-").upper()

today = datetime.today().strftime("%Y%m%d")


# ============================================================
# 3. DEFINE OUTPUT PATHS
# ============================================================

# Main output directory for SVM results.
# This adapts automatically to each patient.
eval_output_dir = (
    project_root
    / "results"
    / patient_id
    / "SVM_pilot"
)

# Directory where the generated JSON config will be saved.
config_output_dir = (
    project_root
    / "src"
    / "pipelines"
    / "3_SVM_pilot"
    / "configs"
)

eval_output_dir.mkdir(parents=True, exist_ok=True)
config_output_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# 4. AUTOMATIC CONFIG FILE NAME
# ============================================================

config_filename = (
    f"config_{patient_id}_IN-{input_data_type_clean}_"
    f"SVM-SCORING-{scoring_clean}_{today}_{version}.json"
)

config_output_path = config_output_dir / config_filename


# ============================================================
# 5. BUILD CONFIG DICTIONARY
# ============================================================

config = {
    "experiment_info": {
        "pipeline_name": "SVM_pilot",
        "patient_id": patient_id,
        "input_data_type": input_data_type_clean,
        "scoring": scoring,
        "date": today,
        "version": version
    },

    "inputs": {
        "input_path": str(input_path)
    },

    "outputs": {
        "eval_output_dir": str(eval_output_dir)
    },

    "data_processing": {
        "time_column": "window_start_time",
        "target_column": "class_label",

        "metadata_cols": [
            "window_id",
            "start_sample",
            "end_sample",
            "fs",
            "n_channels",
            "window_sec",
            "seizure_onsets",
            "file_name",
            "window_start_time",
            "window_end_time",
            "class_label",
            "label_name",
            "excluded_reason"
        ],

        "label_mapping": {
            "1": 0,
            "2": 1
        },

        "class_names": [
            "preictal",
            "seizure"
        ],

        "labels": [
            0,
            1
        ]
    },

    "temporal_split": {
        "ideal_train": 0.70,
        "ideal_val": 0.15,
        "ideal_test": 0.15,

        "train_search_range": [
            0.70,
            0.90
        ],

        "val_search_range": [
            0.05,
            0.20
        ],

        "ratio_weight": 3
    },

    "model": {
        "model_type": "SVM",
        "kernel": "rbf",
        "class_weight": "balanced"
    },

    "grid_search": {
        "n_splits": 4,
        "scoring": scoring,
        "n_jobs": -1,
        "verbose": 1
    },

    "evaluation": {
        "datasets": [
            "Validation",
            "Test"
        ],
        "show_plot": True
    }
}


# ============================================================
# 6. SAVE JSON CONFIG
# ============================================================

with open(config_output_path, "w") as f:
    json.dump(config, f, indent=4)

print("JSON config saved successfully:")
print(config_output_path)