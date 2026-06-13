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

USER_INFO = {
    "patient_id": "10OXG",
    "experiment_name": "SVM_pilot",
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
    # Folder containing the input dataframe
    # This can contain either:
    # 1. PCA dataframe
    # 2. Features dataframe
    "input_dir": str(
        PROJECT_ROOT
        / "results"
        / PATIENT_ID
        / "Feature_ext"
        / "Part3_PCA"
        / "10OXG_10OXG_IN-normalized_npz_FP-fullnpz_W10s_PRE6to5min_ICT0to1min_GAPasINT_FINAL-PREvsSEIZ_20260612_v01_FEAT-TIME-FREQ_20260612_v01_df_features_ictalVspreictal_dropExcluded_dropNaNrows_PCA_VAR90_20260612_v01"
    ),

    # Input dataframe filename
    "input_filename": "10OXG_10OXG_IN-normalized_npz_FP-fullnpz_W10s_PRE6to5min_ICT0to1min_GAPasINT_FINAL-PREvsSEIZ_20260612_v01_FEAT-TIME-FREQ_20260612_v01_df_features_ictalVspreictal_dropExcluded_dropNaNrows_PCA_VAR90_20260612_v01_df_windows_pca.pkl",

    # Input data type for automatic naming.
    # Options:
    # "PCA"
    # "FEATURES"
    "input_data_type": "PCA",

    # Input file type
    "input_type": "pkl"
}

# ------------------------------------------------------------
# Outputs
# ------------------------------------------------------------

OUTPUTS = {
    # Base folder where SVM experiment folders will be created
    "output_base_dir": str(
        PROJECT_ROOT
        / "results"
        / PATIENT_ID
        / "SVM_pilot"
    ),

    # Folder where this generated config JSON will be saved
    "config_output_dir": str(
        PROJECT_ROOT
        / "src"
        / "pipelines"
        / "3_SVM_pilot"
        / "configs"
        / PATIENT_ID
    )
}

# ------------------------------------------------------------
# Model selection
# ------------------------------------------------------------

# Metric used to select the best model in grid search.
# Examples:
# "f1_macro"
# "balanced_accuracy"
# "accuracy"
# "recall_macro"
scoring = "f1_macro"


# ============================================================
# AUTOMATIC SECTION
# Do not edit unless you want to change the generator logic.
# ============================================================

# ============================================================
# 1. CLEAN USER INPUTS
# ============================================================

patient_id = USER_INFO["patient_id"]
version = USER_INFO["version"]

input_path = Path(INPUTS["input_dir"]) / INPUTS["input_filename"]

input_data_type_clean = INPUTS["input_data_type"].upper()

if input_data_type_clean not in ["PCA", "FEATURES"]:
    raise ValueError(
        "input_data_type must be either 'PCA' or 'FEATURES'."
    )

if not input_path.exists():
    raise FileNotFoundError(f"Input dataframe not found: {input_path}")

scoring_clean = scoring.replace("_", "-").upper()

today = datetime.today().strftime("%Y%m%d")

experiment_tag = (
    f"{patient_id}_IN-{input_data_type_clean}_"
    f"SVM-SCORING-{scoring_clean}_{today}_{version}"
)


# ============================================================
# 2. DEFINE OUTPUT DIRECTORIES
# ============================================================

output_base_dir = Path(OUTPUTS["output_base_dir"])
eval_output_dir = output_base_dir / experiment_tag

config_output_dir = Path(OUTPUTS["config_output_dir"])
config_output_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. AUTOMATIC CONFIG FILE NAME
# ============================================================

config_filename = f"config_{experiment_tag}.json"
config_output_path = config_output_dir / config_filename


# ============================================================
# 4. DEFINE OUTPUT PATHS
# ============================================================

output_paths = {
    # Main evaluation folder
    "eval_output_dir": str(eval_output_dir),

    # Prefix used by evaluation functions
    "output_prefix": experiment_tag,

    # Generated config JSON
    "generated_config_json": str(config_output_path),

    # Optional copy of the config inside the experiment output folder
    "config_copy_json": str(eval_output_dir / config_filename),

    # Optional model/grid-search outputs
    "best_model_pickle": str(eval_output_dir / f"{experiment_tag}_best_model.pkl"),
    "best_params_json": str(eval_output_dir / f"{experiment_tag}_best_params.json"),
    "grid_search_results_csv": str(eval_output_dir / f"{experiment_tag}_grid_search_results.csv"),

    # Optional split/QC outputs
    "split_summary_csv": str(eval_output_dir / f"{experiment_tag}_split_summary.csv")
}


# ============================================================
# 5. BUILD CONFIG DICTIONARY
# ============================================================

config = {
    "experiment_info": {
        "experiment_tag": experiment_tag,
        "pipeline_name": "SVM_pilot",
        "patient_id": patient_id,
        "input_data_type": input_data_type_clean,
        "scoring": scoring,
        "date": today,
        "version": version
    },

    "inputs": {
        "input_dir": INPUTS["input_dir"],
        "input_filename": INPUTS["input_filename"],
        "input_path": str(input_path),
        "input_type": INPUTS["input_type"]
    },

    "outputs": output_paths,

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
        "show_plot": False
    }
}


# ============================================================
# 6. SAVE JSON CONFIG
# ============================================================

with open(config_output_path, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=4)

print("JSON config saved successfully.")
print(f"Experiment tag: {experiment_tag}")
print(f"Config saved to: {config_output_path.resolve()}")
print()
print("Input path:")
print(f"- input_path: {input_path.resolve()}")
print()
print("Main output paths that the SVM pipeline should use:")
for key, value in output_paths.items():
    print(f"- {key}: {value}")