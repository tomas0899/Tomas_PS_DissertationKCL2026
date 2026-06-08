# ============================================================
# JSON CONFIG GENERATOR - DECISION TREE + RANDOM FOREST
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
    "patient_id": "XB47Y",
    "experiment_name": "DT_RF_models",
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
    # Folder containing the input dataframe.
    # This can contain either:
    # 1. PCA dataframe
    # 2. Features dataframe
    "input_dir": str(
        PROJECT_ROOT
        / "results"
        / PATIENT_ID
        / "Feature_ext"
        / "Part2_features"
        / "XB47Y_IN-normalized_npz_FP-fullnpz_W10s_PRE6to5min_ICT0to1min_GAPasINT_FINAL-PREvsSEIZ_20260504_v01_FEAT-TIME-FREQ_20260505_v01"
    ),

    # Input dataframe filename
    "input_filename": "df_windowsXB47Y_pca.pkl",

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
    # Base folder where DT/RF experiment folders will be created
    "output_base_dir": str(
        PROJECT_ROOT
        / "results"
        / PATIENT_ID
        / "DT_RF_models"
    ),

    # Folder where this generated config JSON will be saved
    "config_output_dir": str(
        PROJECT_ROOT
        / "src"
        / "pipelines"
        / "4_Models"
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

# Random state for reproducibility
random_state = 42


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
    f"DT-RF_SCORING-{scoring_clean}_{today}_{version}"
)

dt_tag = f"{experiment_tag}_DecisionTree"
rf_tag = f"{experiment_tag}_RandomForest"


# ============================================================
# 2. DEFINE OUTPUT DIRECTORIES
# ============================================================

output_base_dir = Path(OUTPUTS["output_base_dir"])

main_eval_output_dir = output_base_dir / experiment_tag

eval_output_dir_dt = main_eval_output_dir / "Decision_Tree"
eval_output_dir_rf = main_eval_output_dir / "Random_Forest"

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
    # Main experiment folder
    "main_eval_output_dir": str(main_eval_output_dir),

    # Separate model output folders
    "eval_output_dir_dt": str(eval_output_dir_dt),
    "eval_output_dir_rf": str(eval_output_dir_rf),

    # Prefixes used by evaluation functions
    "output_prefix_dt": dt_tag,
    "output_prefix_rf": rf_tag,

    # Generated config JSON
    "generated_config_json": str(config_output_path),

    # Optional copy of the config inside the experiment output folder
    "config_copy_json": str(main_eval_output_dir / config_filename),

    # Decision Tree model/grid-search outputs
    "dt_best_model_pickle": str(eval_output_dir_dt / f"{dt_tag}_best_model.pkl"),
    "dt_best_params_json": str(eval_output_dir_dt / f"{dt_tag}_best_params.json"),
    "dt_grid_search_results_csv": str(eval_output_dir_dt / f"{dt_tag}_grid_search_results.csv"),

    # Random Forest model/grid-search outputs
    "rf_best_model_pickle": str(eval_output_dir_rf / f"{rf_tag}_best_model.pkl"),
    "rf_best_params_json": str(eval_output_dir_rf / f"{rf_tag}_best_params.json"),
    "rf_grid_search_results_csv": str(eval_output_dir_rf / f"{rf_tag}_grid_search_results.csv"),

    # Optional split/QC output
    "split_summary_csv": str(main_eval_output_dir / f"{experiment_tag}_split_summary.csv")
}


# ============================================================
# 5. BUILD CONFIG DICTIONARY
# ============================================================

config = {
    "experiment_info": {
        "experiment_tag": experiment_tag,
        "pipeline_name": "DT_RF_models",
        "patient_id": patient_id,
        "input_data_type": input_data_type_clean,
        "models": [
            "DecisionTreeClassifier",
            "RandomForestClassifier"
        ],
        "scoring": scoring,
        "date": today,
        "version": version,
        "random_state": random_state
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

    "models": {
        "decision_tree": {
            "model_type": "DecisionTreeClassifier",
            "class_weight": "balanced",
            "random_state": random_state,
            "uses_scaler": False,
            "uses_pca_input": input_data_type_clean == "PCA"
        },

        "random_forest": {
            "model_type": "RandomForestClassifier",
            "class_weight": "balanced",
            "random_state": random_state,
            "uses_scaler": False,
            "uses_pca_input": input_data_type_clean == "PCA"
        }
    },

    "model_training": {
        "hyperparameter_grids_defined_in_module": True,
        "module": "tools_EEG_models",
        "decision_tree_function": "train_decision_tree_gridsearch_3_4",
        "random_forest_function": "train_random_forest_gridsearch_3_5"
    },

    "grid_search": {
        "uses_default_parameters_from_module": True,
        "cv_strategy": "TimeSeriesSplit",
        "default_n_splits": 4,
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
print("Main output paths that the DT/RF pipeline should use:")
for key, value in output_paths.items():
    print(f"- {key}: {value}")