# ============================================================
# JSON CONFIG GENERATOR
# Pipeline: SVM pilot model for PCA/features dataframe
# ============================================================

from pathlib import Path
from datetime import datetime
import json
import re


# ============================================================
# 0. FIND PROJECT ROOT
# ============================================================
# This block automatically finds the project root folder.
# It searches upwards from the current script location until it finds
# a parent folder containing "src".
#
# If running in a notebook, it falls back to the current working directory.

try:
    current_file = Path(__file__).resolve()
except NameError:
    current_file = Path.cwd().resolve()

project_root = None

for parent in [current_file] + list(current_file.parents):
    if (parent / "src").exists():
        project_root = parent
        break

if project_root is None:
    raise RuntimeError(
        "Project root not found. Could not find a parent folder containing 'src'."
    )


# ============================================================
# 1. USER SECTION - EDIT THIS
# ============================================================

# -------------------------------
# Patient / experiment information
# -------------------------------

patient_id = "XB47Y"

# Choose one:
# "pca"      -> dataframe already transformed into PCA components
# "features" -> dataframe with original extracted EEG features
input_dataframe_type = "pca"

# Short experiment label.
# This will be included in the config filename and output folders.
experiment_label = "SVM-PCA-TEMPORAL"

# Version of this config.
config_version = "v01"


# -------------------------------
# Input pickle dataframe
# -------------------------------

input_pickle_path = Path(
    "/home/tperezsanchez/FoundationModel_EEG_Dissertation/Main_project/results/XB47Y/Feature_ext/Part2_features/XB47Y_IN-normalized_npz_FP-fullnpz_W10s_PRE6to5min_ICT0to1min_GAPasINT_FINAL-PREvsSEIZ_20260504_v01_FEAT-TIME-FREQ_20260505_v01/df_windowsXB47Y_pca.pkl"
)


# -------------------------------
# Config output folder
# -------------------------------
# Final configs will be saved here.

output_config_dir = (
    project_root
    / "src"
    / "pipelines"
    / "3_SVM_pilot"
    / "configs"
)


# -------------------------------
# Model result output folder
# -------------------------------
# This is written into the JSON config.
# The pipeline will later use this path to save metrics, plots,
# confusion matrices, model objects, predictions, etc.

results_output_base_dir = (
    project_root
    / "Main_project"
    / "results"
    / patient_id
    / "SVM_pilot"
)

# -------------------------------
# Temporal columns and label column
# -------------------------------

time_column = "window_start_time"
label_column = "class_label"


# -------------------------------
# Metadata columns
# -------------------------------
# These columns describe the window but should not be used as model features.
# The pipeline should keep only those that are actually present in the dataframe.

metadata_cols = [
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
]


# -------------------------------
# Label mapping
# -------------------------------
# Original labels:
# 1 = preictal
# 2 = seizure
#
# Binary labels for model:
# 0 = preictal
# 1 = seizure

label_mapping = {
    "1": 0,
    "2": 1
}

class_names = [
    "preictal",
    "seizure"
]

labels_for_evaluation = [
    0,
    1
]


# -------------------------------
# Temporal split parameters
# -------------------------------

temporal_split = {
    "ideal_train": 0.70,
    "ideal_val": 0.15,
    "ideal_test": 0.15,
    "train_search_range": [0.70, 0.90],
    "val_search_range": [0.05, 0.20],
    "ratio_weight": 3,
    "require_both_classes": True
}


# -------------------------------
# SVM model and grid search
# -------------------------------

svm_config = {
    "pipeline_steps": [
        "StandardScaler",
        "SVC"
    ],
    "scaler": {
        "name": "StandardScaler"
    },
    "classifier": {
        "name": "SVC",
        "kernel": "rbf",
        "class_weight": "balanced"
    },
    "param_grid": {
        "svm__C": [0.1, 1, 10, 100],
        "svm__gamma": ["scale", 0.001, 0.01, 0.1, 1]
    },
    "grid_search": {
        "cv_strategy": "TimeSeriesSplit",
        "n_splits": 4,
        "scoring": "f1_macro",
        "n_jobs": -1,
        "verbose": 1,
        "refit": True,
        "return_train_score": True
    }
}


# -------------------------------
# Evaluation options
# -------------------------------

evaluation_config = {
    "evaluate_datasets": [
        "Validation",
        "Test"
    ],
    "metrics": [
        "accuracy",
        "balanced_accuracy",
        "f1_macro",
        "classification_report"
    ],
    "confusion_matrix": {
        "save_counts": True,
        "save_percent": True,
        "save_csv": True,
        "save_pdf": True,
        "normalize": "true_row_percent",
        "show_plot": True
    },
    "save_predictions": True,
    "save_classification_report": True,
    "save_global_metrics": True,
    "save_gridsearch_results": True,
    "save_best_model": True
}


# -------------------------------
# Safety checks
# -------------------------------
# These checks are defined here, but they should be executed
# inside the actual pipeline script.

safety_checks = {
    "check_input_exists": True,
    "check_input_is_pickle": True,
    "check_dataframe_type_is_valid": True,
    "check_time_column_exists": True,
    "check_label_column_exists": True,
    "check_metadata_columns_soft": True,
    "check_no_unmapped_labels": True,
    "check_numeric_features_exist": True,
    "check_no_nan_after_cleaning": True,
    "check_no_infinite_after_cleaning": True,
    "check_non_empty_after_cleaning": True,
    "check_each_split_has_both_classes": True,
    "stop_if_check_fails": True
}


# -------------------------------
# Preprocessing options
# -------------------------------

preprocessing_config = {
    "copy_dataframe": True,
    "convert_time_column_to_datetime": True,
    "sort_by_time_column": True,
    "reset_index_after_sorting": True,
    "drop_all_empty_columns": True,
    "drop_metadata_columns_from_X": True,
    "keep_only_numeric_features": True,
    "replace_inf_with_nan": True,
    "drop_rows_with_nan_features": True,
    "remove_constant_features": True,
    "reset_index_after_cleaning": True
}


# ============================================================
# 2. AUTOMATIC NAMING
# ============================================================

def sanitize_for_filename(text):
    """
    Convert text into a safe filename component.
    """
    text = str(text)
    text = text.replace(" ", "-")
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-")


today_str = datetime.now().strftime("%Y%m%d")

input_stem = sanitize_for_filename(input_pickle_path.stem)
experiment_label_safe = sanitize_for_filename(experiment_label)
dataframe_type_safe = sanitize_for_filename(input_dataframe_type)

experiment_id = (
    f"{patient_id}_"
    f"{experiment_label_safe}_"
    f"DF-{dataframe_type_safe}_"
    f"IN-{input_stem}_"
    f"{today_str}_"
    f"{config_version}"
)

config_filename = f"config_{experiment_id}.json"

config_output_path = output_config_dir / config_filename

experiment_output_dir = results_output_base_dir / experiment_id

eval_output_dir = experiment_output_dir / "evaluation"
model_output_dir = experiment_output_dir / "model"
metrics_output_dir = experiment_output_dir / "metrics"
confusion_matrix_output_dir = experiment_output_dir / "confusion_matrices"
predictions_output_dir = experiment_output_dir / "predictions"
logs_output_dir = experiment_output_dir / "logs"


# ============================================================
# 3. BUILD CONFIG DICTIONARY
# ============================================================

config = {
    "metadata": {
        "config_type": "svm_pilot_config",
        "pipeline_name": "3_SVM_pilot",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "created_by": "json_config_generator",
        "patient_id": patient_id,
        "input_dataframe_type": input_dataframe_type,
        "experiment_label": experiment_label,
        "experiment_id": experiment_id,
        "config_version": config_version
    },

    "project": {
        "project_root": str(project_root),
        "src_dir": str(project_root / "src"),
        "pipeline_dir": str(project_root / "src" / "pipelines" / "3_SVM_pilot")
    },

    "inputs": {
        "input_pickle_path": str(input_pickle_path),
        "input_format": "pickle",
        "input_dataframe_type": input_dataframe_type,
        "time_column": time_column,
        "label_column": label_column
    },

    "outputs": {
        "output_config_dir": str(output_config_dir),
        "config_output_path": str(config_output_path),

        "experiment_output_dir": str(experiment_output_dir),
        "eval_output_dir": str(eval_output_dir),
        "model_output_dir": str(model_output_dir),
        "metrics_output_dir": str(metrics_output_dir),
        "confusion_matrix_output_dir": str(confusion_matrix_output_dir),
        "predictions_output_dir": str(predictions_output_dir),
        "logs_output_dir": str(logs_output_dir),

        "save_config_copy_to_output_dir": True,

        "filenames": {
            "best_model": f"{experiment_id}_best_model.pkl",
            "gridsearch_results": f"{experiment_id}_gridsearch_results.csv",
            "split_summary": f"{experiment_id}_split_summary.csv",
            "cleaning_summary": f"{experiment_id}_cleaning_summary.csv",

            "validation_classification_report": f"{experiment_id}_Validation_classification_report.csv",
            "test_classification_report": f"{experiment_id}_Test_classification_report.csv",

            "validation_global_metrics": f"{experiment_id}_Validation_global_metrics.csv",
            "test_global_metrics": f"{experiment_id}_Test_global_metrics.csv",

            "validation_confusion_matrix_counts_csv": f"{experiment_id}_Validation_confusion_matrix_counts.csv",
            "validation_confusion_matrix_percent_csv": f"{experiment_id}_Validation_confusion_matrix_percent.csv",
            "validation_confusion_matrix_counts_pdf": f"{experiment_id}_Validation_confusion_matrix_counts.pdf",
            "validation_confusion_matrix_percent_pdf": f"{experiment_id}_Validation_confusion_matrix_percent.pdf",

            "test_confusion_matrix_counts_csv": f"{experiment_id}_Test_confusion_matrix_counts.csv",
            "test_confusion_matrix_percent_csv": f"{experiment_id}_Test_confusion_matrix_percent.csv",
            "test_confusion_matrix_counts_pdf": f"{experiment_id}_Test_confusion_matrix_counts.pdf",
            "test_confusion_matrix_percent_pdf": f"{experiment_id}_Test_confusion_matrix_percent.pdf",

            "validation_predictions": f"{experiment_id}_Validation_predictions.csv",
            "test_predictions": f"{experiment_id}_Test_predictions.csv"
        }
    },

    "preprocessing": preprocessing_config,

    "metadata_columns": metadata_cols,

    "labels": {
        "original_label_description": {
            "1": "preictal",
            "2": "seizure"
        },
        "label_mapping": label_mapping,
        "binary_label_description": {
            "0": "preictal",
            "1": "seizure"
        },
        "class_names": class_names,
        "labels_for_evaluation": labels_for_evaluation,
        "positive_class": 1,
        "positive_class_name": "seizure"
    },

    "temporal_split": temporal_split,

    "model": {
        "model_name": "SVM_RBF",
        "model_family": "Support Vector Machine",
        "task": "binary_classification",
        "input_dataframe_type": input_dataframe_type,
        "svm": svm_config
    },

    "evaluation": evaluation_config,

    "safety_checks": safety_checks,

    "notes": {
        "important": (
            "This config can be used with either PCA-transformed dataframes "
            "or feature dataframes. The pipeline should treat all non-metadata "
            "numeric columns as candidate model features."
        ),
        "safety_checks_execution": (
            "Safety checks are only defined in this config. "
            "They must be implemented and executed inside the pipeline script."
        )
    }
}


# ============================================================
# 4. VALIDATE USER INPUTS BEFORE WRITING CONFIG
# ============================================================

valid_dataframe_types = ["pca", "features"]

if input_dataframe_type not in valid_dataframe_types:
    raise ValueError(
        f"Invalid input_dataframe_type: {input_dataframe_type}. "
        f"Expected one of: {valid_dataframe_types}"
    )

if input_pickle_path.suffix != ".pkl":
    raise ValueError(
        f"Input file must be a .pkl file. Got: {input_pickle_path}"
    )

if not str(config_filename).startswith("config_"):
    raise ValueError(
        "Config filename must start with 'config_'."
    )


# ============================================================
# 5. SAVE JSON CONFIG
# ============================================================

output_config_dir.mkdir(parents=True, exist_ok=True)

with open(config_output_path, "w") as f:
    json.dump(config, f, indent=4)

print("JSON config saved successfully:")
print(config_output_path)

print("\nExperiment ID:")
print(experiment_id)

print("\nMain output directory defined in config:")
print(experiment_output_dir)