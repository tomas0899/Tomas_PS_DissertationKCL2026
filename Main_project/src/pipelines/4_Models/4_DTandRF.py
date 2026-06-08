# ============================================================
# 0. IMPORT LIBRARIES
# ============================================================

from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd


# ===============================
# 1 Load project modules
# ===============================

current_file = Path(__file__).resolve()

project_root = None

for parent in current_file.parents:
    if (parent / "src").exists():
        project_root = parent
        break

if project_root is None:
    raise RuntimeError(
        "Project root not found. Could not find a parent folder containing 'src'."
    )

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.modules import tools_EEG_models as TEEG_mod

# ============================================================
# 2. LOAD JSON CONFIG
# ============================================================
# Usage from terminal:
#
# python modelDT_RF_pipeline.py path/to/config.json
#
# If no config path is provided, edit the default path below.
# ============================================================

if len(sys.argv) > 1:
    config_path = Path(sys.argv[1])
else:
    config_path = (
        project_root
        / "src"
        / "pipelines"
        / "4_Models"
        / "configs"
        / "XB47Y"
        / "config_XB47Y_IN-PCA_DT-RF_SCORING-F1-MACRO_20260608_v01.json"
    )

with open(config_path, "r") as f:
    config = json.load(f)

print("Loaded config:")
print(config_path)


# ============================================================
# 3. READ CONFIG PARAMETERS
# ============================================================

patient_id = config["experiment_info"]["patient_id"]
random_state = config["experiment_info"]["random_state"]
input_path = Path(config["inputs"]["input_path"])

main_eval_output_dir = Path(config["outputs"]["main_eval_output_dir"])
eval_output_dir_dt = Path(config["outputs"]["eval_output_dir_dt"])
eval_output_dir_rf = Path(config["outputs"]["eval_output_dir_rf"])

main_eval_output_dir.mkdir(parents=True, exist_ok=True)
eval_output_dir_dt.mkdir(parents=True, exist_ok=True)
eval_output_dir_rf.mkdir(parents=True, exist_ok=True)

time_column = config["data_processing"]["time_column"]
target_column = config["data_processing"]["target_column"]

metadata_cols = config["data_processing"]["metadata_cols"]

# JSON stores dictionary keys as strings.
# Convert label mapping keys back to integers.
label_mapping = {
    int(original_label): binary_label
    for original_label, binary_label in config["data_processing"]["label_mapping"].items()
}

class_names = config["data_processing"]["class_names"]
labels = config["data_processing"]["labels"]

ideal_train = config["temporal_split"]["ideal_train"]
ideal_val = config["temporal_split"]["ideal_val"]
ideal_test = config["temporal_split"]["ideal_test"]


train_search_range = tuple(config["temporal_split"]["train_search_range"])
val_search_range = tuple(config["temporal_split"]["val_search_range"])
ratio_weight = config["temporal_split"]["ratio_weight"]

n_splits = config["grid_search"]["n_splits"]
scoring = config["grid_search"]["scoring"]
n_jobs = config["grid_search"]["n_jobs"]
verbose = config["grid_search"]["verbose"]

show_plot = config["evaluation"]["show_plot"]


# ============================================================
# 4. LOAD DATAFRAME
# ============================================================

df_windows = pd.read_pickle(input_path)

print("\nLoaded dataframe:")
print("Input path:", input_path)
print("Shape:", df_windows.shape)


# ============================================================
# 5. BASIC QUALITY CONTROL AND TEMPORAL SORTING
# ============================================================

df_mod = df_windows.copy()

df_mod[time_column] = pd.to_datetime(df_mod[time_column])

df_mod = (
    df_mod
    .sort_values(time_column)
    .reset_index(drop=True)
)

print("\nFirst window times:")
print(df_mod[time_column].head())

print("\nLast window times:")
print(df_mod[time_column].tail())

# Remove columns that are completely empty.
df_mod = df_mod.dropna(axis=1, how="all").copy()


# ============================================================
# 6. DEFINE METADATA COLUMNS
# ============================================================

metadata_cols = [
    col for col in metadata_cols
    if col in df_mod.columns
]


# ============================================================
# 7. CREATE INITIAL FEATURE MATRIX X AND TARGET y
# ============================================================
y = df_mod[target_column].copy()

columns_to_drop = list(set(metadata_cols + [target_column, time_column]))

X = df_mod.drop(columns=columns_to_drop, errors="ignore").copy()

print("\nInitial feature/target shapes:")
print("df_mod shape:", df_mod.shape)
print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nInitial class counts:")
print(y.value_counts())


# ============================================================
# 8. KEEP ONLY NUMERIC FEATURES
# ============================================================

X = X.select_dtypes(include=[np.number]).copy()

print("\nX shape after keeping numeric columns only:")
print(X.shape)


# ============================================================
# 9. HANDLE INFINITE AND MISSING VALUES
# ============================================================

X = X.replace([np.inf, -np.inf], np.nan)

mask = X.notna().all(axis=1)

X = X.loc[mask].copy()
y = y.loc[mask].copy()

df_model = df_mod.loc[mask].copy()

print("\nAfter removing rows with missing feature values:")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("df_model shape:", df_model.shape)
print("Total NaNs in X:", X.isna().sum().sum())


# ============================================================
# 10. REMOVE CONSTANT FEATURE COLUMNS
# ============================================================

constant_cols = [
    col for col in X.columns
    if X[col].nunique() <= 1
]

X = X.drop(columns=constant_cols)

print("\nRemoved constant columns:")
print(constant_cols)

print("\nX shape after removing constant columns:")
print(X.shape)


# ============================================================
# 11. CONVERT LABELS TO BINARY FORMAT
# ============================================================

y_binary = y.map(label_mapping)

# This is kept because otherwise the model could silently train
# with wrong labels if the config does not match the dataframe.
if y_binary.isna().any():
    unexpected_labels = y[y_binary.isna()].unique()
    raise ValueError(f"Unexpected labels found: {unexpected_labels}")

y = y_binary.astype(int).copy()

print("\nBinary class counts:")
print(y.value_counts())

global_ratio = y.mean()

print("\nGlobal seizure ratio:")
print(global_ratio)


# ============================================================
# 12. RESET INDEX AFTER CLEANING
# ============================================================

X = X.reset_index(drop=True)
y = y.reset_index(drop=True)
df_model = df_model.reset_index(drop=True)

n = len(X)


# ============================================================
# 13. FIND BEST TEMPORAL TRAIN / VALIDATION / TEST SPLIT
# ============================================================

train_end, val_end, best_score = TEEG_mod.find_best_temporal_split_3_2(
    y=y,
    ideal_train=ideal_train,
    ideal_val=ideal_val,
    ideal_test=ideal_test,
    train_search_range=train_search_range,
    val_search_range=val_search_range,
    ratio_weight=ratio_weight
)

print("\nBest temporal split found:")
print("train_end:", train_end)
print("val_end:", val_end)
print("best_score:", best_score)


# ============================================================
# 14. CREATE FINAL TEMPORAL SPLITS
# ============================================================

X_train = X.iloc[:train_end].copy()
y_train = y.iloc[:train_end].copy()

X_val = X.iloc[train_end:val_end].copy()
y_val = y.iloc[train_end:val_end].copy()

X_test = X.iloc[val_end:].copy()
y_test = y.iloc[val_end:].copy()


# ============================================================
# 15. PRINT SPLIT SUMMARY
# ============================================================

print("\nSet sizes:")
print(f"Train: {len(X_train)} ({len(X_train) / n:.3%})")
print(f"Val:   {len(X_val)} ({len(X_val) / n:.3%})")
print(f"Test:  {len(X_test)} ({len(X_test) / n:.3%})")

print("\nSeizure ratios:")
print(f"Global: {y.mean():.5f}")
print(f"Train:  {y_train.mean():.5f}")
print(f"Val:    {y_val.mean():.5f}")
print(f"Test:   {y_test.mean():.5f}")

print("\nTime ranges:")
print(
    "Train:",
    df_model.loc[0, time_column],
    "->",
    df_model.loc[train_end - 1, time_column]
)

print(
    "Val:  ",
    df_model.loc[train_end, time_column],
    "->",
    df_model.loc[val_end - 1, time_column]
)

print(
    "Test: ",
    df_model.loc[val_end, time_column],
    "->",
    df_model.loc[n - 1, time_column]
)

# ============================================================
# 16. DECISION TREE TRAINING
# ============================================================

grid_dt, best_model_dt, best_params_dt, best_score_dt = TEEG_mod.train_decision_tree_gridsearch_3_4(
    X_train,
    y_train
)
# ============================================================
# 17. DECISION TREE EVALUATION
# ============================================================

val_results = TEEG_mod.evaluate_and_plot_3_1(
    model=best_model_dt,
    X_data=X_val,
    y_true=y_val,
    class_names=class_names,
    dataset_name="Validation - Decision Tree",
    patient_id=patient_id,
    output_dir=eval_output_dir_dt,
    labels=labels,
    show_plot=show_plot
)

test_results = TEEG_mod.evaluate_and_plot_3_1(
    model=best_model_dt,
    X_data=X_test,
    y_true=y_test,
    class_names=class_names,
    dataset_name="Test - Decision Tree",
    patient_id=patient_id,
    output_dir=eval_output_dir_dt,
    labels=labels,
    show_plot=show_plot
)

# ============================================================
# 18. RANDOM FOREST TRAINING
# ============================================================


grid_rf, best_model_rf, best_params_rf, best_score_rf = TEEG_mod.train_random_forest_gridsearch_3_5(
    X_train,
    y_train
)
# ============================================================
# 19.RANDOM FOREST TRAINING
# ============================================================


val_results = TEEG_mod.evaluate_and_plot_3_1(
    model=best_model_rf,
    X_data=X_val,
    y_true=y_val,
    class_names=class_names,
    dataset_name="Validation - Random Forest",
    patient_id=patient_id,
    output_dir=eval_output_dir_rf,
    labels=labels,
    show_plot=show_plot
)

test_results = TEEG_mod.evaluate_and_plot_3_1(
    model=best_model_rf,
    X_data=X_test,
    y_true=y_test,
    class_names=class_names,
    dataset_name="Test - Random Forest",
    patient_id=patient_id,
    output_dir=eval_output_dir_rf,
    labels=labels,
    show_plot=show_plot
)
