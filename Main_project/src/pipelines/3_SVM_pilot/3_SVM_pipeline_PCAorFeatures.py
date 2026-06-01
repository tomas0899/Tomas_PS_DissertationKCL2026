# ============================================================
# 0. IMPORT LIBRARIES
# ============================================================

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    f1_score
)


# ============================================================
# 1. LOAD PROJECT MODULES
# ============================================================
# This block automatically finds the project root folder.
# It searches upwards from the current script location until it finds
# a parent folder containing "src".
#
# This allows importing custom modules from:
# project_root/src/modules/

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

# Add project root to sys.path so Python can import modules from src/
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.modules import tools_EEG_models as TEEG_mod


# ============================================================
# 2. DEFINE INPUTS
# ============================================================
# Patient ID is useful for plots, tables, and output tracking.

patient_id = "XB47Y"

# Input dataframe containing PCA-transformed EEG window features.
# Each row should correspond to one EEG window.

input_path = Path(
    "/home/tperezsanchez/FoundationModel_EEG_Dissertation/Main_project/results/XB47Y/Feature_ext/Part2_features/XB47Y_IN-normalized_npz_FP-fullnpz_W10s_PRE6to5min_ICT0to1min_GAPasINT_FINAL-PREvsSEIZ_20260504_v01_FEAT-TIME-FREQ_20260505_v01/df_windowsXB47Y_pca.pkl"
)


# ============================================================
# 3. LOAD DATAFRAME
# ============================================================
# Load the dataframe from pickle.

df_windows_pca = pd.read_pickle(input_path)

print("Loaded dataframe:")
print("Shape:", df_windows_pca.shape)

# Optional preview if running interactively
df_windows_pca.head()


# ============================================================
# 4. BASIC QUALITY CONTROL AND TEMPORAL SORTING
# ============================================================
# Make a copy for the SVM pipeline.
# Convert window_start_time to datetime format.
# Sort rows chronologically to preserve temporal order.

df_SVM = df_windows_pca.copy()

df_SVM["window_start_time"] = pd.to_datetime(df_SVM["window_start_time"])

df_SVM = (
    df_SVM
    .sort_values("window_start_time")
    .reset_index(drop=True)
)

print("\nFirst window times:")
print(df_SVM["window_start_time"].head())

print("\nLast window times:")
print(df_SVM["window_start_time"].tail())

# Remove columns that are completely empty.
# Example: excluded_reason may be completely empty.
df_SVM = df_SVM.dropna(axis=1, how="all").copy()


# ============================================================
# 5. DEFINE METADATA COLUMNS
# ============================================================
# These columns describe the window but should not be used as model features.
# They include IDs, timestamps, target labels, file names, and recording metadata.

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

# Keep only metadata columns that are actually present in the dataframe.
metadata_cols = [col for col in metadata_cols if col in df_SVM.columns]


# ============================================================
# 6. CREATE INITIAL FEATURE MATRIX X AND TARGET y
# ============================================================
# y is the target variable.
# X contains only candidate feature columns.

y = df_SVM["class_label"].copy()

X = df_SVM.drop(columns=metadata_cols).copy()

print("\nInitial feature/target shapes:")
print("df_SVM shape:", df_SVM.shape)
print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nInitial class counts:")
print(y.value_counts())


# ============================================================
# 7. KEEP ONLY NUMERIC FEATURES
# ============================================================
# SVM requires numeric input.
# This removes any remaining non-numeric columns.

X = X.select_dtypes(include=[np.number]).copy()

print("\nX shape after keeping numeric columns only:")
print(X.shape)


# ============================================================
# 8. HANDLE INFINITE AND MISSING VALUES
# ============================================================
# Replace positive and negative infinity with NaN.
# Then remove rows where any EEG feature contains NaN.

X = X.replace([np.inf, -np.inf], np.nan)

# Keep only rows with complete feature values.
mask = X.notna().all(axis=1)

X = X.loc[mask].copy()
y = y.loc[mask].copy()

# Keep an aligned copy of the metadata dataframe.
# This is important for correctly reporting time ranges after filtering rows.
df_model = df_SVM.loc[mask].copy()

print("\nAfter removing rows with missing feature values:")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("df_model shape:", df_model.shape)
print("Total NaNs in X:", X.isna().sum().sum())


# ============================================================
# 9. REMOVE CONSTANT FEATURE COLUMNS
# ============================================================
# Constant columns do not help classification because they do not vary
# between windows.

constant_cols = [col for col in X.columns if X[col].nunique() <= 1]

X = X.drop(columns=constant_cols)

print("\nRemoved constant columns:")
print(constant_cols)

print("\nX shape after removing constant columns:")
print(X.shape)


# ============================================================
# 10. CONVERT LABELS TO BINARY FORMAT
# ============================================================
# Original labels:
# 1 = preictal
# 2 = seizure
#
# Binary labels for model:
# 0 = preictal
# 1 = seizure

label_mapping = {
    1: 0,
    2: 1
}

y_binary = y.map(label_mapping)

# Safety check: this catches labels that were not included in the mapping.
if y_binary.isna().any():
    unexpected_labels = y[y_binary.isna()].unique()
    raise ValueError(f"Unexpected labels found: {unexpected_labels}")

y = y_binary.astype(int).copy()

print("\nBinary class counts:")
print(y.value_counts())

# y.mean() corresponds to the proportion of class 1.
# In this binary encoding, class 1 = seizure.
global_ratio = y.mean()

print("\nGlobal seizure ratio:")
print(global_ratio)


# ============================================================
# 11. RESET INDEX AFTER CLEANING
# ============================================================
# Reset indexes so X, y, and df_model are aligned from 0 to n-1.

X = X.reset_index(drop=True)
y = y.reset_index(drop=True)
df_model = df_model.reset_index(drop=True)

n = len(X)

print("\nFinal cleaned dataset size:")
print("n =", n)


# ============================================================
# 12. FIND BEST TEMPORAL TRAIN / VALIDATION / TEST SPLIT
# ============================================================
# The goal is to preserve chronological order while finding a split close to:
# Train = 70%
# Validation = 15%
# Test = 15%
#
# The code also tries to keep the seizure ratio similar across splits.
# Each split must contain both classes.


train_end, val_end, best_score = find_best_temporal_split_3_2(
    y=y,
    ideal_train=0.70,
    ideal_val=0.15,
    ideal_test=0.15,
    train_search_range=(0.70, 0.90),
    val_search_range=(0.05, 0.20),
    ratio_weight=3
)

print("Best temporal split found:")
print("train_end:", train_end)
print("val_end:", val_end)
print("best_score:", best_score)
# ============================================================
# 13. CREATE FINAL TEMPORAL SPLITS
# ============================================================
# Train: beginning of the recording up to train_end.
# Validation: train_end to val_end.
# Test: val_end to the end of the recording.

X_train = X.iloc[:train_end].copy()
y_train = y.iloc[:train_end].copy()

X_val = X.iloc[train_end:val_end].copy()
y_val = y.iloc[train_end:val_end].copy()

X_test = X.iloc[val_end:].copy()
y_test = y.iloc[val_end:].copy()


# ============================================================
# 14. PRINT SPLIT SUMMARY
# ============================================================

print("\nSet sizes:")
print(f"Train: {len(X_train)} ({len(X_train) / n:.3%})")
print(f"Val:   {len(X_val)} ({len(X_val) / n:.3%})")
print(f"Test:  {len(X_test) / n:.3%}")

print("\nSeizure ratios:")
print(f"Global: {y.mean():.5f}")
print(f"Train:  {y_train.mean():.5f}")
print(f"Val:    {y_val.mean():.5f}")
print(f"Test:   {y_test.mean():.5f}")

print("\nTime ranges:")
print(
    "Train:",
    df_model.loc[0, "window_start_time"],
    "->",
    df_model.loc[train_end - 1, "window_start_time"]
)

print(
    "Val:  ",
    df_model.loc[train_end, "window_start_time"],
    "->",
    df_model.loc[val_end - 1, "window_start_time"]
)

print(
    "Test: ",
    df_model.loc[val_end, "window_start_time"],
    "->",
    df_model.loc[n - 1, "window_start_time"]
)


# ============================================================
# 15. TRAIN SVM MODEL WITH TEMPORAL GRID SEARCH
# ============================================================

best_model_f1, grid_f1 = train_svm_gridsearch(
    X_train=X_train,
    y_train=y_train,
    n_splits=4,
    scoring="f1_macro",
    n_jobs=-1,
    verbose=1
)
# ============================================================
# 16. EVALUATE MODEL
# ============================================================
# Class names must match the binary label encoding:
# 0 = preictal
# 1 = seizure

class_names = ["preictal", "seizure"]

# NOTE:
# This assumes that evaluate_and_plot() was defined previously.
# If you added patient_id to that function, pass it here.

val_results = evaluate_and_plot_3_1(
    model=best_model_f1,
    X_data=X_val,
    y_true=y_val,
    class_names=class_names,
    dataset_name="Validation",
    patient_id=patient_id,
    output_dir=eval_output_dir,
    labels=[0, 1],
    show_plot=True
)


test_results = evaluate_and_plot_3_1(
    model=best_model_f1,
    X_data=X_test,
    y_true=y_test,
    class_names=class_names,
    dataset_name="Test",
    patient_id=patient_id,
    output_dir=eval_output_dir,
    labels=[0, 1],
    show_plot=True
)