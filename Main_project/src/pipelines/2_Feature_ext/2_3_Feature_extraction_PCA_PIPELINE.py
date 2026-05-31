import os
import sys
import json
import shutil
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


#==========================
#==========================
#==========================
# 0.1 Load project root and JSON config

current_file = Path(__file__).resolve()

# Go up until you find the project root where "src" exists
for parent in current_file.parents:
    if (parent / "src").exists():
        project_root = parent
        break

# Add to PYTHONPATH if not already there
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


#==========================
#==========================
#==========================
# 0.2 Load JSON config

if len(sys.argv) > 1:
    config_path = Path(sys.argv[1])
else:
    config_path = project_root / "configs" / "config_JYXFE_pca.json"

with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)

print(f"Loaded config from: {config_path.resolve()}")
print(f"Patient ID: {config['user_info']['patient_id']}")
print(f"Experiment name: {config['user_info']['experiment_name']}")
print(f"Version: {config['user_info']['version']}")


#==========================
#==========================
#==========================
# 0.3 Load paths and parameters from config

features_pickle = Path(config["inputs"]["features_pickle"])
input_type = config["inputs"]["input_type"]

output_dir = Path(config["outputs"]["output_dir"])
output_dir.mkdir(parents=True, exist_ok=True)

metadata_cols = config["parameters"]["metadata_cols"]
target_col = config["parameters"]["target_col"]

scaling_apply = config["parameters"]["scaling"]["apply"]
scaling_method = config["parameters"]["scaling"]["method"]

pca_mode = config["parameters"]["pca"]["mode"]
variance_threshold = config["parameters"]["pca"]["explained_variance_threshold"]
fixed_n_components = config["parameters"]["pca"]["fixed_n_components"]

checks = config["parameters"]["checks"]
save_options = config["parameters"]["save"]

print(f"Input pickle: {features_pickle.resolve()}")
print(f"Output directory: {output_dir.resolve()}")


#==========================
#==========================
#==========================
# 0.4 Input checks

if input_type != "pkl":
    raise ValueError(f"Expected input_type='pkl', but got: {input_type}")

if not features_pickle.exists():
    raise FileNotFoundError(f"Input pickle not found: {features_pickle}")


#==========================
#==========================
#==========================
# 1. Import DF with preictal vs ictal data

df_feat_ictalVspreictal = pd.read_pickle(features_pickle)

print(f"Loaded feature dataframe from: {features_pickle.resolve()}")
print("df_feat_ictalVspreictal shape:", df_feat_ictalVspreictal.shape)
print(df_feat_ictalVspreictal.head())


#==========================
#==========================
#==========================
# 2. Separate metadata from features

metadata_cols_available = [
    col for col in metadata_cols
    if col in df_feat_ictalVspreictal.columns
]

metadata_cols_missing = [
    col for col in metadata_cols
    if col not in df_feat_ictalVspreictal.columns
]

if len(metadata_cols_missing) > 0:
    if checks["ignore_missing_metadata_cols"]:
        print("Ignored missing metadata columns:")
        print(metadata_cols_missing)
    else:
        raise ValueError(f"Missing metadata columns: {metadata_cols_missing}")

print("Using metadata columns:")
print(metadata_cols_available)


# Check target column
if target_col not in df_feat_ictalVspreictal.columns:
    if checks["stop_if_target_missing"]:
        raise ValueError(f"Target column not found: {target_col}")
    else:
        print(f"Warning: target column not found: {target_col}")


# Feature columns = everything except metadata
# Important: target_col is already inside metadata_cols in your config,
# but this makes the exclusion safer.
cols_to_exclude = metadata_cols_available.copy()

if target_col not in cols_to_exclude:
    cols_to_exclude.append(target_col)

feature_cols = [
    col for col in df_feat_ictalVspreictal.columns
    if col not in cols_to_exclude
]

print("Number of feature columns:", len(feature_cols))
print(feature_cols)


#==========================
#==========================
#==========================
# 3. Build matrix

X = df_feat_ictalVspreictal[feature_cols].copy()

if target_col in df_feat_ictalVspreictal.columns:
    y = df_feat_ictalVspreictal[target_col].copy()
    print("y shape:", y.shape)

print("X shape:", X.shape)


#==========================
#==========================
#==========================
# 4. Check NaN or Inf

non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

if len(non_numeric_cols) > 0:
    raise ValueError(
        "Non-numeric feature columns found. These cannot be used for PCA:\n"
        f"{non_numeric_cols}"
    )

print("NaNs per column:")
print(X.isna().sum())

total_nan = X.isna().sum().sum()

print("Total NaN values:", total_nan)

if total_nan > 0 and checks["stop_if_nan"]:
    raise ValueError("NaN values found in feature matrix. PCA stopped.")

print("Any infinite values?")
total_inf = np.isinf(X.to_numpy()).sum()
print(total_inf)

if total_inf > 0 and checks["stop_if_inf"]:
    raise ValueError("Infinite values found in feature matrix. PCA stopped.")


#==========================
#==========================
#==========================
# 5. Scale features before PCA

if scaling_apply:

    if scaling_method != "StandardScaler":
        raise ValueError(f"Unsupported scaling method: {scaling_method}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Scaling applied: StandardScaler")

else:
    X_scaled = X.to_numpy()

    print("Scaling not applied")

print("X_scaled shape:", X_scaled.shape)


#==========================
#==========================
#==========================
# 6. Apply PCA with all possible components

pca_full = PCA()
X_pca_full = pca_full.fit_transform(X_scaled)

explained_var = pca_full.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var)

for i, (var, cum_var) in enumerate(zip(explained_var, cumulative_var), start=1):
    print(f"PC{i}: {var*100:.2f}% | cumulative: {cum_var*100:.2f}%")


#==========================
#==========================
#==========================
# 7. Select number of PCA components from config

if pca_mode == "variance_threshold":

    n_components = np.argmax(cumulative_var >= variance_threshold) + 1

    print("PCA mode: variance_threshold")
    print(f"Explained variance threshold: {variance_threshold}")
    print(f"Selected n_components: {n_components}")

elif pca_mode == "fixed_n_components":

    n_components = fixed_n_components

    if n_components is None:
        raise ValueError(
            "fixed_n_components cannot be None when pca.mode is fixed_n_components."
        )

    print("PCA mode: fixed_n_components")
    print(f"Selected n_components: {n_components}")

else:
    raise ValueError(
        "Invalid PCA mode. Use 'variance_threshold' or 'fixed_n_components'."
    )


#==========================
#==========================
#==========================
# 8. Apply final PCA

pca = PCA(n_components=n_components)

X_pca = pca.fit_transform(X_scaled)

print("X_pca shape:", X_pca.shape)

explained_variance = pca.explained_variance_ratio_

for i, var in enumerate(explained_variance, start=1):
    print(f"PC{i}: {var:.4f} ({var*100:.2f}%)")

print("Total explained variance:", explained_variance.sum())


#==========================
#==========================
#==========================
# 9. Explained variance dataframe

df_explained_variance = pd.DataFrame({
    "PC": [f"PC{i+1}" for i in range(len(explained_var))],
    "explained_variance_ratio": explained_var,
    "explained_variance_percent": explained_var * 100,
    "cumulative_variance_ratio": cumulative_var,
    "cumulative_variance_percent": cumulative_var * 100
})

print("df_explained_variance shape:", df_explained_variance.shape)
print(df_explained_variance.head())


#==========================
#==========================
#==========================
# 10. Final dataframe: windows x PCA

pca_cols = [f"PC{i+1}" for i in range(X_pca.shape[1])]

df_pca = pd.DataFrame(
    X_pca,
    columns=pca_cols,
    index=df_feat_ictalVspreictal.index
)

print("df_pca shape:", df_pca.shape)
print(df_pca.head())


#==========================
#==========================
#==========================
# 11. Add metadata again

df_windows_pca = pd.concat(
    [
        df_feat_ictalVspreictal[metadata_cols_available].reset_index(drop=True),
        df_pca.reset_index(drop=True)
    ],
    axis=1
)

print("df_windows_pca shape:", df_windows_pca.shape)
print(df_windows_pca.head())


#==========================
#==========================
#==========================
# 12. Sanity check

same_rows = df_feat_ictalVspreictal.shape[0] == df_windows_pca.shape[0]

print("Same number of rows as original df:", same_rows)

if not same_rows:
    raise ValueError("Row mismatch between original dataframe and PCA dataframe.")


#==========================
#==========================
#==========================
# 13. Export outputs from config

if save_options["pca_dataframe_pickle"]:

    output_path = Path(config["outputs"]["pca_dataframe_pickle"])

    df_windows_pca.to_pickle(output_path)

    print(f"Saved PCA dataframe to: {output_path.resolve()}")
    print("Shape:", df_windows_pca.shape)


if save_options["explained_variance_csv"]:

    explained_variance_path = Path(config["outputs"]["explained_variance_csv"])

    df_explained_variance.to_csv(explained_variance_path, index=False)

    print(f"Saved explained variance CSV to: {explained_variance_path.resolve()}")


if save_options["config_copy_json"]:

    config_copy_path = Path(config["outputs"]["config_copy_json"])

    shutil.copy2(config_path, config_copy_path)

    print(f"Saved config copy to: {config_copy_path.resolve()}")


print("PCA completed successfully.")