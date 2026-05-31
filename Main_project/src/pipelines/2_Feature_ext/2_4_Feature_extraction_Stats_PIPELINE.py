import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ===============================
# 0.1 Load modules
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

from src.modules import tools_EEG_FE as TEEG_FE


# ===============================
# 0.2 Load JSON config
# ===============================

if len(sys.argv) > 1:
    config_path = Path(sys.argv[1])
else:
    config_path = (
        project_root
        / "configs"
        / "Feature_ext"
        / "Part3_Feat_stats"
        / "config_XB47Y_FEAT-STATS_20260531_v01.json"
    )

if not config_path.exists():
    raise FileNotFoundError(f"Config file not found: {config_path}")

with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)

print(f"Loaded config from: {config_path.resolve()}")
print(f"Patient ID: {config['config_metadata']['patient_id']}")
print(f"Experiment ID: {config['config_metadata']['experiment_id']}")
print(f"Version: {config['config_metadata']['version']}")


# ===============================
# 0.3 Read config values
# ===============================

input_pkl_path = Path(config["inputs"]["input_pkl_path"])

output_dir = Path(config["outputs"]["output_dir"])

violin_pdf_path = Path(config["outputs"]["violin_pdf_path"])
top20_pdf_path = Path(config["outputs"]["top20_pdf_path"])
top20_by_channel_pdf_path = Path(config["outputs"]["top20_by_channel_pdf_path"])

mannwhitney_csv_path = Path(config["outputs"]["mannwhitney_csv_path"])
top20_csv_path = Path(config["outputs"]["top20_csv_path"])
top20_by_channel_csv_path = Path(config["outputs"]["top20_by_channel_csv_path"])

# Optional output, depending on whether it exists in the config
ranked_features_csv_path = config["outputs"].get("ranked_features_csv_path", None)

if ranked_features_csv_path is not None:
    ranked_features_csv_path = Path(ranked_features_csv_path)

class_label_column = config["label_settings"]["class_label_column"]

preictal_label = config["label_settings"]["class_labels"]["preictal"]
seizure_label = config["label_settings"]["class_labels"]["seizure"]

exclude_cols = config["feature_selection"]["exclude_cols"]

alpha = config["statistics"]["alpha"]

top_n = config["plot_settings"]["top_n"]
show_plots = config["plot_settings"]["show_plots"]

channel_patterns = config["plot_settings"]["channel_patterns"]


# ===============================
# 0.4 Validate input and output paths
# ===============================

if not input_pkl_path.exists():
    raise FileNotFoundError(
        f"Input pickle file not found:\n{input_pkl_path}"
    )

output_dir.mkdir(parents=True, exist_ok=True)

for output_path in [
    violin_pdf_path,
    top20_pdf_path,
    top20_by_channel_pdf_path,
    mannwhitney_csv_path,
    top20_csv_path,
    top20_by_channel_csv_path,
]:
    output_path.parent.mkdir(parents=True, exist_ok=True)

if ranked_features_csv_path is not None:
    ranked_features_csv_path.parent.mkdir(parents=True, exist_ok=True)


# ===============================
# 1. Load dataframe
# ===============================

df_feat_ictalVspreictal = pd.read_pickle(input_pkl_path)

print("\nInput dataframe loaded:")
print(input_pkl_path)
print("Dataframe shape:", df_feat_ictalVspreictal.shape)
print(df_feat_ictalVspreictal.head())


# ===============================
# 2. Divide into 2 groups
# ===============================
# Label mapping from config:
# preictal = 1
# seizure  = 2

if class_label_column not in df_feat_ictalVspreictal.columns:
    raise ValueError(
        f"Class label column '{class_label_column}' not found in dataframe."
    )

group_1_PREICTAL = df_feat_ictalVspreictal[
    df_feat_ictalVspreictal[class_label_column] == preictal_label
].copy()

group_2_SEIZURE = df_feat_ictalVspreictal[
    df_feat_ictalVspreictal[class_label_column] == seizure_label
].copy()

print("\nGroup sizes:")
print("Shape of group 1 preictal:", group_1_PREICTAL.shape)
print("Shape of group 2 seizure:", group_2_SEIZURE.shape)

if group_1_PREICTAL.empty:
    raise ValueError(
        f"No preictal rows found using {class_label_column} == {preictal_label}"
    )

if group_2_SEIZURE.empty:
    raise ValueError(
        f"No seizure rows found using {class_label_column} == {seizure_label}"
    )


# ===============================
# 3. Select only numeric feature columns
# ===============================

numeric_cols = df_feat_ictalVspreictal.select_dtypes(
    include=[np.number]
).columns.tolist()

print("\nNumeric columns:")
print(numeric_cols)
print("Number of numeric columns:", len(numeric_cols))

feature_cols = [
    col for col in numeric_cols
    if col not in exclude_cols
]

print("\nFeature columns:")
print(feature_cols)
print("Number of feature columns:", len(feature_cols))

if len(feature_cols) == 0:
    raise ValueError(
        "No feature columns found after excluding metadata columns."
    )


# ===============================
# 4. Violin plots + Mann-Whitney test
# ===============================

df_mannwhitney_results_violin = TEEG_FE.plot_mannwhitney_feature_violins_2_8(
    feature_cols=feature_cols,
    group_1_PREICTAL=group_1_PREICTAL,
    group_2_SEIZURE=group_2_SEIZURE,
    pdf_output_path=violin_pdf_path,
    alpha=alpha,
    show_plots=show_plots
)

df_mannwhitney_results_violin.to_csv(
    mannwhitney_csv_path,
    index=False
)

print("\nMann-Whitney results saved to:")
print(mannwhitney_csv_path)


# ===============================
# 5. Top N Mann-Whitney barplot
# ===============================

df_top_mannwhitney_features = TEEG_FE.plot_top_mannwhitney_features_2_9(
    df_mannwhitney_results=df_mannwhitney_results_violin,
    top_n=top_n,
    pdf_output_path=top20_pdf_path,
    show_plot=show_plots
)

df_top_mannwhitney_features.to_csv(
    top20_csv_path,
    index=False
)

print(f"\nTop {top_n} features saved to:")
print(top20_csv_path)


# ===============================
# 6. Top N by channel
# ===============================

df_top_by_channel, df_ranked = TEEG_FE.plot_top_features_by_channel_2_10(
    df_mannwhitney_results=df_mannwhitney_results_violin,
    top_n=top_n,
    channel_patterns=channel_patterns,
    pdf_output_path=top20_by_channel_pdf_path,
    show_plot=show_plots
)

df_top_by_channel.to_csv(
    top20_by_channel_csv_path,
    index=False
)

print(f"\nTop {top_n} by channel saved to:")
print(top20_by_channel_csv_path)

if ranked_features_csv_path is not None:
    df_ranked.to_csv(
        ranked_features_csv_path,
        index=False
    )

    print("\nRanked features by channel saved to:")
    print(ranked_features_csv_path)


# ===============================
# 7. Final summary
# ===============================

print("\nFeature statistics pipeline completed successfully.")

print("\nGenerated files:")
print("Violin PDF:", violin_pdf_path)
print("Top features PDF:", top20_pdf_path)
print("Top features by channel PDF:", top20_by_channel_pdf_path)
print("Mann-Whitney CSV:", mannwhitney_csv_path)
print("Top features CSV:", top20_csv_path)
print("Top features by channel CSV:", top20_by_channel_csv_path)

if ranked_features_csv_path is not None:
    print("Ranked features CSV:", ranked_features_csv_path)