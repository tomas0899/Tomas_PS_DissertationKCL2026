import sys
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
    raise RuntimeError("Project root not found. Could not find a parent folder containing 'src'.")

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.modules import tools_EEG_FE as TEEG_FE


# ===============================
# 1. Load dataframe
# ===============================

input_pkl_path = Path(
    "/home/tperezsanchez/FoundationModel_EEG_Dissertation/Main_project/results/XB47Y/Feature_ext/Part2_features/XB47Y_IN-normalized_npz_FP-fullnpz_W10s_PRE6to5min_ICT0to1min_GAPasINT_FINAL-PREvsSEIZ_20260504_v01_FEAT-TIME-FREQ_20260505_v01/XB47Y_IN-normalized_npz_FP-fullnpz_W10s_PRE6to5min_ICT0to1min_GAPasINT_FINAL-PREvsSEIZ_20260504_v01_FEAT-TIME-FREQ_20260505_v01_df_features_ictalVspreictal.pkl"
)

df_feat_ictalVspreictal_5min = pd.read_pickle(input_pkl_path)

print(df_feat_ictalVspreictal_5min.head())


# ===============================
# 2. Divide into 2 groups
# ===============================
# label mapping:
# preictal = 1
# seizure  = 2

group_1_PREICTAL = df_feat_ictalVspreictal_5min[
    df_feat_ictalVspreictal_5min["class_label"] == 1
].copy()

group_2_SEIZURE = df_feat_ictalVspreictal_5min[
    df_feat_ictalVspreictal_5min["class_label"] == 2
].copy()

print("Shape of group 1 preictal:", group_1_PREICTAL.shape)
print("Shape of group 2 seizure:", group_2_SEIZURE.shape)


# ===============================
# 3. Select only numeric feature columns
# ===============================

numeric_cols = df_feat_ictalVspreictal_5min.select_dtypes(
    include=[np.number]
).columns.tolist()

print("Numeric columns:")
print(numeric_cols)
print("Number of numeric columns:", len(numeric_cols))

exclude_cols = [
    "window_id",
    "start_sample",
    "end_sample",
    "fs",
    "n_channels",
    "class_label",
    "window_sec"
]

feature_cols = [col for col in numeric_cols if col not in exclude_cols]

print("Feature columns:")
print(feature_cols)
print("Number of feature columns:", len(feature_cols))


# ===============================
# 4. Define output paths
# ===============================

output_dir = Path(
    "/home/tperezsanchez/FoundationModel_EEG_Dissertation/Main_project/results/XB47Y/Feature_ext/Part3_Feat_stats"
)

output_dir.mkdir(parents=True, exist_ok=True)

violin_pdf_path = output_dir / "mannwhitney_violin_plots.pdf"
top20_pdf_path = output_dir / "top20_mannwhitney_features.pdf"
top20_by_channel_pdf_path = output_dir / "top20_mannwhitney_features_by_channel.pdf"

mannwhitney_csv_path = output_dir / "mannwhitney_results_violin.csv"
top20_csv_path = output_dir / "top20_mannwhitney_features.csv"
top20_by_channel_csv_path = output_dir / "top20_mannwhitney_features_by_channel.csv"


# ===============================
# 5. Violin plots + Mann-Whitney test
# ===============================

df_mannwhitney_results_violin = TEEG_FE.plot_mannwhitney_feature_violins_2_8(
    feature_cols=feature_cols,
    group_1_PREICTAL=group_1_PREICTAL,
    group_2_SEIZURE=group_2_SEIZURE,
    pdf_output_path=violin_pdf_path,
    alpha=0.05,
    show_plots=False
)

df_mannwhitney_results_violin.to_csv(mannwhitney_csv_path, index=False)

print("Mann-Whitney results saved to:", mannwhitney_csv_path)


# ===============================
# 6. Top 20 Mann-Whitney barplot
# ===============================

df_top_mannwhitney_features = TEEG_FE.plot_top_mannwhitney_features_2_9(
    df_mannwhitney_results=df_mannwhitney_results_violin,
    top_n=20,
    pdf_output_path=top20_pdf_path,
    show_plot=False
)

df_top_mannwhitney_features.to_csv(top20_csv_path, index=False)

print("Top 20 features saved to:", top20_csv_path)


# ===============================
# 7. Top 20 by channel
# ===============================

df_top_by_channel, df_ranked = TEEG_FE.plot_top_features_by_channel_2_10(
    df_mannwhitney_results=df_mannwhitney_results_violin,
    top_n=20,
    pdf_output_path=top20_by_channel_pdf_path,
    show_plot=False
)

df_top_by_channel.to_csv(top20_by_channel_csv_path, index=False)

print("Top 20 by channel saved to:", top20_by_channel_csv_path)