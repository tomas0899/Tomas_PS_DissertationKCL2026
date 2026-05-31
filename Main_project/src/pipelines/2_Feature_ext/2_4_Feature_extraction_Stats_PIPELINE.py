import pandas as pd
import numpy as np
# 0.1 Load modules and json config
# Get current file location
#===============================
#===============================
current_file = Path(__file__).resolve()

# Go up until you find the project root (where "src" exists)
for parent in current_file.parents:
    if (parent / "src").exists():
        project_root = parent
        break
# Add to PYTHONPATH if not already there
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Now import works
from src.modules import tools_EEG_FE as TEEG_FE
#===============================
#===============================
#===============================
# 1. load df
df_feat_ictalVspreictal_5min = pd.read_pickle("/home/tperezsanchez/FoundationModel_EEG_Dissertation/Main_project/results/XB47Y/Feature_ext/Part2_features/df_features_ictal_Vs_Preictal.pkl")

print(df_feat_ictalVspreictal_5min.head())
# 2. Divide into 2 groups according to label: preictal and ictal
  #  "preictal": 1,
  #  "seizure": 2
# Group 1: preictal windows
group_1_PREICTAL = df_feat_ictalVspreictal_5min[df_feat_ictalVspreictal_5min["class_label"] == 1].copy()

# Group 2: seizure windows
group_2_SEIZURE = df_feat_ictalVspreictal_5min[df_feat_ictalVspreictal_5min["class_label"] == 2].copy()

print("Shape of group 1 preictal:", group_1_PREICTAL.shape)
print("Shape of group 2 seizure:", group_2_SEIZURE.shape)

# 3. Select only numeric columns:

numeric_cols = df_feat_ictalVspreictal_5min.select_dtypes(include=[np.number]).columns.tolist()

print("Numeric columns:")
print(numeric_cols)
print("Number of numeric columns:", len(numeric_cols))
# Numeric columns that are metadata or not real signal features
exclude_cols = [
    "window_id",
    "start_sample",
    "end_sample",
    "fs",
    "n_channels",
    "class_label",
    "window_sec"
]

# Keep only true numeric feature columns
feature_cols = [col for col in numeric_cols if col not in exclude_cols]

print("Feature columns:")
print(feature_cols)
print("Number of feature columns:", len(feature_cols))
# 4. VIOLIN plots
TEEG_FE.df_mannwhitney_results_violin_2_8 = plot_mannwhitney_feature_violins(
    feature_cols=feature_cols,
    group_1_PREICTAL=group_1_PREICTAL,
    group_2_SEIZURE=group_2_SEIZURE,
    pdf_output_path="/home/tperezsanchez/FoundationModel_EEG_Dissertation/Main_project/results/mannwhitney_violin_plots.pdf",
    alpha=0.05,
    show_plots=False
)

df_mannwhitney_results_violin
# 4. Top 20 Mann-Whitney barplot
df_top_mannwhitney_features = plot_top_mannwhitney_features(
    df_mannwhitney_results=df_mannwhitney_results,
    top_n=20,
    pdf_output_path="/home/tperezsanchez/FoundationModel_EEG_Dissertation/Main_project/results/top_20_mannwhitney_features.pdf",
    show_plot=True
)

df_top_mannwhitney_features
# 5. top 20 by channel
df_top_by_channel, df_ranked = plot_top_features_by_channel(
    df_mannwhitney_results=df_mannwhitney_results,
    top_n=20,
    pdf_output_path="/home/tperezsanchez/FoundationModel_EEG_Dissertation/Main_project/results/top_20_features_by_channel.pdf",
    show_plot=True
)

df_top_by_channel