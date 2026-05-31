import json
from pathlib import Path
from datetime import datetime


# ============================================================
# JSON CONFIG GENERATOR
# Feature statistics: Mann-Whitney + violin plots + top features
# ============================================================


# ============================================================
# 1. USER-DEFINED INFORMATION
# ============================================================

patient_id = "XB47Y"

input_pkl_path = Path(
    "/home/tperezsanchez/FoundationModel_EEG_Dissertation/Main_project/results/XB47Y/Feature_ext/Part2_features/XB47Y_IN-normalized_npz_FP-fullnpz_W10s_PRE6to5min_ICT0to1min_GAPasINT_FINAL-PREvsSEIZ_20260504_v01_FEAT-TIME-FREQ_20260505_v01/XB47Y_IN-normalized_npz_FP-fullnpz_W10s_PRE6to5min_ICT0to1min_GAPasINT_FINAL-PREvsSEIZ_20260504_v01_FEAT-TIME-FREQ_20260505_v01_df_features_ictalVspreictal.pkl"
)

output_dir = Path(
    "/home/tperezsanchez/FoundationModel_EEG_Dissertation/Main_project/results/XB47Y/Feature_ext/Part3_Feat_stats"
)

config_output_dir = Path(
    "/home/tperezsanchez/FoundationModel_EEG_Dissertation/Main_project/configs/Feature_ext/Part3_Feat_stats"
)

version = "v01"

alpha = 0.05
top_n = 20

show_plots = False

class_label_column = "class_label"

class_labels = {
    "preictal": 1,
    "seizure": 2
}

exclude_cols = [
    "window_id",
    "start_sample",
    "end_sample",
    "fs",
    "n_channels",
    "class_label",
    "window_sec"
]

channel_patterns = [
    "EEG_SQ_D_SQ_C",
    "EEG_SQ_P_SQ_C"
]


# ============================================================
# 2. AUTO-GENERATED INFORMATION
# ============================================================

date_str = datetime.now().strftime("%Y%m%d")

# Extract experiment ID from the parent folder of the input pkl
experiment_id = input_pkl_path.parent.name


def clean_for_filename(text):
    """
    Clean text to make it safer for filenames.
    Keeps the structure mostly unchanged but removes problematic characters.
    """
    text = str(text)
    text = text.replace(" ", "_")
    text = text.replace("/", "-")
    text = text.replace("\\", "-")
    text = text.replace(":", "-")
    return text


experiment_id_clean = clean_for_filename(experiment_id)

output_dir.mkdir(parents=True, exist_ok=True)
config_output_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. AUTO-GENERATED OUTPUT FILENAMES
# ============================================================
# Format:
# PATIENTID_GRAPHTYPE_EXPERIMENTID_DATE_VERSION.extension

violin_pdf_path = output_dir / (
    f"{patient_id}_VIOLIN-MANNWHITNEY_"
    f"{experiment_id_clean}_"
    f"{date_str}_{version}.pdf"
)

top20_pdf_path = output_dir / (
    f"{patient_id}_TOP20-MANNWHITNEY-BARPLOT_"
    f"{experiment_id_clean}_"
    f"{date_str}_{version}.pdf"
)

top20_by_channel_pdf_path = output_dir / (
    f"{patient_id}_TOP20-BY-CHANNEL-BARPLOT_"
    f"{experiment_id_clean}_"
    f"{date_str}_{version}.pdf"
)

mannwhitney_csv_path = output_dir / (
    f"{patient_id}_MANNWHITNEY-RESULTS_"
    f"{experiment_id_clean}_"
    f"{date_str}_{version}.csv"
)

top20_csv_path = output_dir / (
    f"{patient_id}_TOP20-MANNWHITNEY-FEATURES_"
    f"{experiment_id_clean}_"
    f"{date_str}_{version}.csv"
)

top20_by_channel_csv_path = output_dir / (
    f"{patient_id}_TOP20-BY-CHANNEL-FEATURES_"
    f"{experiment_id_clean}_"
    f"{date_str}_{version}.csv"
)

ranked_features_csv_path = output_dir / (
    f"{patient_id}_RANKED-FEATURES-BY-CHANNEL_"
    f"{experiment_id_clean}_"
    f"{date_str}_{version}.csv"
)


# ============================================================
# 4. BUILD CONFIG DICTIONARY
# ============================================================

config = {
    "config_metadata": {
        "pipeline_name": "2_4_Feature_extraction_Stats_PIPELINE",
        "analysis_type": "feature_statistics_mannwhitney",
        "patient_id": patient_id,
        "experiment_id": experiment_id,
        "date_generated": date_str,
        "version": version
    },

    "inputs": {
        "input_pkl_path": str(input_pkl_path)
    },

    "outputs": {
        "output_dir": str(output_dir),

        "violin_pdf_path": str(violin_pdf_path),
        "top20_pdf_path": str(top20_pdf_path),
        "top20_by_channel_pdf_path": str(top20_by_channel_pdf_path),

        "mannwhitney_csv_path": str(mannwhitney_csv_path),
        "top20_csv_path": str(top20_csv_path),
        "top20_by_channel_csv_path": str(top20_by_channel_csv_path),
        "ranked_features_csv_path": str(ranked_features_csv_path)
    },

    "label_settings": {
        "class_label_column": class_label_column,
        "class_labels": class_labels
    },

    "feature_selection": {
        "exclude_cols": exclude_cols
    },

    "statistics": {
        "test": "mannwhitneyu",
        "alternative": "two-sided",
        "alpha": alpha
    },

    "plot_settings": {
        "top_n": top_n,
        "show_plots": show_plots,
        "channel_patterns": channel_patterns,
        "palette": {
            "preictal": "skyblue",
            "ictal": "salmon"
        }
    }
}


# ============================================================
# 5. SAVE CONFIG JSON
# ============================================================

config_filename = (
    f"config_{patient_id}_FEAT-STATS_"
    f"{date_str}_{version}.json"
)

config_output_path = config_output_dir / config_filename

with open(config_output_path, "w") as f:
    json.dump(config, f, indent=4)

print("Config JSON saved to:")
print(config_output_path)

print("\nGenerated output files:")
for key, value in config["outputs"].items():
    print(f"{key}: {value}")