import json
from pathlib import Path
from datetime import datetime

# ==========================================================
# JSON CONFIG GENERATOR FOR EEG WINDOWING + LABELING PIPELINE
# ==========================================================
# This script creates a JSON configuration file for the EEG
# windowing + labeling pipeline.
#
# The user defines:
# - patient ID
# - input folder
# - output root folder
# - file pattern
# - window size
# - labeling ranges
# - filtering options
#
# The script automatically generates:
# - experiment_id
# - output folder
# - output filenames
# - config filename
#
# Output names depend on:
# - patient_id
# - input folder name
# - file pattern
# - window size
# - preictal range
# - ictal range
# - gap/interictal setting
# - final filtering option
# - date
# - version
# ==========================================================


# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def clean_number_for_name(x):
    """
    Convert numbers into filename-safe strings.

    Examples:
    10   -> 10
    0.5  -> 0p5
    -6   -> 6
    """
    if x is None:
        return "None"

    x = float(x)

    if x.is_integer():
        x = int(x)

    text = str(abs(x))
    text = text.replace(".", "p")

    return text


def clean_text_for_name(text):
    """
    Convert text into a filename-safe string.

    Example:
    '*full.npz' -> 'fullnpz'
    """
    text = str(text)
    text = text.replace("*", "")
    text = text.replace(".", "")
    text = text.replace("/", "-")
    text = text.replace(" ", "")
    return text


def range_to_code(range_min, prefix):
    """
    Convert a time range in minutes into a compact code.

    Examples:
    preictal_range_min = [-6, -5] -> PRE6to5min
    ictal_range_min    = [0, 1]   -> ICT0to1min
    """
    start, end = range_min

    start_code = clean_number_for_name(start)
    end_code = clean_number_for_name(end)

    return f"{prefix}{start_code}to{end_code}min"


def build_experiment_id(
    patient_id,
    input_npz_dir,
    file_pattern,
    window_sec,
    preictal_range_min,
    ictal_range_min,
    include_gap_as_interictal,
    keep_only_preictal_seizure,
    date_code,
    version
):
    """
    Build a reproducible experiment ID based on user-defined
    inputs, outputs and parameters.
    """

    input_name = Path(input_npz_dir).name

    file_pattern_code = f"FP-{clean_text_for_name(file_pattern)}"
    window_code = f"W{clean_number_for_name(window_sec)}s"
    preictal_code = range_to_code(preictal_range_min, prefix="PRE")
    ictal_code = range_to_code(ictal_range_min, prefix="ICT")

    gap_code = "GAPasINT" if include_gap_as_interictal else "NOgapINT"

    filter_code = (
        "FINAL-PREvsSEIZ"
        if keep_only_preictal_seizure
        else "FINAL-ALLLABELS"
    )

    experiment_id = (
        f"{patient_id}_"
        f"IN-{input_name}_"
        f"{file_pattern_code}_"
        f"{window_code}_"
        f"{preictal_code}_"
        f"{ictal_code}_"
        f"{gap_code}_"
        f"{filter_code}_"
        f"{date_code}_"
        f"{version}"
    )

    return experiment_id


def create_labeling_config():

    # ======================================================
    # USER-DEFINED SETTINGS
    # ======================================================

    # ------------------------------------------------------
    # PATIENT INFORMATION
    # ------------------------------------------------------
    patient_id = "JYXFE"

    # ------------------------------------------------------
    # INPUT PATHS
    # ------------------------------------------------------
    input_npz_dir = ("/home/tperezsanchez/FoundationModel_EEG_Dissertation/Main_project/results/JYXFE/Pre_processing/JYXFE_IN-npz_GLOBALCH-NORM_CH2_EPS1e-08_20260511/normalized_npz"
    )

    # ------------------------------------------------------
    # OUTPUT ROOT FOLDER
    # ------------------------------------------------------
    # The user chooses this folder.
    # The script will create a specific experiment folder inside it.
    output_root_dir = ("/home/tperezsanchez/FoundationModel_EEG_Dissertation/Main_project/results/JYXFE/Feature_ext/Part1_labeling"
    )

    # ------------------------------------------------------
    # CONFIG OUTPUT FOLDER
    # ------------------------------------------------------
    config_output_dir = ("/home/tperezsanchez/FoundationModel_EEG_Dissertation/Main_project/src/pipelines/2_Feature_ext/configs"
    )

    # ------------------------------------------------------
    # VERSION
    # ------------------------------------------------------
    version = "v01"

    # ------------------------------------------------------
    # FILE SELECTION PARAMETERS
    # ------------------------------------------------------
    file_selection = {
        "file_pattern": "*full.npz"
    }

    # ------------------------------------------------------
    # WINDOWING PARAMETERS
    # ------------------------------------------------------
    windowing = {
        "window_sec": 10
    }

    # ------------------------------------------------------
    # LABELING PARAMETERS
    # ------------------------------------------------------
    labeling = {
        # Preictal window range in minutes before seizure onset
        "preictal_range_min": [-8, -5],

        # Ictal window range in minutes after seizure onset
        "ictal_range_min": [0, 3],

        # Whether to label gaps as interictal class 0
        "include_gap_as_interictal": True,

        # Class mapping used by the labeling pipeline
        "class_mapping": {
            "interictal": 0,
            "preictal": 1,
            "seizure": 2
        },

        # Human-readable explanation
        "labeling_description": (
            "Windows are labelled according to their temporal position relative "
            "to seizure onset. Preictal windows are defined by preictal_range_min. "
            "Ictal/seizure windows are defined by ictal_range_min. Gaps can "
            "optionally be labelled as interictal."
        )
    }

    # ------------------------------------------------------
    # FILTERING PARAMETERS
    # ------------------------------------------------------
    filtering = {
        # Keep only preictal (1) and seizure (2)
        "keep_only_preictal_seizure": True
    }

    # ------------------------------------------------------
    # PIPELINE STEPS
    # ------------------------------------------------------
    pipeline_steps = {
        "load_metadata": True,
        "temporal_sanity_check": True,
        "clean_onsets": True,
        "run_windowing": True,
        "run_labeling": True,
        "run_filtering": True,
        "save_outputs": True
    }

    # ======================================================
    # AUTOMATIC NAMING
    # ======================================================

    date_code = datetime.now().strftime("%Y%m%d")
    created_at = datetime.now().isoformat(timespec="seconds")

    experiment_id = build_experiment_id(
        patient_id=patient_id,
        input_npz_dir=input_npz_dir,
        file_pattern=file_selection["file_pattern"],
        window_sec=windowing["window_sec"],
        preictal_range_min=labeling["preictal_range_min"],
        ictal_range_min=labeling["ictal_range_min"],
        include_gap_as_interictal=labeling["include_gap_as_interictal"],
        keep_only_preictal_seizure=filtering["keep_only_preictal_seizure"],
        date_code=date_code,
        version=version
    )

    # Main experiment output folder
    output_dir = Path(output_root_dir) / experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Automatically generated output filenames
    df_labeled_filename = f"{experiment_id}_df_labeled_all.pkl"
    df_final_filename = f"{experiment_id}_df_final_preictal_vs_seizure.pkl"
    label_summary_filename = f"{experiment_id}_label_summary.csv"

    # Full output paths
    df_labeled_path = output_dir / df_labeled_filename
    df_final_path = output_dir / df_final_filename
    label_summary_path = output_dir / label_summary_filename

    # ======================================================
    # CONFIG DICTIONARY
    # ======================================================

    config = {
        # --------------------------------------------------
        # EXPERIMENT INFORMATION
        # --------------------------------------------------
        "experiment_id": experiment_id,
        "created_at": created_at,
        "date_code": date_code,
        "version": version,

        # --------------------------------------------------
        # PATIENT INFORMATION
        # --------------------------------------------------
        "patient_id": patient_id,

        # --------------------------------------------------
        # FILE AND FOLDER PATHS
        # --------------------------------------------------
        "paths": {
            # Folder containing normalized .npz files
            "input_npz_dir": str(input_npz_dir),

            # User-defined root output folder
            "output_root_dir": str(output_root_dir),

            # Automatically generated output folder for this experiment
            "output_dir": str(output_dir)
        },

        # --------------------------------------------------
        # FILE SELECTION PARAMETERS
        # --------------------------------------------------
        "file_selection": file_selection,

        # --------------------------------------------------
        # WINDOWING PARAMETERS
        # --------------------------------------------------
        "windowing": windowing,

        # --------------------------------------------------
        # LABELING PARAMETERS
        # --------------------------------------------------
        "labeling": labeling,

        # --------------------------------------------------
        # FILTERING PARAMETERS
        # --------------------------------------------------
        "filtering": filtering,

        # --------------------------------------------------
        # OUTPUT FILES
        # --------------------------------------------------
        # These filenames are generated automatically.
        # Your labeling pipeline can still use:
        # config["output_files"]["df_labeled_filename"]
        # config["output_files"]["df_final_filename"]
        "output_files": {
            "df_labeled_filename": df_labeled_filename,
            "df_final_filename": df_final_filename,
            "label_summary_filename": label_summary_filename
        },

        # --------------------------------------------------
        # FULL OUTPUT PATHS
        # --------------------------------------------------
        # These are optional but useful because they avoid manually
        # combining output_dir + filename inside the pipeline.
        "output_paths": {
            "df_labeled_path": str(df_labeled_path),
            "df_final_path": str(df_final_path),
            "label_summary_path": str(label_summary_path)
        },

        # --------------------------------------------------
        # AUTOMATIC NAMING INFORMATION
        # --------------------------------------------------
        "naming": {
            "input_name": Path(input_npz_dir).name,
            "file_pattern_code": clean_text_for_name(file_selection["file_pattern"]),
            "window_code": f"W{clean_number_for_name(windowing['window_sec'])}s",
            "preictal_code": range_to_code(labeling["preictal_range_min"], prefix="PRE"),
            "ictal_code": range_to_code(labeling["ictal_range_min"], prefix="ICT"),
            "gap_code": (
                "GAPasINT"
                if labeling["include_gap_as_interictal"]
                else "NOgapINT"
            ),
            "filter_code": (
                "FINAL-PREvsSEIZ"
                if filtering["keep_only_preictal_seizure"]
                else "FINAL-ALLLABELS"
            ),
            "date_code": date_code,
            "version": version
        },

        # --------------------------------------------------
        # PIPELINE STEPS
        # --------------------------------------------------
        "pipeline_steps": pipeline_steps
    }

    # ======================================================
    # SAVE JSON FILE
    # ======================================================

    config_output_dir = Path(config_output_dir)
    config_output_dir.mkdir(parents=True, exist_ok=True)

    config_path = config_output_dir / f"config_{experiment_id}.json"

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    print("Config saved at:")
    print(config_path.resolve())

    print("\nExperiment ID:")
    print(experiment_id)

    print("\nExperiment output directory:")
    print(output_dir.resolve())

    print("\nGenerated output files:")
    print(df_labeled_path)
    print(df_final_path)
    print(label_summary_path)

    return config


# ==========================================================
# RUN SCRIPT
# ==========================================================

if __name__ == "__main__":
    create_labeling_config()


# ==========================================================
# HOW TO RUN
# ==========================================================
# 1. Generate the JSON config:
# python create_config_labeling.py
#
# 2. Run the labeling pipeline with the generated config:
# python 2_1_Feature_ext_labeling.py configs/config_<experiment_id>.json