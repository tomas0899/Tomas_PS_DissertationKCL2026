from pathlib import Path
import sys
import json
import pandas as pd
#==========================
#==========================
#==========================
# 0.1 Load modules 
# Get current file location
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
from src.modules import tools_EEG_Preprocess as TEEG_PR
#==========================
#==========================
#==========================
# 0.2 json config
# Usage:
# python run_pipeline.py configs/config_XB47Y.json
#
# If no config path is passed, a default one will be used.
if len(sys.argv) > 1:
    config_path = Path(sys.argv[1])
else:
    config_path = project_root / "configs" / "config_XB47Y.json"

# Check that the config file exists
if not config_path.exists():
    raise FileNotFoundError(f"Config file not found: {config_path}")

# Read config file
with open(config_path, "r") as f:
    config = json.load(f)
#=========================
#==========================
#==========================
# 0.3 EXTRACT VALUES FROM CONFIG

patient_id = config["patient_id"]

input_dir = config["paths"]["input_dir"]
seizure_file = config["paths"]["seizure_file"]
map_output_path = config["paths"]["map_output_path"]
npz_output_dir = config["paths"]["npz_output_dir"]
viz_output_dir = config["paths"]["viz_output_dir"]
qc_output_dir = config["paths"]["qc_output_dir"]

histogram_cfg = config["daily_recording_histogram"]
histogram_pdf_output_path = histogram_cfg["pdf_output_path"]

summary_cfg = config["recording_onset_summary"]
amp_threshold = config["filtering"]["amp_threshold"]
lowcut = config["filtering"]["lowcut"]
highcut = config["filtering"]["highcut"]
order = config["filtering"]["order"]
do_zscore = config["filtering"]["do_zscore"]
notch_freq = config["filtering"]["notch_freq"]

show_plot = config["plotting"]["show_plot"]
channel_idx_1 = config["plotting"]["channel_idx_1"]
channel_idx_2 = config["plotting"]["channel_idx_2"]
window_sec = config["plotting"]["window_sec"]
n_windows = config["plotting"]["n_windows"]
pre_onset_sec = config["plotting"]["pre_onset_sec"]
vertical_offset_uv = config["plotting"]["vertical_offset_uv"]

target_date = config["inspection"]["target_date"]
print("Config loaded successfully")
print("patient_id:", patient_id)
print("input_dir:", input_dir)
print("seizure_file:", seizure_file)
print("npz_output_dir:", npz_output_dir)
print("viz_output_dir:", viz_output_dir)
#==========================
#==========================
#==========================
# 1. Open/load .mat files
# open all the .mat files from the folder of a single patient
# create a df with all the information
df_patient, error_list = TEEG_PR.process_eeg_mat_files_1_1(input_dir)
#print(df_patient.head())
#==========================
#==========================
#==========================

# 1.2 Visualize distribution of daily accumulated recording time
print("STEP 1.2 START - daily recording histogram", flush=True)

if histogram_cfg.get("save_pdf", True):
    print("me tranque cuando entre al loop del histo")
    TEEG_PR.plot_daily_recording_histogram_1_2(
        df_patient,
        patient_id=patient_id,
        pdf_output_path=histogram_pdf_output_path
    )

else:

    TEEG_PR.plot_daily_recording_histogram_1_2(
        df_patient,
        patient_id=patient_id,
        pdf_output_path=None
    )

print("STEP 1.2 END - daily recording histogram", flush=True)


# 1.3 Gather seizure data from CSV
print("STEP 1.3 START - preprocess seizure file", flush=True)
print("Seizure file:", seizure_file, flush=True)

df_sq, df_di = TEEG_PR.preprocess_seizure_data_1_3(seizure_file)

print("STEP 1.3 END - preprocess seizure file", flush=True)
if histogram_cfg.get("save_pdf", True):

    TEEG_PR.plot_daily_recording_histogram_1_2(
        df_patient,
        patient_id=patient_id,
        pdf_output_path=histogram_pdf_output_path
    )

else:

    TEEG_PR.plot_daily_recording_histogram_1_2(
        df_patient,
        patient_id=patient_id,
        pdf_output_path=None
    )
print("STEP 1.2 END - daily recording histogram", flush=True)


# 1.3 Gather seizure data from CSV
print("STEP 1.3 START - preprocess seizure file", flush=True)
print("Seizure file:", seizure_file, flush=True)

df_sq, df_di = TEEG_PR.preprocess_seizure_data_1_3(seizure_file)

print("STEP 1.3 END - preprocess seizure file", flush=True)
#==========================
#==========================
#==========================
# 1.3 Gather seizure data from CSV
df_sq, df_di = TEEG_PR.preprocess_seizure_data_1_3(seizure_file)
# 1.3.2 Processing summary
summary_cfg = config["recording_onset_summary"]

if summary_cfg.get("save_csv", True):

    df_recording_onset_summary = TEEG_PR.save_recording_onset_summary_1_2_1(
        df_files=df_patient,
        df_onsets=df_sq,
        patient_id=patient_id,
        output_dir=summary_cfg["output_dir"],
        output_filename=summary_cfg.get("output_filename", None),
        t0_col=summary_cfg.get("t0_col", "T0"),
        tf_col=summary_cfg.get("tf_col", "TF"),
        onset_col=summary_cfg.get("onset_col", "onset")
    )
#==========================
#==========================
#==========================
print("STEP 1.4 START - seizure availability map", flush=True)
# 1.4 Mapping of seizures in all mat files
df_matches = TEEG_PR.plot_eeg_availability_with_onsetsV2_1_5(
    df_files=df_patient, 
    df_onsets=df_sq, 
    pdf_output_path=map_output_path,
    plots_per_page=10,
    show_plot=show_plot
)
print("STEP 1.4 END - seizure availability map", flush=True)
#search for the mat file with the onset on the 2019-12-11


# 1. Ensure T0 is in datetime format (just in case)
df_matches['T0'] = pd.to_datetime(df_matches['T0'])

# 2. Filter by comparing only the date part (.dt.date)
# Note: both pd.Timestamp or datetime.date objects work for this comparison
target_date_cfg = pd.to_datetime(target_date).date()
df_filtered = df_matches[df_matches['T0'].dt.date == target_date_cfg]

# Show results
#print(f"Found {len(df_filtered)} records for the date: {target_date}")
#display(df_matches.head())
df_matches
#==========================
#==========================
#==========================
# 1.5 Merge both Df 
# Step 1: Create a reduced version of df_matches with only relevant columns
# This does NOT modify df_matches; it creates a new DataFrame
df_match_small = df_matches[["file", "onset", "captured"]]

# Step 2: Merge df_patient with df_match_small using a LEFT JOIN on "file"
# - Keeps ALL rows from df_patient
# - Adds "onset" and "captured" where a match is found
# - If no match exists, NaN values are assigned
# - Result is stored in a new DataFrame (df_merged), original DataFrames remain unchanged
df_merged = df_patient.merge(df_match_small, on="file", how="left")

# Final result: df_merged contains EEG data + matched clinical/event metadata
df_merged

# checking that the files with onset are repeated. as they should be, because the 
#print(df_matches.columns.tolist())
#print(df_matches[df_matches["file"] == "patients.mat"])

# checking that the files with onset are repeated. as they should be, because the df should keep the onset records
#print(df_merged.columns.tolist())
#print(df_merged[df_merged["file"] == "XB47Y_182.mat"])
#==========================
#==========================
#==========================

# 1.6 GET LIST FOR UNIQUE MATCH
# PRINT ALL THE MAT FILES THAT HAVE A PRESENCE OF A SEIZURE
#df_matches
df_Unique_match = df_matches['file'].unique()
df_Unique_match
#print(type(df_Unique_match))
#for file in df_Unique_match:
#    print(file)
list_Unique_match = df_Unique_match.tolist()
#print(type(list_Unique_match))
# unique list from all the mat files
# this is the input for my function to get the npz
files_to_process = sorted(df_patient["file"].dropna().astype(str).unique().tolist())
#print(files_to_process)
#everything must pass
#==========================
#==========================
#==========================

# 1.7 GENERATE ALL .NPZ FILES
import os

output_dir = npz_output_dir
os.makedirs(output_dir, exist_ok=True)
os.makedirs(viz_output_dir, exist_ok=True)
#Output .npz contains:
 # X:              (C, N) full z-scored signal
 # mu:             (C,)   mean per channel
 # sigma:          (C,)   standard deviation per channel
 # fs:             float  sampling rate
 # channel_names:  (C,)
 # source_file:    (1,)
 # seizure_onsets: (K,)   ISO-format datetimes associated with this .mat file (K may be 0)
#T0:              (K,)   recording start timestamps (ISO format) from
                           #df_matches (may repeat if multiple matches exist)
 # TF:              (K,)   recording end timestamps (ISO format) from
                           #df_matches (may repeat if multiple matches exist)
print("STEP 1.7 START - generate NPZ files", flush=True)
TEEG_PR.full_recording_from_matfiles_1_9_V2(
    input_dir=input_dir,
    output_dir=output_dir,
    files_to_process=files_to_process,
    df_matches=df_merged,
    amp_threshold=amp_threshold,
    lowcut=lowcut,
    highcut=highcut,
    order=order,
    do_zscore=do_zscore,
    notch_freq=notch_freq,
)
print("STEP 1.7 END - generate NPZ files", flush=True)
#==========================
#==========================
#==========================

# 1.8 VISUALIZE WINDOWS FROM NPZ FILES
# VERSION With channel overlap and pre-ictal
# Final
import os
print("STEP 1.8 START - visualize seizure windows from NPZ", flush=True)

directory = npz_output_dir
for file_name in sorted(os.listdir(directory)):

    if file_name.endswith("_preproc_full.npz"):

        full_path = os.path.join(directory, file_name)

        print(f"\nProcessing: {file_name}")

        
        TEEG_PR.visualize_seizure_windows_from_npz_1_10V3(
            npz_path=full_path,
            channel_idx_1=channel_idx_1,
            channel_idx_2=channel_idx_2,
            window_sec=window_sec,
            n_windows=n_windows,
            pre_onset_sec=pre_onset_sec,
            vertical_offset_uv=vertical_offset_uv,
            output_dir=viz_output_dir
        )

print("patient_id:", patient_id)
print("input_dir:", input_dir)
print("seizure_file:", seizure_file)
print("npz_output_dir:", npz_output_dir)
print("viz_output_dir:", viz_output_dir)