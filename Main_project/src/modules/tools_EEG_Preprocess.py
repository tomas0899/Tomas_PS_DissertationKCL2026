# Change - Tears for fears 
import os
import glob
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import scipy.io as sio
from scipy.io import loadmat
from scipy.stats import skew, kurtosis
from scipy.signal import welch, iirnotch, tf2sos, butter, sosfiltfilt
from matplotlib.backends.backend_pdf import PdfPages
#=================================================================================
#=================================================================================
#=================================================================================
# FUNCTION #1

def process_eeg_mat_files_1_1(folder_path: str) -> Tuple[pd.DataFrame, list]:
    """
    Scans a folder for .mat EEG files, extracts temporal metadata (T0, TF), 
    calculates durations, and identifies recording gaps.

    Args:
        folder_path (str): Path to the directory containing .mat files.

    Returns:
        Tuple[pd.DataFrame, list]: A sorted DataFrame of results and a list of errors.
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The path '{folder_path}' does not exist.")

    # 1) Filter and Sort Files
    all_files = os.listdir(folder_path)
    mat_files = sorted([
        f for f in all_files 
        if f.lower().endswith(".mat") and not f.startswith(".")
    ])

    results = []
    errors = []

    # 2) Extract Metadata
    for i, filename in enumerate(mat_files):
        file_path = os.path.join(folder_path, filename)

        try:
            data_mat = loadmat(file_path)
            hdr = data_mat["hdr"]

            # Extract T0 (Start Time)
            t0_raw = hdr["orig"][0, 0]["T0"][0, 0][0]
            t0_dt = datetime(
                int(t0_raw[0]), int(t0_raw[1]), int(t0_raw[2]),
                int(t0_raw[3]), int(t0_raw[4]), int(t0_raw[5])
            )

            # Extract Sampling Frequency and Data
            fs = float(hdr["Fs"][0, 0].item())
            signal = np.asarray(data_mat["data"])
            
            # Channel validation
            channels_raw = hdr["label"][0, 0]
            n_channels = channels_raw.shape[0]

            if signal.ndim == 2:
                # Transpose if shape is (n_channels, n_samples)
                if signal.shape[1] != n_channels and signal.shape[0] == n_channels:
                    signal = signal.T
            else:
                raise ValueError(f"Unexpected signal dimensions: {signal.shape}")

            n_samples = signal.shape[0]
            duration_seconds = n_samples / fs
            tf_dt = t0_dt + timedelta(seconds=duration_seconds)

            results.append({
                "list_idx": i,
                "file": filename,
                "T0": t0_dt,
                "TF": tf_dt,
                "duration_s": duration_seconds
            })

        except Exception as e:
            errors.append((filename, str(e)))

    # 3) Data Organization & Gap Calculation
    if not results:
        print("No valid data processed.")
        return pd.DataFrame(), errors

    df = pd.DataFrame(results)
    
    # Sort by actual Start Time (T0)
    df = df.sort_values("T0").reset_index(drop=True)
    
    # Calculate Gaps between files: T0 of current - TF of previous
    df["gap_s"] = (df["T0"] - df["TF"].shift(1)).dt.total_seconds().fillna(0)

    # Logging summary
    print(f"--- Processing Summary ---")
    print(f"Successfully processed: {len(df)}")
    print(f"Errors encountered: {len(errors)}")
    print(f"Total significant gaps (>1s): {(df['gap_s'] > 1).sum()}")
    
    return df, errors

# --- Example Usage ---
# path = "/your/folder/path/here/"
# df_results, error_list = process_eeg_mat_files(path)
# print(df_results.head())

    
#=================================================================================
#=================================================================================
#=================================================================================
# FUNCTION #2 
def plot_daily_recording_histogram_1_2(df, patient_id="Unknown", pdf_output_path=None):    
    """
    Groups recording data by day, calculates statistics, and plots a histogram 
    of total accumulated hours per day.
    """
    # --- 1. Preparation ---
    df_use = df.copy()

    # Ensure T0 is datetime
    df_use['T0'] = pd.to_datetime(df_use['T0'])

    # --- 2. Daily Grouping ---
    # Group by date and sum duration in seconds
    df_daily = df_use.groupby(df_use['T0'].dt.date)['duration_s'].sum().reset_index()

    # Rename for clarity
    df_daily.columns = ['date', 'total_duration_s']

    # Convert daily total to hours
    df_daily["hours_accumulated"] = df_daily["total_duration_s"] / 3600.0

    # --- 3. Statistics ---
    mean_h = df_daily["hours_accumulated"].mean()
    median_h = df_daily["hours_accumulated"].median()
    min_h = df_daily['hours_accumulated'].min()
    max_h = df_daily['hours_accumulated'].max()

    print(f"--- Statistics per Day (Patient: {patient_id}) ---")
    print(f"Days analyzed : {len(df_daily)}")
    print(f"Mean duration : {mean_h:.2f} h")
    print(f"Median        : {median_h:.2f} h")
    print(f"Min / Max     : {min_h:.2f} / {max_h:.2f} h")

    # --- 4. Histogram ---
    plt.figure(figsize=(10, 6))
    
    # Using 15 bins as requested
    plt.hist(df_daily["hours_accumulated"], bins=15, color="#3498db", edgecolor="white", alpha=0.8)

    # Reference lines
    plt.axvline(mean_h, color="red", linestyle='-', linewidth=2, label=f"Mean: {mean_h:.2f}h")
    plt.axvline(median_h, color="orange", linestyle='--', linewidth=2, label=f"Median: {median_h:.2f}h")

    # Titles and Labels
    plt.title(f"Distribution of Daily Accumulated Recording Time - Patient: {patient_id}", fontsize=14)
    plt.xlabel("Total Hours per Day", fontsize=12)
    plt.ylabel("Frequency (Number of Days)", fontsize=12)
    plt.legend()
    plt.grid(axis='y', linestyle=':', alpha=0.7)

    plt.tight_layout()
    if pdf_output_path is not None:
        pdf_output_path = Path(pdf_output_path)
        pdf_output_path.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(
            pdf_output_path,
            format="pdf",
            bbox_inches="tight"
        )
    
        print(f"PDF saved to: {pdf_output_path}")
    plt.close()

# --- Example Usage ---
# plot_daily_recording_histogram(df_XB47Y, patient_id="XB47Y")
def save_recording_onset_summary_1_2_1(
    df_files,
    df_onsets,
    patient_id="Unknown",
    output_dir=".",
    output_filename=None,
    t0_col="T0",
    tf_col="TF",
    onset_col="onset"
):
    """
    Creates and saves a CSV summary comparing EEG recording files and seizure onsets.

    The output CSV includes:
    - Number of recording files
    - Number of seizure onsets
    - First and last recording timestamps
    - First and last onset timestamps
    - Number of onsets per day

    Args:
        df_files (pd.DataFrame): DataFrame containing EEG file metadata.
        df_onsets (pd.DataFrame): DataFrame containing seizure onset information.
        patient_id (str): Patient identifier.
        output_dir (str or Path): Directory where the CSV will be saved.
        output_filename (str, optional): Name of the output CSV file.
        t0_col (str): Column name for recording start time.
        tf_col (str): Column name for recording end time.
        onset_col (str): Column name for seizure onset time.

    Returns:
        pd.DataFrame: Summary DataFrame saved as CSV.
    """

    # --- 1. Prepare output path ---
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_filename is None:
        output_filename = f"{patient_id}_recording_onset_summary.csv"

    output_path = output_dir / output_filename

    # --- 2. Convert date columns ---
    df_files = df_files.copy()
    df_onsets = df_onsets.copy()

    df_files[t0_col] = pd.to_datetime(df_files[t0_col])
    df_files[tf_col] = pd.to_datetime(df_files[tf_col])
    df_onsets[onset_col] = pd.to_datetime(df_onsets[onset_col])

    # --- 3. General summary ---
    summary_rows = [
        {
            "patient_id": patient_id,
            "section": "recording_summary",
            "metric": "number_of_files",
            "value": len(df_files)
        },
        {
            "patient_id": patient_id,
            "section": "onset_summary",
            "metric": "number_of_onsets",
            "value": len(df_onsets)
        },
        {
            "patient_id": patient_id,
            "section": "recording_summary",
            "metric": "first_T0",
            "value": df_files[t0_col].min()
        },
        {
            "patient_id": patient_id,
            "section": "recording_summary",
            "metric": "last_TF",
            "value": df_files[tf_col].max()
        },
        {
            "patient_id": patient_id,
            "section": "onset_summary",
            "metric": "first_onset",
            "value": df_onsets[onset_col].min()
        },
        {
            "patient_id": patient_id,
            "section": "onset_summary",
            "metric": "last_onset",
            "value": df_onsets[onset_col].max()
        }
    ]

    # --- 4. Onsets per day ---
    onsets_per_day = (
        df_onsets[onset_col]
        .dt.date
        .value_counts()
        .sort_index()
    )

    for date, count in onsets_per_day.items():
        summary_rows.append(
            {
                "patient_id": patient_id,
                "section": "onsets_per_day",
                "metric": str(date),
                "value": count
            }
        )

    # --- 5. Save CSV ---
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(output_path, index=False)

    print(f"Recording/onset summary CSV saved to: {output_path}")

    return df_summary
#=================================================================================
#=================================================================================
#=================================================================================
# FUNCTION #3


def preprocess_seizure_data_1_3(seizure_xlsx_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extracts seizure sheets from Excel, saves them as CSVs in a patient-specific 
    folder, and normalizes datetime columns.

    Args:
        seizure_xlsx_path (str): Full path to the .xlsx file.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Cleaned (df_sqEEG, df_diary)
    """
    # 1) Setup Paths and Folder Names
    xlsx_path = Path(seizure_xlsx_path)
    base_dir = xlsx_path.parent
    patient_id = base_dir.name.upper()
    
    output_folder = base_dir / f"preprocessCSV_{patient_id}"
    output_folder.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing Patient: {patient_id}")
    print(f"Output directory: {output_folder}")

    # 2) Load Sheets
    # Using a dictionary to handle potential missing sheets gracefully
    try:
        df_sqEEG = pd.read_excel(xlsx_path, sheet_name="sqEEG")
        df_diary = pd.read_excel(xlsx_path, sheet_name="diary")
    except Exception as e:
        print(f"Error reading sheets: {e}")
        raise

    # 3) Normalize Datetime Format
    # sqEEG usually uses 'onset', diary uses 'Timestamp'
    if "onset" in df_sqEEG.columns:
        df_sqEEG["onset"] = pd.to_datetime(df_sqEEG["onset"])
        
    if "Timestamp" in df_diary.columns:
        df_diary["Timestamp"] = pd.to_datetime(df_diary["Timestamp"])

    # 4) Save to CSV
    sqeeg_csv_path = output_folder / "sqEEG.csv"
    diary_csv_path = output_folder / "diary.csv"
    
    df_sqEEG.to_csv(sqeeg_csv_path, index=False)
    df_diary.to_csv(diary_csv_path, index=False)
    
    print(f"Successfully saved: \n - {sqeeg_csv_path.name} \n - {diary_csv_path.name}")

    return df_sqEEG, df_diary

# --- Example Usage ---
# file_path = "/home/tperezsanchez/FoundationModel_EEG_Dissertation/EEG_data_vis/data/Working/XB47Y/XB47Y_seizures.xlsx"
# df_sq, df_di = preprocess_seizure_data(file_path)

    
#=================================================================================
#=================================================================================
#=================================================================================
# FUNCTION #4



def plot_eeg_availability_with_onsetsV2_1_4(
    df_files: pd.DataFrame, 
    df_onsets: pd.DataFrame, 
    output_path: Optional[str] = None,
    show_plot: bool = True
) -> pd.DataFrame:
    """
    Plots daily EEG recording availability, overlays seizure onsets,
    shows total daily hours and lists matched onset times in the plot.
    """
    # 0) Preparación de datos
    df_files = df_files.copy()
    df_onsets = df_onsets.copy()
    
    df_files["T0"] = pd.to_datetime(df_files["T0"])
    df_files["TF"] = pd.to_datetime(df_files["TF"])
    df_onsets["onset"] = pd.to_datetime(df_onsets["onset"])

    # --- LÓGICA DE MATCHING ---
    df_files = df_files.sort_values("T0")
    df_onsets = df_onsets.sort_values("onset")

    # Unimos para saber qué onset cae en qué archivo
    matched_df = pd.merge_asof(
        df_onsets, 
        df_files, 
        left_on="onset", 
        right_on="T0", 
        direction="backward"
    )

    # El onset debe estar dentro del rango [T0, TF] del archivo
    matched_df["captured"] = (matched_df["onset"] >= matched_df["T0"]) & \
                             (matched_df["onset"] <= matched_df["TF"])
    
    df_captured_onsets = matched_df[matched_df["captured"] == True].copy()

    # 1) Calcular Estado Binario (para el escalón del gráfico)
    events = []
    for _, row in df_files.iterrows():
        events.append((row["T0"], +1))
        events.append((row["TF"], -1))

    events_df = pd.DataFrame(events, columns=["Time", "Delta"]).sort_values("Time")
    events_df = events_df.groupby("Time", as_index=False)["Delta"].sum().sort_values("Time")
    events_df["State"] = events_df["Delta"].cumsum()
    events_df["Presence"] = (events_df["State"] > 0).astype(int)
    events_df["DayStart"] = events_df["Time"].dt.floor("D")
    
    unique_days = sorted(events_df["DayStart"].unique())

    # 2) Configuración del Plot
    fig, axes = plt.subplots(
        len(unique_days), 1, 
        figsize=(14, 3 * len(unique_days)), 
        sharey=True, 
        constrained_layout=True
    )
    if len(unique_days) == 1: axes = [axes]

    # 3) Plot por cada día
    for ax, start_day in zip(axes, unique_days):
        start_day = pd.Timestamp(start_day)
        end_day = start_day + pd.Timedelta(days=1)

        # Filtrar datos del día para la línea de presencia
        day_data = events_df[(events_df["Time"] >= start_day) & (events_df["Time"] < end_day)].copy()
        
        # Lógica de bordes para que no haya huecos al inicio/fin del día
        prev_state = events_df.loc[events_df["Time"] < start_day, "State"]
        presence_at_start = int(prev_state.iloc[-1] > 0) if not prev_state.empty else 0
        boundary_points = pd.DataFrame({"Time": [start_day, end_day], "Presence": [presence_at_start, None]})
        day_data = pd.concat([day_data[["Time", "Presence"]], boundary_points], ignore_index=True).sort_values("Time")
        day_data["Presence"] = day_data["Presence"].ffill().astype(int)

        # --- CÁLCULO DE DURACIÓN TOTAL ---
        # Calculamos la diferencia entre puntos de cambio de estado
        day_data["Duration"] = day_data["Time"].diff().shift(-1)
        total_duration_td = day_data.loc[day_data["Presence"] == 1, "Duration"].sum()
        total_hours = total_duration_td.total_seconds() / 3600

        # --- IDENTIFICAR ONSETS DEL DÍA ---
        day_onsets = matched_df[(matched_df["onset"] >= start_day) & (matched_df["onset"] < end_day)]
        captured_list = []

        for _, s_row in day_onsets.iterrows():
            color = "red" if s_row["captured"] else "gray"
            ax.axvline(s_row["onset"], color=color, linestyle="--", linewidth=1.5, alpha=0.8)
            
            if s_row["captured"]:
                # Guardamos la hora formateada para la leyenda interna
                captured_list.append(s_row["onset"].strftime("%H:%M:%S"))

        # --- VISUALS ---
        ax.step(day_data["Time"], day_data["Presence"], where="post", color="steelblue", linewidth=2)
        ax.fill_between(day_data["Time"], day_data["Presence"], step="post", alpha=0.2, color="steelblue")
        
        # Título con horas acumuladas
        ax.set_title(f"Date: {start_day.date()} | Total Recording: {total_hours:.2f} hrs", 
                     loc='left', fontweight='bold', fontsize=12)
        
        # Cuadro de texto con los onsets detectados
        if captured_list:
            onset_text = "Captured Onsets:\n" + "\n".join(captured_list)
            ax.text(1.01, 0.5, onset_text, transform=ax.transAxes, fontsize=9, 
                    verticalalignment='center', color="red",
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='red'))

        ax.set_ylim(-0.1, 1.1)
        ax.set_xlim(start_day, end_day)
        ax.set_ylabel("Presence")

    axes[-1].set_xlabel("Time (HH:MM)")

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    if show_plot: 
        plt.show()
    else: 
        plt.close()

    return df_captured_onsets
import os
from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def plot_eeg_availability_with_onsetsV2_1_5(
    df_files: pd.DataFrame, 
    df_onsets: pd.DataFrame, 
    pdf_output_path: Optional[str] = None,
    plots_per_page: int = 10,
    show_plot: bool = False
) -> pd.DataFrame:
    """
    Plots daily EEG recording availability, overlays seizure onsets,
    shows total daily hours, lists matched onset times in the plot,
    and optionally saves:
    
    1. A full PNG figure.
    2. A paginated PDF with a fixed number of daily plots per page.

    Args:
        df_files (pd.DataFrame): DataFrame containing EEG file metadata with T0 and TF columns.
        df_onsets (pd.DataFrame): DataFrame containing seizure onset times in an 'onset' column.
        output_path (Optional[str]): Path to save the full PNG figure.
        pdf_output_path (Optional[str]): Path to save the paginated PDF.
        plots_per_page (int): Number of daily plots per PDF page.
        show_plot (bool): Whether to display the full PNG-style figure.

    Returns:
        pd.DataFrame: DataFrame containing only captured seizure onsets.
    """

    # ==========================================================
    # 0) Prepare data
    # ==========================================================
    df_files = df_files.copy()
    df_onsets = df_onsets.copy()
    
    df_files["T0"] = pd.to_datetime(df_files["T0"])
    df_files["TF"] = pd.to_datetime(df_files["TF"])
    df_onsets["onset"] = pd.to_datetime(df_onsets["onset"])

    df_files = df_files.sort_values("T0")
    df_onsets = df_onsets.sort_values("onset")

    # ==========================================================
    # 1) Match each onset to the previous file start time
    # ==========================================================
    matched_df = pd.merge_asof(
        df_onsets, 
        df_files, 
        left_on="onset", 
        right_on="T0", 
        direction="backward"
    )

    matched_df["captured"] = (
        (matched_df["onset"] >= matched_df["T0"]) &
        (matched_df["onset"] <= matched_df["TF"])
    )
    
    df_captured_onsets = matched_df[matched_df["captured"] == True].copy()

    # ==========================================================
    # 2) Build binary EEG presence state
    # ==========================================================
    events = []

    for _, row in df_files.iterrows():
        events.append((row["T0"], +1))
        events.append((row["TF"], -1))

    events_df = pd.DataFrame(events, columns=["Time", "Delta"]).sort_values("Time")

    events_df = (
        events_df
        .groupby("Time", as_index=False)["Delta"]
        .sum()
        .sort_values("Time")
    )

    events_df["State"] = events_df["Delta"].cumsum()
    events_df["Presence"] = (events_df["State"] > 0).astype(int)
    events_df["DayStart"] = events_df["Time"].dt.floor("D")
    
    unique_days = sorted(events_df["DayStart"].unique())

    # ==========================================================
    # Helper function to plot one day in one axis
    # ==========================================================
    def plot_single_day(ax, start_day):
        start_day = pd.Timestamp(start_day)
        end_day = start_day + pd.Timedelta(days=1)

        day_data = events_df[
            (events_df["Time"] >= start_day) &
            (events_df["Time"] < end_day)
        ].copy()

        # State at the beginning of the day
        prev_state = events_df.loc[events_df["Time"] < start_day, "State"]
        presence_at_start = int(prev_state.iloc[-1] > 0) if not prev_state.empty else 0

        boundary_points = pd.DataFrame({
            "Time": [start_day, end_day],
            "Presence": [presence_at_start, None]
        })

        day_data = (
            pd.concat(
                [day_data[["Time", "Presence"]], boundary_points],
                ignore_index=True
            )
            .sort_values("Time")
        )

        day_data["Presence"] = day_data["Presence"].ffill().astype(int)

        # Calculate total recording duration for that day
        day_data["Duration"] = day_data["Time"].diff().shift(-1)
        total_duration_td = day_data.loc[day_data["Presence"] == 1, "Duration"].sum()
        total_hours = total_duration_td.total_seconds() / 3600

        # Identify onsets for that day
        day_onsets = matched_df[
            (matched_df["onset"] >= start_day) &
            (matched_df["onset"] < end_day)
        ]

        captured_list = []

        for _, s_row in day_onsets.iterrows():
            color = "red" if s_row["captured"] else "gray"

            ax.axvline(
                s_row["onset"],
                color=color,
                linestyle="--",
                linewidth=1.5,
                alpha=0.8
            )
            
            if s_row["captured"]:
                captured_list.append(s_row["onset"].strftime("%H:%M:%S"))

        # Plot EEG availability
        ax.step(
            day_data["Time"],
            day_data["Presence"],
            where="post",
            color="steelblue",
            linewidth=2
        )

        ax.fill_between(
            day_data["Time"],
            day_data["Presence"],
            step="post",
            alpha=0.2,
            color="steelblue"
        )

        ax.set_title(
            f"Date: {start_day.date()} | Total Recording: {total_hours:.2f} hrs",
            loc="left",
            fontweight="bold",
            fontsize=12
        )

        if captured_list:
            onset_text = "Captured Onsets:\n" + "\n".join(captured_list)

            ax.text(
                1.01,
                0.5,
                onset_text,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="center",
                color="red",
                bbox=dict(
                    facecolor="white",
                    alpha=0.6,
                    edgecolor="red"
                )
            )

        ax.set_ylim(-0.1, 1.1)
        ax.set_xlim(start_day, end_day)
        ax.set_ylabel("Presence")

    # ==========================================================
    # 3) Optional display of the full figure
    # ==========================================================
    # Important: only create this full figure if explicitly requested.
    # Otherwise, long recordings may generate an extremely large figure
    # and cause memory/X11 errors.
    
    if show_plot:
        max_days_to_show = 30  # safety limit for display
        days_to_show = unique_days[:max_days_to_show]
    
        fig, axes = plt.subplots(
            len(days_to_show),
            1,
            figsize=(14, 3 * len(days_to_show)),
            sharey=True,
            constrained_layout=True
        )
    
        if len(days_to_show) == 1:
            axes = [axes]
    
        for ax, start_day in zip(axes, days_to_show):
            plot_single_day(ax, start_day)
    
        axes[-1].set_xlabel("Time (HH:MM)")
    
        plt.show()
        plt.close(fig)

    # ==========================================================
    # 4) Save paginated PDF, 10 plots per page by default
    # ==========================================================
    if pdf_output_path:
        pdf_output_dir = os.path.dirname(pdf_output_path)

        if pdf_output_dir:
            os.makedirs(pdf_output_dir, exist_ok=True)

        with PdfPages(pdf_output_path) as pdf:
            for i in range(0, len(unique_days), plots_per_page):
                days_subset = unique_days[i:i + plots_per_page]

                fig_pdf, axes_pdf = plt.subplots(
                    len(days_subset),
                    1,
                    figsize=(14, 3 * len(days_subset)),
                    sharey=True,
                    constrained_layout=True
                )

                if len(days_subset) == 1:
                    axes_pdf = [axes_pdf]

                for ax, start_day in zip(axes_pdf, days_subset):
                    plot_single_day(ax, start_day)

                axes_pdf[-1].set_xlabel("Time (HH:MM)")

                pdf.savefig(fig_pdf, bbox_inches="tight")
                plt.close(fig_pdf)

        print(f"PDF saved to: {pdf_output_path}")

    return df_captured_onsets
#=================================================================================
#=================================================================================
#=================================================================================
# FUNCTION #5

def apply_amplitude_cutoff_1_5(
    EEG_Table: pd.DataFrame,
    threshold: float = 200,
    start_sec: float = None,
    end_sec: float = None
):
    """
    Clip EEG amplitudes at ±threshold (µV),
    optionally selecting a time window in seconds.

    Parameters
    ----------
    EEG_Table : pandas.DataFrame
        DataFrame containing 'Time' column OR time as index (in seconds)
    threshold : float
        Amplitude threshold in µV (default 200)
    start_sec : float, optional
        Start time of window (in seconds)
    end_sec : float, optional
        End time of window (in seconds)

    Returns
    -------
    EEG_clipped : pandas.DataFrame
        Windowed and clipped DataFrame
    """

    # Copiar para no modificar original
    EEG_clipped = EEG_Table.copy()

    # ---------------------------------------------------
    # 1) Selección de ventana temporal (si se especifica)
    # ---------------------------------------------------
    if start_sec is not None and end_sec is not None:

        if "Time" in EEG_clipped.columns:
            EEG_clipped = EEG_clipped[
                (EEG_clipped["Time"] >= start_sec) &
                (EEG_clipped["Time"] <= end_sec)
            ]
        else:
            EEG_clipped = EEG_clipped.loc[
                (EEG_clipped.index >= start_sec) &
                (EEG_clipped.index <= end_sec)
            ]

    # ---------------------------------------------------
    # 2) Aplicar clipping
    # ---------------------------------------------------
    if "Time" in EEG_clipped.columns:
        signal_cols = EEG_clipped.columns.drop("Time")
        EEG_clipped[signal_cols] = EEG_clipped[signal_cols].clip(
            lower=-threshold,
            upper=threshold
        )
    else:
        EEG_clipped = EEG_clipped.clip(
            lower=-threshold,
            upper=threshold
        )

    return EEG_clipped
#=================================================================================
#=================================================================================
#=================================================================================
# FUNCTION #6


def build_eeg_array_from_mat_1_6(
    hdr,
    mat_data,
    output_dir=".",
    file_prefix="EEG_data",
    save_format="npz",   # "npy" or "npz"
    return_dataframe=True,
    save=True
):
    """
    Build EEG array from .mat structure and save as .npy or .npz.

    Parameters
    ----------
    hdr : dict
        Header structure from .mat file
    mat_data : dict
        Full .mat dictionary
    output_dir : str
        Directory to save output
    file_prefix : str
        Prefix for output filename
    save_format : str
        "npy" (signal only) or "npz" (signal + metadata)
    return_dataframe : bool
        If True, also returns a DataFrame

    Returns
    -------
    signal : np.ndarray
    file_path : str
    (optional) EEG_Table : pandas.DataFrame
    """

    # Sampling frequency
    fs = float(np.squeeze(hdr['Fs']))

    # Channel labels
    channels_raw = hdr['label'][0,0]
    channels = [str(row[0][0]) for row in channels_raw]

    # Extract signal
    signal = np.asarray(mat_data['data'], dtype=np.float32)

    # Fix orientation if needed
    if signal.shape[1] != len(channels) and signal.shape[0] == len(channels):
        signal = signal.T

    n_samples = signal.shape[0]
    time = np.arange(n_samples, dtype=np.float64) / fs

    os.makedirs(output_dir, exist_ok=True)

    # -------- SAVE --------
    if save:
        
        if save_format == "npy":
            file_path = os.path.join(output_dir, f"{file_prefix}.npy")
            np.save(file_path, signal)
    
        elif save_format == "npz":
            file_path = os.path.join(output_dir, f"{file_prefix}.npz")
            np.savez(
                file_path,
                signal=signal,
                fs=fs,
                channels=channels,
                time=time
            )
    
        else:
            raise ValueError("save_format must be 'npy' or 'npz'")
    
        print(f"Saved EEG data to: {file_path}")
        print(f"Shape: {signal.shape}")
        print(f"Sampling frequency: {fs} Hz")
    else:
        file_path= None 
        #print(f"Saved EEG data to: {file_path}")
        print(f"Shape: {signal.shape}")
        print(f"Sampling frequency: {fs} Hz")
    if return_dataframe:
        EEG_Table = pd.DataFrame(signal, columns=channels)
        EEG_Table.insert(0, "Time", time)
        return signal, file_path, EEG_Table

    return signal, file_path
import matplotlib.pyplot as plt

#=================================================================================
#=================================================================================
#=================================================================================
# FUNCTION #7

def plot_eeg_signals_1_7(
    EEG_Table,
    time_window=None,      # tuple (start, end) in seconds
    y_limit=None,          # tuple (-200, 200)
    figsize=(12,6),
    color=None             # str or list of colors
):
    """
    Plot EEG signals from a DataFrame with Time column.

    Parameters
    ----------
    EEG_Table : pandas.DataFrame
        DataFrame containing 'Time' + EEG channels
    time_window : tuple or None
        (start_time, end_time) in seconds
    y_limit : tuple or None
        (ymin, ymax)
    figsize : tuple
        Figure size
    color : str or list
        Single color for all channels OR list of colors per channel
    """

    # Apply time window if provided
    if time_window is not None:
        start, end = time_window
        EEG_Table = EEG_Table[
            (EEG_Table["Time"] >= start) &
            (EEG_Table["Time"] <= end)
        ]

    EEG_TimeTable = EEG_Table.set_index("Time")

    fig, axes = plt.subplots(
        nrows=EEG_TimeTable.shape[1],
        ncols=1,
        sharex=True,
        figsize=figsize
    )

    if EEG_TimeTable.shape[1] == 1:
        axes = [axes]

    for i, (ax, channel) in enumerate(zip(axes, EEG_TimeTable.columns)):

        # Select color
        if isinstance(color, list):
            plot_color = color[i] if i < len(color) else None
        else:
            plot_color = color

        ax.plot(
            EEG_TimeTable.index,
            EEG_TimeTable[channel],
            color=plot_color
        )

        ax.set_ylabel(channel)

        if y_limit is not None:
            ax.set_ylim(y_limit)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()
#=================================================================================
#=================================================================================
#=================================================================================
# FUNCTION #8

def bandpass_filter_eegwin_1_8(
    EEG_win: pd.DataFrame,
    lowcut: float = 0.5,
    highcut: float = 48.0,
    order: int = 4,
    check_nans: bool = True,
    notch_freq: float | None = None,
    notch_Q: float = 10.0
):
    """
    Band-pass robusto usando SOS + sosfiltfilt (fase cero).

    EEG_win:
      - index: tiempo en segundos (numérico, creciente)
      - columns: canales
      - values: amplitud
    """

    # --- 1) Inferir fs desde el índice ---
    t = EEG_win.index.to_numpy(dtype=float)
    if t.size < 3:
        raise ValueError("Muy pocas muestras para inferir fs y filtrar (necesitas >= 3).")

    dt = np.median(np.diff(t))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("El índice de tiempo debe ser numérico, finito y estrictamente creciente.")

    fs = 1.0 / dt
    nyq = fs / 2.0

    # --- 2) Validaciones de cortes ---
    if lowcut <= 0:
        raise ValueError("lowcut debe ser > 0 Hz.")
    if highcut >= nyq:
        raise ValueError(f"highcut ({highcut} Hz) debe ser < Nyquist ({nyq:.2f} Hz).")
    if lowcut >= highcut:
        raise ValueError("lowcut debe ser < highcut.")
    if notch_freq is not None and notch_freq >= nyq:
        
        raise ValueError(f"notch_freq ({notch_freq} Hz) debe ser < Nyquist ({nyq:.2f} Hz).")
    # --- 3) NaNs ---
    if check_nans and EEG_win.isna().any().any():
        raise ValueError("EEG_win contiene NaNs. Rellena/interpola antes de filtrar.")

    # --- 4) Diseñar filtro ---
    low  = lowcut / nyq
    high = highcut / nyq
    sos_band = butter(order, [low, high], btype="band", output="sos")
    sos_notch = None
    if notch_freq is not None:
        w0 = notch_freq / nyq
        b, a = iirnotch(w0, notch_Q)
        sos_notch = tf2sos(b, a)

    # --- 5) Filtrar ---
    X = EEG_win.to_numpy(dtype=float)  # (n_samples, n_channels)
    try:
        Xf = sosfiltfilt(sos_band, X, axis=0)

        if sos_notch is not None:
            Xf = sosfiltfilt(sos_notch, Xf, axis=0)

    except ValueError as e:
        raise ValueError(
            f"No se pudo filtrar (posible ventana corta/padding). "
            f"Prueba una ventana más larga o baja el orden. Error: {e}"
        )
    EEG_win_filt = pd.DataFrame(Xf, index=EEG_win.index, columns=EEG_win.columns)
    return EEG_win_filt, fs
#=================================================================================
#=================================================================================
#=================================================================================
# FUNCTION #9

def full_recording_from_matfiles_1_9(
    input_dir: str,
    output_dir: str,
    files_to_process: list[str],
    df_matches: pd.DataFrame,              
    amp_threshold: float = 200.0,
    lowcut: float = 0.5,
    highcut: float = 48.0,
    order: int = 4,
    eps: float = 1e-8,
    save_format: str = "npz",
    do_zscore: bool = True,   
):

    """
    Processes full EEG .mat recordings (no windowing) and saves the entire
    z-scored recording.
    
    The output .npz file contains:
    
      X:               (C, N) full z-scored signal (channels × samples)
      mu:              (C,)   mean per channel (computed over full recording)
      sigma:           (C,)   standard deviation per channel
      fs:              float  sampling rate
      channel_names:   (C,)   channel labels
      source_file:     (1,)   original .mat file name
      seizure_onsets:  (K,)   seizure onset timestamps (ISO format) associated
                               with this .mat file (K may be 0)
      T0:              (K,)   recording start timestamps (ISO format) from
                               df_matches (may repeat if multiple matches exist)
      TF:              (K,)   recording end timestamps (ISO format) from
                               df_matches (may repeat if multiple matches exist)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Normalizamos columnas clave (por si vienen como string)
    dfm = df_matches.copy()
    if "file" not in dfm.columns or "onset" not in dfm.columns:
        raise ValueError("df_matches must contain columns: ['file', 'onset']")
    if "T0" not in dfm.columns or "TF" not in dfm.columns:
        raise ValueError("df_matches must contain columns: ['T0', 'TF']")
        
    dfm["file"] = dfm["file"].astype(str)
    dfm["onset"] = pd.to_datetime(dfm["onset"], errors="coerce")

    for file_name in files_to_process:
        mat_path = os.path.join(input_dir, file_name)
        base_name = os.path.splitext(file_name)[0]

        if not os.path.exists(mat_path):
            print(f"File not found, skipping: {mat_path}")
            continue

        try:
            print(f"\n--- Processing file: {file_name} ---")

            #  0) Obtener onsets asociados a ESTE mat file
            onsets = (
                dfm.loc[dfm["file"] == file_name, "onset"]
                .dropna()
                .sort_values()
            )

            # Guardar en formato ISO (fácil de leer y reproducible)
            seizure_onsets_iso = onsets.dt.strftime("%Y-%m-%d %H:%M:%S.%f").to_numpy(dtype=object)
            #  0b) Extract T0 and TF associated with THIS .mat file
            df_file = dfm.loc[dfm["file"] == file_name].copy()
            
            # Ensure datetime format
            df_file["T0"] = pd.to_datetime(df_file["T0"], errors="coerce")
            df_file["TF"] = pd.to_datetime(df_file["TF"], errors="coerce")
            
            # Convert to ISO string format (will repeat if multiple matches exist)
            t0_iso = df_file["T0"].dt.strftime("%Y-%m-%d %H:%M:%S.%f").to_numpy(dtype=object)
            tf_iso = df_file["TF"].dt.strftime("%Y-%m-%d %H:%M:%S.%f").to_numpy(dtype=object)
            # 1) Load MAT
            mat_contents = sio.loadmat(mat_path)
            header_dict = mat_contents["hdr"]

            # 2) Convert to DataFrame (Time + channels)
            _, _, df_eeg = build_eeg_array_from_mat_1_6(
                hdr=header_dict,
                mat_data=mat_contents,
                output_dir=output_dir,
                file_prefix=base_name,
                save_format=save_format,
                return_dataframe=True
            )

            channel_cols = [c for c in df_eeg.columns if c != "Time"]

            # 3) Amplitude cutoff
            df_cutoff = apply_amplitude_cutoff_1_5(
                df_eeg,
                threshold=amp_threshold,
                start_sec=float(df_eeg["Time"].min()),
                end_sec=float(df_eeg["Time"].max())
            )

            # 4) Bandpass filter
            df_cutoff_idx = df_cutoff.set_index("Time")
            df_filtered_idx, fs = bandpass_filter_eegwin_1_8(
                df_cutoff_idx, lowcut=lowcut, highcut=highcut, order=order
            )
            df_filtered = df_filtered_idx.reset_index()

         # 5) Convert to numpy (C, N)
            arr = df_filtered[channel_cols].to_numpy(dtype=np.float32).T
        
            # stats (siempre)
            mu = arr.mean(axis=1, keepdims=True)      # (C,1)
            sigma = arr.std(axis=1, keepdims=True)    # (C,1)
            sigma = np.where(sigma < eps, eps, sigma)
        
            if do_zscore:
                X = (arr - mu) / sigma
                suffix = "zscore_full"
            else:
                X = arr
                suffix = "preproc_full"
        
            out_path = os.path.join(output_dir, f"{base_name}_{suffix}.npz")
        
            np.savez_compressed(
                out_path,
                X=X,
                mu=mu.squeeze(1),
                sigma=sigma.squeeze(1),
                fs=float(fs),
                channel_names=np.array(channel_cols, dtype=object),
                source_file=np.array([file_name], dtype=object),
                seizure_onsets=seizure_onsets_iso,
                T0=t0_iso,
                TF=tf_iso,
            )

        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    print("\nAll specified files processed!")

def full_recording_from_matfiles_1_9_V2(
    input_dir: str,
    output_dir: str,
    files_to_process: list[str],
    df_matches: pd.DataFrame,              
    amp_threshold: float = 200.0,
    lowcut: float = 0.5,
    highcut: float = 48.0,
    order: int = 4,
    eps: float = 1e-8,
    save_format: str = "npz",
    do_zscore: bool = True,   
    notch_freq: float | None = None,
    notch_Q: float = 30.0,

):

    """
    Processes full EEG .mat recordings (no windowing) and saves the entire
    z-scored recording.
    
    The output .npz file contains:
    
      X:               (C, N) full z-scored signal (channels × samples). if the option is selected, if not is going to store the normal data
      mu:              (C,)   mean per channel (computed over full recording)
      sigma:           (C,)   standard deviation per channel
      fs:              float  sampling rate
      channel_names:   (C,)   channel labels
      source_file:     (1,)   original .mat file name
      seizure_onsets:  (K,)   seizure onset timestamps (ISO format) associated
                               with this .mat file (K may be 0)
      T0:              (K,)   recording start timestamps (ISO format) from
                               df_matches (may repeat if multiple matches exist)
      TF:              (K,)   recording end timestamps (ISO format) from
                               df_matches (may repeat if multiple matches exist)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Normalizamos columnas clave (por si vienen como string)
    dfm = df_matches.copy()
    if "file" not in dfm.columns or "onset" not in dfm.columns:
        raise ValueError("df_matches must contain columns: ['file', 'onset']")
    if "T0" not in dfm.columns or "TF" not in dfm.columns:
        raise ValueError("df_matches must contain columns: ['T0', 'TF']")
        
    dfm["file"] = dfm["file"].astype(str)
    dfm["onset"] = pd.to_datetime(dfm["onset"], errors="coerce")

    for file_name in files_to_process:
        mat_path = os.path.join(input_dir, file_name)
        base_name = os.path.splitext(file_name)[0]

        if not os.path.exists(mat_path):
            print(f"File not found, skipping: {mat_path}")
            continue

        try:
            print(f"\n--- Processing file: {file_name} ---")


            # ✅ 0) Extraer filas asociadas a ESTE mat file
            df_file = dfm.loc[dfm["file"] == file_name].copy()
            
            # Ensure datetime format
            df_file["onset"] = pd.to_datetime(df_file["onset"], errors="coerce")
            df_file["T0"] = pd.to_datetime(df_file["T0"], errors="coerce")
            df_file["TF"] = pd.to_datetime(df_file["TF"], errors="coerce")
            
            # Caso 1: el archivo NI SIQUIERA está en df_matches
            if df_file.empty:
                seizure_onsets_iso = np.array([np.nan], dtype=object)
                t0_iso = np.array([np.nan], dtype=object)
                tf_iso = np.array([np.nan], dtype=object)
            
            else:
                # onsets válidos
                onsets = df_file["onset"].dropna().sort_values()
            
                # Si no hay onset válido, guardar NaN
                if onsets.empty:
                    seizure_onsets_iso = np.array([np.nan], dtype=object)
                else:
                    seizure_onsets_iso = onsets.dt.strftime("%Y-%m-%d %H:%M:%S.%f").to_numpy(dtype=object)
            
                # T0 / TF
                t0_iso = df_file["T0"].dt.strftime("%Y-%m-%d %H:%M:%S.%f").to_numpy(dtype=object)
                tf_iso = df_file["TF"].dt.strftime("%Y-%m-%d %H:%M:%S.%f").to_numpy(dtype=object)
            
                # si por alguna razón quedan vacíos
                if t0_iso.size == 0:
                    t0_iso = np.array([np.nan], dtype=object)
                if tf_iso.size == 0:
                    tf_iso = np.array([np.nan], dtype=object)
            # 1) Load MAT
            mat_contents = sio.loadmat(mat_path)
            header_dict = mat_contents["hdr"]

            # 2) Convert to DataFrame (Time + channels)
            _, _, df_eeg = build_eeg_array_from_mat_1_6(
                hdr=header_dict,
                mat_data=mat_contents,
                output_dir=output_dir,
                file_prefix=base_name,
                save_format=save_format,
                return_dataframe=True,
                save=False
            )

            channel_cols = [c for c in df_eeg.columns if c != "Time"]

            # 3) Amplitude cutoff
            df_cutoff = apply_amplitude_cutoff_1_5(
                df_eeg,
                threshold=amp_threshold,
                start_sec=float(df_eeg["Time"].min()),
                end_sec=float(df_eeg["Time"].max())
            )

            # 4) Bandpass filter
            df_cutoff_idx = df_cutoff.set_index("Time")
            df_filtered_idx, fs = bandpass_filter_eegwin_1_8(
                df_cutoff_idx,
                lowcut=lowcut,
                highcut=highcut,
                order=order,
                notch_freq=notch_freq,
                notch_Q=notch_Q
            )
            df_filtered = df_filtered_idx.reset_index()

         # 5) Convert to numpy (C, N)
            arr = df_filtered[channel_cols].to_numpy(dtype=np.float32).T
        
            # stats (siempre)
            mu = arr.mean(axis=1, keepdims=True)      # (C,1)
            sigma = arr.std(axis=1, keepdims=True)    # (C,1)
            sigma = np.where(sigma < eps, eps, sigma)
        
            if do_zscore:
                X = (arr - mu) / sigma
                suffix = "zscore_full"
            else:
                X = arr
                suffix = "preproc_full"
        
            out_path = os.path.join(output_dir, f"{base_name}_{suffix}.npz")
        
            np.savez_compressed(
                out_path,
                X=X,
                mu=mu.squeeze(1),
                sigma=sigma.squeeze(1),
                fs=float(fs),
                channel_names=np.array(channel_cols, dtype=object),
                source_file=np.array([file_name], dtype=object),
                seizure_onsets=seizure_onsets_iso,
                T0=t0_iso,
                TF=tf_iso,
            )

        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    print("\nAll specified files processed!")


#=================================================================================
#=================================================================================
#=================================================================================
# FUNCTION #10

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import re
import pandas as pd

def _parse_compact_datetime_str(dt_str: str) -> pd.Timestamp:
    """
    Robust parser for malformed datetime strings like:
      '2019-110107:43:13.000000'
      '2019-11010 7:43:13.000000'
    Returns pandas Timestamp or raises ValueError.
    """
    s = str(dt_str).strip()

    # Extract all digit groups (year, month, day, hour, min, sec, microsec)
    nums = re.findall(r"\d+", s)

    # Common patterns:
    # 1) ['2019', '110107', '43', '13', '000000']  -> need split month/day/hour
    # 2) ['2019', '11010', '7', '43', '13', '000000'] -> need split month/day
    # 3) ['2019', '11', '01', '07', '43', '13', '000000'] -> already fine-ish

    if len(nums) == 7:
        y, mo, d, h, mi, se, us = nums
    elif len(nums) == 6:
        y, mday, h, mi, se, us = nums
        # mday should be 4 digits: MMDD
        if len(mday) == 4:
            mo, d = mday[:2], mday[2:]
        else:
            raise ValueError(f"Cannot parse date part: {dt_str}")
    elif len(nums) == 5:
        y, mdayhour, mi, se, us = nums
        # mdayhour should be 6 digits: MMDDHH
        if len(mdayhour) == 6:
            mo, d, h = mdayhour[:2], mdayhour[2:4], mdayhour[4:]
        else:
            raise ValueError(f"Cannot parse date/time part: {dt_str}")
    else:
        raise ValueError(f"Unrecognized datetime format: {dt_str}")

    # Zero pad
    mo = mo.zfill(2)
    d  = d.zfill(2)
    h  = h.zfill(2)
    mi = mi.zfill(2)
    se = se.zfill(2)
    us = us.ljust(6, "0")[:6]  # ensure 6 digits

    fixed = f"{y}-{mo}-{d} {h}:{mi}:{se}.{us}"
    return pd.to_datetime(fixed, format="%Y-%m-%d %H:%M:%S.%f", errors="raise")
def visualize_seizure_windows_from_npz_1_10(
    npz_path: str,
    channel_idx: int = 0,
    window_sec: int = 10,
    n_windows: int = 5
):
    """
    Visualize consecutive EEG segments starting from each seizure onset.

    For each seizure:
        - Plots n_windows windows
        - Each window is window_sec long
        - Total duration = window_sec * n_windows

    Parameters
    ----------
    npz_path : str
        Path to preprocessed .npz file.
    channel_idx : int
        Channel index to visualize.
    window_sec : int
        Length of each window in seconds.
    n_windows : int
        Number of consecutive windows to plot.
    """

    data = np.load(npz_path, allow_pickle=True)

    X = data["X"]                     # (C, N)
    fs = float(data["fs"])
    seizure_onsets = data["seizure_onsets"]
    T0 = data["T0"][0]                # first T0 (they repeat)
    source_file = str(data["source_file"][0])
    if len(seizure_onsets) == 0:
        print("No seizures found in this file.")
        return
    if channel_idx < 0 or channel_idx >= X.shape[0]:
        
        raise ValueError(f"channel_idx={channel_idx} out of bounds for X with shape {X.shape}")

    T0_str = str(T0)

    # Fix common formatting bug: missing space between date and time
    # e.g. "2019-11-0107:43:13.000000" -> "2019-11-01 07:43:13.000000"
    if len(T0_str) >= 11 and T0_str[10] != " ":
        T0_str = T0_str[:10] + " " + T0_str[10:]
    
    T0_dt = _parse_compact_datetime_str(T0_str)

    window_samples = int(window_sec * fs)
    total_samples = window_samples * n_windows
    pdf_path = f"{source_file}_seizures.pdf"
    pdf = PdfPages(pdf_path)
    for s_idx, onset_str in enumerate(seizure_onsets):

        onset_dt = _parse_compact_datetime_str(onset_str)

        # Compute seizure position in samples
        delta_sec = (onset_dt - T0_dt).total_seconds()
        onset_sample = int(round(delta_sec * fs))

        end_sample = onset_sample + total_samples

        if onset_sample < 0 or end_sample > X.shape[1]:
            print(f"Seizure {s_idx}: out of bounds, skipping.")
            continue

        print(
            f"[{source_file}] Seizure {s_idx} | "
            f"Onset: {onset_dt} | "
            f"sample={onset_sample} | "
            f"t={delta_sec:.2f} s from T0"
)

        fig, axes = plt.subplots(
            n_windows,
            1,
            figsize=(12, 2*n_windows),
            sharex=False
        )

        for w in range(n_windows):

            start = onset_sample + w * window_samples
            end = start + window_samples

            segment = X[channel_idx, start:end]
            window_start_dt = onset_dt + pd.to_timedelta(w * window_sec, unit="s")
            window_end_dt   = window_start_dt + pd.to_timedelta(window_sec, unit="s")
            t_sec = np.arange(len(segment)) / fs
            axes[w].plot(t_sec, segment, linewidth=0.8)
            axes[w].set_xlim(0, window_sec)
            axes[w].set_xlabel("Time within window (s)")
            axes[w].set_ylabel("Amplitude")
            axes[w].set_title(
                f"Seizure {s_idx} | Onset: {onset_dt} | "
                f"Window {w+1}/{n_windows} | "
                f"{window_start_dt} to {window_end_dt}"
)
            axes[w].axhline(0, linestyle="--")

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    pdf.close()
    print(f"Saved PDF: {pdf_path}")
#visualize_seizure_windows_from_npz(
#    npz_path=file_path,
#    channel_idx=0,
#    window_sec=10,
#    n_windows=5
#)
#version 2
def visualize_seizure_windows_from_npz_1_10V2(
npz_path: str,
channel_idx_1: int = 0,
channel_idx_2: int = 1,
window_sec: int = 10,
n_windows: int = 5,
output_dir: str = "."
):

    """
    Visualize consecutive EEG segments starting from each seizure onset.

    For each seizure:
        - Plots n_windows windows
        - Each window is window_sec long
        - Total duration = window_sec * n_windows

    Parameters
    ----------
    npz_path : str
        Path to preprocessed .npz file.
    channel_idx : int
        Channel index to visualize.
    window_sec : int
        Length of each window in seconds.
    n_windows : int
        Number of consecutive windows to plot.
    """

    data = np.load(npz_path, allow_pickle=True)

    X = data["X"]                     # (C, N)
    fs = float(data["fs"])
    seizure_onsets = data["seizure_onsets"]
    T0 = data["T0"][0]                # first T0 (they repeat)
    source_file = str(data["source_file"][0])
    if len(seizure_onsets) == 0:
        print("No seizures found in this file.")
        return
    if channel_idx_1 < 0 or channel_idx_1 >= X.shape[0]:
        raise ValueError(f"channel_idx_1={channel_idx_1} out of bounds for X with shape {X.shape}")
    
    if channel_idx_2 < 0 or channel_idx_2 >= X.shape[0]:
        raise ValueError(f"channel_idx_2={channel_idx_2} out of bounds for X with shape {X.shape}")

    T0_str = str(T0)

    # Fix common formatting bug: missing space between date and time
    # e.g. "2019-11-0107:43:13.000000" -> "2019-11-01 07:43:13.000000"
    if len(T0_str) >= 11 and T0_str[10] != " ":
        T0_str = T0_str[:10] + " " + T0_str[10:]
    
    T0_dt = _parse_compact_datetime_str(T0_str)

    window_samples = int(window_sec * fs)
    total_samples = window_samples * n_windows
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(npz_path))[0]
    pdf_path = os.path.join(output_dir, f"{base_name}_seizures.pdf")
    pdf = PdfPages(pdf_path)
    for s_idx, onset_str in enumerate(seizure_onsets):

        onset_dt = _parse_compact_datetime_str(onset_str)

        # Compute seizure position in samples
        delta_sec = (onset_dt - T0_dt).total_seconds()
        onset_sample = int(round(delta_sec * fs))

        end_sample = onset_sample + total_samples

        if onset_sample < 0 or end_sample > X.shape[1]:
            print(f"Seizure {s_idx}: out of bounds, skipping.")
            continue

        print(
            f"[{source_file}] Seizure {s_idx} | "
            f"Onset: {onset_dt} | "
            f"sample={onset_sample} | "
            f"t={delta_sec:.2f} s from T0 | "
            f"channels=({channel_idx_1}, {channel_idx_2})"
        )

        fig, axes = plt.subplots(n_windows,2,figsize=(16, 2.5*n_windows),sharex=False)
        if n_windows == 1:
            axes = np.array([axes])
        for w in range(n_windows):

            start = onset_sample + w * window_samples
            end = start + window_samples

            segment_1 = X[channel_idx_1, start:end]
            segment_2 = X[channel_idx_2, start:end]
            window_start_dt = onset_dt + pd.to_timedelta(w * window_sec, unit="s")
            window_end_dt   = window_start_dt + pd.to_timedelta(window_sec, unit="s")
            t_sec = np.arange(len(segment_1)) / fs
            axes[w, 0].plot(t_sec, segment_1, color="blue", linewidth=0.8)
            axes[w, 0].set_xlim(0, window_sec)
            axes[w, 0].set_xlabel("Time within window (s)")
            axes[w, 0].set_ylabel("Amplitude")
            axes[w, 0].set_title(
                f"Seizure {s_idx} | Window {w+1}/{n_windows} | "
                f"Ch {channel_idx_1} | {window_start_dt} to {window_end_dt}"
            )
            axes[w, 0].axhline(0, linestyle="--", linewidth=0.8)
            
            axes[w, 1].plot(t_sec, segment_2, color="red", linewidth=0.8)
            axes[w, 1].set_xlim(0, window_sec)
            axes[w, 1].set_xlabel("Time within window (s)")
            axes[w, 1].set_ylabel("Amplitude")
            axes[w, 1].set_title(
                f"Seizure {s_idx} | Window {w+1}/{n_windows} | "
                f"Ch {channel_idx_2} | {window_start_dt} to {window_end_dt}"
            )
            axes[w, 1].axhline(0, linestyle="--", linewidth=0.8)
        fig.suptitle(f"{source_file} | Seizure {s_idx} | Onset: {onset_dt}",y=1.02,fontsize=12)
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    
    pdf.close()
    print(f"Saved PDF: {pdf_path}")
#visualize_seizure_windows_from_npz(
#    npz_path=file_path,
#    channel_idx=0,
#    window_sec=10,
#    n_windows=5
#)
    
# VERSION 3 WITH PRE-ICTAL
def visualize_seizure_windows_from_npz_1_10V3(
    npz_path: str,
    channel_idx_1: int = 0,
    channel_idx_2: int = 1,
    window_sec: int = 10,
    n_windows: int = 11,   
    pre_onset_sec: int = 60,
    vertical_offset_uv: float = 100.0,
    output_dir: str = "."
):
    """
    Visualiza segmentos consecutivos de EEG alrededor de cada seizure onset.

    Modificaciones:
    - Ambos canales se plotean en el mismo subplot por ventana
    - channel_idx_2 se desplaza verticalmente +vertical_offset_uv
    - El ploteo empieza pre_onset_sec antes del onset
    - Se agrega sombreado amarillo suave de 2 s desde el onset
    """

    data = np.load(npz_path, allow_pickle=True)

    X = data["X"]                     # shape: (C, N)
    fs = float(data["fs"])
    seizure_onsets = data["seizure_onsets"]
    T0 = data["T0"][0]
    source_file = str(data["source_file"][0])

    # --- limpiar onsets inválidos ---
    seizure_onsets_clean = []
    
    for s in seizure_onsets:
        if s is None:
            continue
        s_str = str(s).strip().lower()
        if s_str == "nan" or s_str == "":
            continue
        seizure_onsets_clean.append(s)
    
    # si no hay seizures válidos → omitir archivo
    if len(seizure_onsets_clean) == 0:
        print(f"{os.path.basename(npz_path)} → no seizures, skipping.")
        return

    if channel_idx_1 < 0 or channel_idx_1 >= X.shape[0]:
        raise ValueError(f"channel_idx_1={channel_idx_1} out of bounds for X with shape {X.shape}")

    if channel_idx_2 < 0 or channel_idx_2 >= X.shape[0]:
        raise ValueError(f"channel_idx_2={channel_idx_2} out of bounds for X with shape {X.shape}")

    T0_str = str(T0)

    # Corrige formato tipo "2019-11-0107:43:13.000000"
    if len(T0_str) >= 11 and T0_str[10] != " ":
        T0_str = T0_str[:10] + " " + T0_str[10:]

    T0_dt = _parse_compact_datetime_str(T0_str)

    window_samples = int(window_sec * fs)
    pre_onset_samples = int(pre_onset_sec * fs)
    total_samples = window_samples * n_windows

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(npz_path))[0]
    pdf_path = os.path.join(output_dir, f"{base_name}_seizures_overlay.pdf")
    pdf = PdfPages(pdf_path)

    for s_idx, onset_str in enumerate(seizure_onsets_clean):
        onset_dt = _parse_compact_datetime_str(onset_str)

        # posición del onset en samples
        delta_sec = (onset_dt - T0_dt).total_seconds()
        onset_sample = int(round(delta_sec * fs))

        # arrancar pre_onset_sec antes del onset
        plot_start_sample = onset_sample - pre_onset_samples
        plot_end_sample = plot_start_sample + total_samples

        if plot_start_sample < 0 or plot_end_sample > X.shape[1]:
            print(f"Seizure {s_idx}: out of bounds, skipping.")
            continue

        print(
            f"[{source_file}] Seizure {s_idx} | "
            f"Onset: {onset_dt} | "
            f"onset_sample={onset_sample} | "
            f"plot_start={plot_start_sample} | "
            f"channels=({channel_idx_1}, {channel_idx_2})"
        )

        fig, axes = plt.subplots(n_windows, 1, figsize=(16, 2.8 * n_windows), sharex=False)

        if n_windows == 1:
            axes = np.array([axes])

        for w in range(n_windows):
            start = plot_start_sample + w * window_samples
            end = start + window_samples

            segment_1 = X[channel_idx_1, start:end]
            segment_2 = X[channel_idx_2, start:end] + vertical_offset_uv

            # tiempo relativo dentro de la ventana
            t_sec = np.arange(len(segment_1)) / fs

            # tiempo absoluto de inicio/fin de la ventana
            window_start_dt = T0_dt + pd.to_timedelta(start / fs, unit="s")
            window_end_dt   = T0_dt + pd.to_timedelta(end / fs, unit="s")

            # tiempo relativo al onset
            rel_start_sec = (start - onset_sample) / fs
            rel_end_sec   = (end - onset_sample) / fs

            ax = axes[w]

            # sombreado amarillo suave de 2 s desde el onset, si cae en esta ventana
            if rel_start_sec <= 0 < rel_end_sec:
                onset_in_window_sec = -rel_start_sec
                shade_end = min(onset_in_window_sec + 2.0, window_sec)

                ax.axvspan(
                    onset_in_window_sec,
                    shade_end,
                    color="gold",
                    alpha=0.22,
                    zorder=0,
                    label="Onset + 2 s"
                )

                ax.axvline(
                    onset_in_window_sec,
                    color="black",
                    linestyle="--",
                    linewidth=1.0,
                    label="Onset"
                )

            ax.plot(t_sec, segment_1, color="blue", linewidth=0.8, label=f"Ch {channel_idx_1}")
            ax.plot(t_sec, segment_2, color="red", linewidth=0.8, label=f"Ch {channel_idx_2} (+{vertical_offset_uv:.0f} µV)")

            # líneas guía
            ax.axhline(0, color="blue", linestyle="--", linewidth=0.6, alpha=0.7)
            ax.axhline(vertical_offset_uv, color="red", linestyle="--", linewidth=0.6, alpha=0.7)

            ax.set_xlim(0, window_sec)
            ax.set_xlabel("Time within window (s)")
            ax.set_ylabel("Amplitude (µV)")
            ax.set_title(
                f"Seizure {s_idx} | Window {w+1}/{n_windows} | "
                f"{window_start_dt} to {window_end_dt} | "
                f"rel. to onset: {rel_start_sec:.1f}s to {rel_end_sec:.1f}s"
            )

            if w == 0:
                ax.legend(loc="upper right", fontsize=8)

        fig.suptitle(
            f"{source_file} | Seizure {s_idx} | Onset: {onset_dt} | "
            f"Start plotting {pre_onset_sec}s before onset",
            y=1.02,
            fontsize=12
        )

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    pdf.close()
    print(f"Saved PDF: {pdf_path}")
def visualize_seizure_windows_from_npz_1_10VNormal(
    npz_path: str,
    channel_idx_1: int = 0,
    channel_idx_2: int = 1,
    window_sec: int = 10,
    n_windows: int = 11,   
    pre_onset_sec: int = 60,
    vertical_offset_uv: float = 20,
    output_dir: str = "."
):
    """
    Visualiza segmentos consecutivos de EEG alrededor de cada seizure onset.

    Modificaciones:
    - Ambos canales se plotean en el mismo subplot por ventana
    - channel_idx_2 se desplaza verticalmente +vertical_offset_uv
    - El ploteo empieza pre_onset_sec antes del onset
    - Se agrega sombreado amarillo suave de 2 s desde el onset
    """

    data = np.load(npz_path, allow_pickle=True)

    X = data["X"]                     # shape: (C, N)
    fs = float(data["fs"])
    seizure_onsets = data["seizure_onsets"]
    T0 = data["T0"][0]
    source_file = str(data["source_file"][0])

    # --- limpiar onsets inválidos ---
    seizure_onsets_clean = []
    
    for s in seizure_onsets:
        if s is None:
            continue
        s_str = str(s).strip().lower()
        if s_str == "nan" or s_str == "":
            continue
        seizure_onsets_clean.append(s)
    
    # si no hay seizures válidos → omitir archivo
    if len(seizure_onsets_clean) == 0:
        print(f"{os.path.basename(npz_path)} → no seizures, skipping.")
        return

    if channel_idx_1 < 0 or channel_idx_1 >= X.shape[0]:
        raise ValueError(f"channel_idx_1={channel_idx_1} out of bounds for X with shape {X.shape}")

    if channel_idx_2 < 0 or channel_idx_2 >= X.shape[0]:
        raise ValueError(f"channel_idx_2={channel_idx_2} out of bounds for X with shape {X.shape}")

    T0_str = str(T0)

    # Corrige formato tipo "2019-11-0107:43:13.000000"
    if len(T0_str) >= 11 and T0_str[10] != " ":
        T0_str = T0_str[:10] + " " + T0_str[10:]

    T0_dt = _parse_compact_datetime_str(T0_str)

    window_samples = int(window_sec * fs)
    pre_onset_samples = int(pre_onset_sec * fs)
    total_samples = window_samples * n_windows

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(npz_path))[0]
    pdf_path = os.path.join(output_dir, f"{base_name}_seizures_overlay.pdf")
    pdf = PdfPages(pdf_path)

    for s_idx, onset_str in enumerate(seizure_onsets_clean):
        onset_dt = _parse_compact_datetime_str(onset_str)

        # posición del onset en samples
        delta_sec = (onset_dt - T0_dt).total_seconds()
        onset_sample = int(round(delta_sec * fs))

        # arrancar pre_onset_sec antes del onset
        plot_start_sample = onset_sample - pre_onset_samples
        plot_end_sample = plot_start_sample + total_samples

        if plot_start_sample < 0 or plot_end_sample > X.shape[1]:
            print(f"Seizure {s_idx}: out of bounds, skipping.")
            continue

        print(
            f"[{source_file}] Seizure {s_idx} | "
            f"Onset: {onset_dt} | "
            f"onset_sample={onset_sample} | "
            f"plot_start={plot_start_sample} | "
            f"channels=({channel_idx_1}, {channel_idx_2})"
        )

        fig, axes = plt.subplots(n_windows, 1, figsize=(16, 2.8 * n_windows), sharex=False)

        if n_windows == 1:
            axes = np.array([axes])

        for w in range(n_windows):
            start = plot_start_sample + w * window_samples
            end = start + window_samples

            segment_1 = X[channel_idx_1, start:end]
            segment_2 = X[channel_idx_2, start:end] + vertical_offset_uv

            # tiempo relativo dentro de la ventana
            t_sec = np.arange(len(segment_1)) / fs

            # tiempo absoluto de inicio/fin de la ventana
            window_start_dt = T0_dt + pd.to_timedelta(start / fs, unit="s")
            window_end_dt   = T0_dt + pd.to_timedelta(end / fs, unit="s")

            # tiempo relativo al onset
            rel_start_sec = (start - onset_sample) / fs
            rel_end_sec   = (end - onset_sample) / fs

            ax = axes[w]

            # sombreado amarillo suave de 2 s desde el onset, si cae en esta ventana
            # Mark the exact seizure onset time if it falls inside this window
            onset_in_window_sec = (onset_dt - window_start_dt).total_seconds()
            
            if 0 <= onset_in_window_sec <= window_sec:
            
                shade_end = min(onset_in_window_sec + 2.0, window_sec)
            
                ax.axvspan(
                    onset_in_window_sec,
                    shade_end,
                    color="gold",
                    alpha=0.22,
                    zorder=0,
                    label="Onset + 2 s"
                )
            
                ax.axvline(
                    onset_in_window_sec,
                    color="red",
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.9,
                    label=f"Onset at {onset_in_window_sec:.2f}s"
                )
            ax.plot(t_sec, segment_1, color="blue", linewidth=0.8, label=f"Ch {channel_idx_1}")
            ax.plot(t_sec, segment_2, color="red", linewidth=0.8, label=f"Ch {channel_idx_2} (+{vertical_offset_uv:.0f} z-score)")

            # líneas guía
            ax.axhline(0, color="blue", linestyle="--", linewidth=0.6, alpha=0.7)
            ax.axhline(vertical_offset_uv, color="red", linestyle="--", linewidth=0.6, alpha=0.7)

            ax.set_xlim(0, window_sec)
            ax.set_xlabel("Time within window (s)")
            ax.set_ylabel("Z-score")
            ax.set_title(
                f"Seizure {s_idx} | Window {w+1}/{n_windows} | "
                f"{window_start_dt} to {window_end_dt} | "
                f"rel. to onset: {rel_start_sec:.1f}s to {rel_end_sec:.1f}s"
            )

            if w == 0:
                ax.legend(loc="upper right", fontsize=8)

        fig.suptitle(
            f"{source_file} | Seizure {s_idx} | Onset: {onset_dt} | "
            f"Start plotting {pre_onset_sec}s before onset",
            y=1.02,
            fontsize=12
        )

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    pdf.close()
    print(f"Saved PDF: {pdf_path}")
#=================================================================================
#=================================================================================
#=================================================================================
# FUNCTION #11


def plot_welch_overlay_from_npz_1_11(npz_path, nperseg_seconds=4, max_freq=120):
    # 1. Open the npz file
    data = np.load(npz_path, allow_pickle=True)

    # 2. Extract the stored arrays
    X = data["X"]                      # shape: (C, N)
    fs = float(data["fs"])             # sampling frequency
    channel_names = data["channel_names"]

    # 3. Convert Welch window length from seconds to samples
    nperseg = int(fs * nperseg_seconds)

    # 4. Create one figure for all channels
    plt.figure(figsize=(9, 5))

    # 5. Loop over channels
    n_channels = X.shape[0]

    for i in range(n_channels):
        # Extract one channel
        signal_1d = X[i, :]

        # Compute Welch PSD
        frequencies, power = welch(signal_1d, fs=fs, nperseg=nperseg)

        # Plot this channel
        plt.semilogy(frequencies, power, label=str(channel_names[i]))

    # 6. Add filter cutoff reference lines
    plt.axvline(0.5, linestyle="--", label="0.5 Hz cutoff")
    plt.axvline(40, linestyle="--", label="40 Hz cutoff")

    # 7. Limit x-axis to desired range or Nyquist frequency
    plt.xlim(0, min(max_freq, fs / 2))

    # 8. Labels and layout
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD")
    plt.title("Welch PSD of preprocessed EEG")
    plt.legend()
    plt.tight_layout()
    plt.show()
#=================================================================================
#=================================================================================
#=================================================================================
# FUNCTION #12
# PSD graph visualizer

def plot_psd_from_npz_1_12(
    npz_path_1: str,
    npz_path_2: str | None = None,
    channel: int = 0,
    nperseg_sec: float = 4.0,
    overlap_frac: float = 0.5,
    xlim: tuple[float, float] | None = None,
    peak_band: tuple[float, float] | None = None,
    top_n_peaks: int = 1,
    use_semilogy: bool = True,
    figsize: tuple[int, int] = (10, 5),
    strict_checks: bool = True,
):
    """
    Plot Welch PSD from one or two EEG NPZ files.

    Supports:
    - raw NPZ with keys like: signal (N, C), fs, channels
    - processed NPZ with keys like: X (C, N), fs, channel_names

    Parameters
    ----------
    npz_path_1 : str
        Path to first NPZ file.
    npz_path_2 : str | None
        Optional path to second NPZ file for comparison.
    channel : int
        Channel index to analyze.
    nperseg_sec : float
        Welch window length in seconds.
    overlap_frac : float
        Fractional overlap between windows (e.g. 0.5 = 50%).
    xlim : tuple[float, float] | None
        Frequency range to display, e.g. (0, 80) or (25, 45).
        If None, defaults to full available range.
    peak_band : tuple[float, float] | None
        Frequency band in which to search for peaks, e.g. (30, 40).
    top_n_peaks : int
        Number of top peaks to report inside peak_band.
    use_semilogy : bool
        If True, plot PSD on log scale.
    figsize : tuple[int, int]
        Figure size.
    strict_checks : bool
        If True, enforce equal fs / channels / samples when comparing two NPZs.

    Returns
    -------
    results : dict
        Dictionary with PSDs, frequencies, metadata, and detected peaks.
    """

    def _load_npz_signal(npz_path: str):
        data = np.load(npz_path, allow_pickle=True)

        # Detect signal key and orientation
        if "signal" in data:
            sig = data["signal"]  # usually (N, C)
            if sig.ndim != 2:
                raise ValueError(f"'signal' in {npz_path} is not 2D.")
            # convert to (C, N)
            if sig.shape[0] > sig.shape[1]:
                # probably (N, C)
                sig = sig.T
            signal_key = "signal"

            channel_names = data["channels"] if "channels" in data else np.array(
                [f"ch_{i}" for i in range(sig.shape[0])]
            )

        elif "X" in data:
            sig = data["X"]  # usually (C, N)
            if sig.ndim != 2:
                raise ValueError(f"'X' in {npz_path} is not 2D.")
            # keep as (C, N), but if suspicious shape, try transpose
            if sig.shape[0] > sig.shape[1]:
                # unlikely for EEG channels > samples, but guard anyway
                pass
            signal_key = "X"

            channel_names = data["channel_names"] if "channel_names" in data else np.array(
                [f"ch_{i}" for i in range(sig.shape[0])]
            )
        else:
            raise KeyError(
                f"Could not find signal array in {npz_path}. Expected 'signal' or 'X'."
            )

        if "fs" not in data:
            raise KeyError(f"'fs' not found in {npz_path}.")

        fs = float(data["fs"])

        if channel >= sig.shape[0]:
            raise IndexError(
                f"Requested channel={channel}, but file {npz_path} has only {sig.shape[0]} channels."
            )

        channel_names = np.array(channel_names)
        return {
            "path": npz_path,
            "data": data,
            "signal": sig,
            "signal_key": signal_key,
            "fs": fs,
            "channel_names": channel_names,
            "n_channels": sig.shape[0],
            "n_samples": sig.shape[1],
        }

    def _compute_welch(sig_1d: np.ndarray, fs: float, nperseg_sec: float, overlap_frac: float):
        nperseg = max(8, int(fs * nperseg_sec))
        noverlap = int(nperseg * overlap_frac)

        if noverlap >= nperseg:
            raise ValueError("overlap_frac produces noverlap >= nperseg.")

        # If signal is shorter than nperseg, shrink nperseg
        if len(sig_1d) < nperseg:
            nperseg = len(sig_1d)
            noverlap = int(nperseg * overlap_frac)
            if noverlap >= nperseg and nperseg > 1:
                noverlap = nperseg - 1

        f, pxx = welch(
            sig_1d,
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
            scaling="density"
        )
        return f, pxx, nperseg, noverlap

    def _find_top_peaks_in_band(f: np.ndarray, pxx: np.ndarray, band: tuple[float, float], top_n: int):
        fmin, fmax = band
        mask = (f >= fmin) & (f <= fmax)

        if not np.any(mask):
            return []

        f_band = f[mask]
        pxx_band = pxx[mask]

        if len(f_band) == 0:
            return []

        idx_sorted = np.argsort(pxx_band)[::-1]
        peaks = []
        used_freqs = set()

        for idx in idx_sorted:
            freq = float(f_band[idx])
            power = float(pxx_band[idx])

            # simple guard against returning the exact same bin twice
            if freq in used_freqs:
                continue

            peaks.append({"freq_hz": freq, "power": power})
            used_freqs.add(freq)

            if len(peaks) >= top_n:
                break

        return peaks

    # -------------------------
    # Load first file
    # -------------------------
    d1 = _load_npz_signal(npz_path_1)
    sig1 = d1["signal"][channel, :]
    fs1 = d1["fs"]

    f1, pxx1, nperseg1, noverlap1 = _compute_welch(sig1, fs1, nperseg_sec, overlap_frac)

    # -------------------------
    # Optional second file
    # -------------------------
    d2 = None
    f2 = pxx2 = None

    if npz_path_2 is not None:
        d2 = _load_npz_signal(npz_path_2)
        sig2 = d2["signal"][channel, :]
        fs2 = d2["fs"]

        if strict_checks:
            if not np.isclose(fs1, fs2):
                raise ValueError(f"Sampling frequency mismatch: fs1={fs1}, fs2={fs2}")
            if d1["n_channels"] != d2["n_channels"]:
                raise ValueError(
                    f"Channel count mismatch: {d1['n_channels']} vs {d2['n_channels']}"
                )
            if d1["n_samples"] != d2["n_samples"]:
                raise ValueError(
                    f"Sample count mismatch: {d1['n_samples']} vs {d2['n_samples']}"
                )

        f2, pxx2, nperseg2, noverlap2 = _compute_welch(sig2, fs2, nperseg_sec, overlap_frac)

    # -------------------------
    # Peaks
    # -------------------------
    peaks_1 = []
    peaks_2 = []

    if peak_band is not None:
        peaks_1 = _find_top_peaks_in_band(f1, pxx1, peak_band, top_n_peaks)
        if d2 is not None:
            peaks_2 = _find_top_peaks_in_band(f2, pxx2, peak_band, top_n_peaks)

    # -------------------------
    # Plot limits
    # -------------------------
    nyquist_1 = fs1 / 2
    nyquist_2 = d2["fs"] / 2 if d2 is not None else None
    max_possible_freq = min([x for x in [nyquist_1, nyquist_2] if x is not None])

    if xlim is None:
        plot_xlim = (0, max_possible_freq)
    else:
        xmin, xmax = xlim
        plot_xlim = (xmin, min(xmax, max_possible_freq))
        if xmax > max_possible_freq:
            print(
                f"[INFO] Requested xmax={xmax:.2f} Hz, but max possible is {max_possible_freq:.2f} Hz "
                f"(Nyquist). Using xmax={plot_xlim[1]:.2f} Hz instead."
            )

    # -------------------------
    # Plot
    # -------------------------
    plt.figure(figsize=figsize)

    label1 = f"{d1['path'].split('/')[-1]} | ch={d1['channel_names'][channel]}"
    if use_semilogy:
        plt.semilogy(f1, pxx1, label=label1)
    else:
        plt.plot(f1, pxx1, label=label1)

    if d2 is not None:
        label2 = f"{d2['path'].split('/')[-1]} | ch={d2['channel_names'][channel]}"
        if use_semilogy:
            plt.semilogy(f2, pxx2, label=label2)
        else:
            plt.plot(f2, pxx2, label=label2)

    # Mark peaks
    if peak_band is not None:
        for i, pk in enumerate(peaks_1, start=1):
            plt.axvline(pk["freq_hz"], linestyle="--", alpha=0.7)
            print(f"[NPZ1] Peak {i} in band {peak_band}: {pk['freq_hz']:.4f} Hz | PSD={pk['power']:.6e}")

        for i, pk in enumerate(peaks_2, start=1):
            plt.axvline(pk["freq_hz"], linestyle=":", alpha=0.7)
            print(f"[NPZ2] Peak {i} in band {peak_band}: {pk['freq_hz']:.4f} Hz | PSD={pk['power']:.6e}")

    plt.xlim(*plot_xlim)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD")
    plt.title(f"Welch PSD - channel {channel}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # -------------------------
    # Summary prints
    # -------------------------
    print("\n--- FILE 1 INFO ---")
    print("path:", d1["path"])
    print("signal key:", d1["signal_key"])
    print("signal shape (C, N):", d1["signal"].shape)
    print("fs:", d1["fs"])
    print("channel names:", d1["channel_names"])
    print("nperseg:", nperseg1)
    print("noverlap:", noverlap1)
    print("nyquist:", nyquist_1)

    if d2 is not None:
        print("\n--- FILE 2 INFO ---")
        print("path:", d2["path"])
        print("signal key:", d2["signal_key"])
        print("signal shape (C, N):", d2["signal"].shape)
        print("fs:", d2["fs"])
        print("channel names:", d2["channel_names"])
        print("nperseg:", nperseg2)
        print("noverlap:", noverlap2)
        print("nyquist:", nyquist_2)

    return {
        "file1": {
            "path": d1["path"],
},
        "file2": None if d2 is None else {
            "path": d2["path"],
            "fs": d2["fs"],
            "channel_names": d2["channel_names"],
            "f": f2,
            "pxx": pxx2,
            "peaks_in_band": peaks_2,
        },
    }
#=================================================================================
#=================================================================================
#=================================================================================
# FUNCTION #13


def analyze_peak_in_npz_folder_1_13(
    input_dir: str,
    pattern: str = "*.npz",
    band: tuple[float, float] = (30.0, 40.0),
    target_freq: float = 35.0,
    tolerance_hz: float = 1.0,
    nperseg_sec: float = 20.0,
    overlap_frac: float = 0.5,
    baseline_exclusion_hz: float = 1.0,
    ratio_threshold: float = 3.0,
    save_csv_path: str | None = None,
):
    """
    Analyze whether NPZ files show a stable high PSD peak near 35 Hz.

    Supported NPZ formats
    ---------------------
    Raw-style:
        signal : (N, C)
        fs
        channels
    Processed-style:
        X : (C, N)
        fs
        channel_names

    Parameters
    ----------
    input_dir : str
        Folder containing NPZ files.
    pattern : str
        Glob pattern for files.
    band : tuple
        Frequency band in which to search peaks, e.g. (30, 40).
    target_freq : float
        Frequency of interest, e.g. 35 Hz.
    tolerance_hz : float
        Peak considered "near target" if abs(peak_freq - target_freq) <= tolerance_hz.
    nperseg_sec : float
        Welch window length in seconds.
    overlap_frac : float
        Welch overlap fraction.
    baseline_exclusion_hz : float
        Exclude ± this range around peak when computing local baseline.
    ratio_threshold : float
        Threshold for flagging "high peak" based on peak/local_baseline ratio.
    save_csv_path : str | None
        Optional path to save results CSV.

    Returns
    -------
    df : pd.DataFrame
        One row per file per channel.
    summary : dict
        Aggregate statistics.
    """

    def _load_npz_signal(npz_path: str):
        data = np.load(npz_path, allow_pickle=True)

        if "signal" in data:
            sig = data["signal"]  # usually (N, C)
            if sig.ndim != 2:
                raise ValueError(f"{npz_path}: 'signal' is not 2D")
            if sig.shape[0] > sig.shape[1]:
                sig = sig.T  # -> (C, N)
            else:
                # if already (C, N), keep
                pass
            channel_names = data["channels"] if "channels" in data else np.array(
                [f"ch_{i}" for i in range(sig.shape[0])]
            )
            signal_key = "signal"

        elif "X" in data:
            sig = data["X"]  # usually (C, N)
            if sig.ndim != 2:
                raise ValueError(f"{npz_path}: 'X' is not 2D")
            channel_names = data["channel_names"] if "channel_names" in data else np.array(
                [f"ch_{i}" for i in range(sig.shape[0])]
            )
            signal_key = "X"

        else:
            raise KeyError(f"{npz_path}: expected 'signal' or 'X' key")

        if "fs" not in data:
            raise KeyError(f"{npz_path}: missing 'fs'")

        fs = float(data["fs"])

        return sig, fs, np.array(channel_names), signal_key

    def _compute_welch(sig_1d: np.ndarray, fs: float):
        nperseg = max(8, int(fs * nperseg_sec))
        noverlap = int(nperseg * overlap_frac)

        if len(sig_1d) < nperseg:
            nperseg = len(sig_1d)
            noverlap = min(int(nperseg * overlap_frac), max(0, nperseg - 1))

        f, pxx = welch(
            sig_1d,
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
            scaling="density"
        )
        return f, pxx

    def _analyze_channel(f: np.ndarray, pxx: np.ndarray):
        fmin, fmax = band
        band_mask = (f >= fmin) & (f <= fmax)

        if not np.any(band_mask):
            return None

        f_band = f[band_mask]
        pxx_band = pxx[band_mask]

        if len(f_band) == 0:
            return None

        peak_idx = np.argmax(pxx_band)
        peak_freq = float(f_band[peak_idx])
        peak_power = float(pxx_band[peak_idx])

        # Local baseline excluding area around the peak
        baseline_mask = np.abs(f_band - peak_freq) > baseline_exclusion_hz
        baseline_vals = pxx_band[baseline_mask]

        if len(baseline_vals) == 0:
            local_baseline = np.nan
            peak_ratio = np.nan
        else:
            local_baseline = float(np.median(baseline_vals))
            peak_ratio = float(peak_power / local_baseline) if local_baseline > 0 else np.nan

        near_target = abs(peak_freq - target_freq) <= tolerance_hz
        high_peak = (peak_ratio >= ratio_threshold) if not np.isnan(peak_ratio) else False

        return {
            "peak_freq_hz": peak_freq,
            "peak_power": peak_power,
            "local_baseline_power": local_baseline,
            "peak_to_baseline_ratio": peak_ratio,
            "near_target": near_target,
            "high_peak": high_peak,
        }

    npz_files = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if len(npz_files) == 0:
        raise FileNotFoundError(f"No files found in {input_dir} matching pattern '{pattern}'")

    rows = []

    for npz_path in npz_files:
        try:
            sig, fs, channel_names, signal_key = _load_npz_signal(npz_path)
        except Exception as e:
            rows.append({
                "file": os.path.basename(npz_path),
                "file_path": npz_path,
                "status": f"load_error: {e}",
            })
            continue

        nyquist = fs / 2
        if band[1] > nyquist:
            rows.append({
                "file": os.path.basename(npz_path),
                "file_path": npz_path,
                "status": f"skipped_band_above_nyquist ({band[1]} > {nyquist:.2f})",
                "fs": fs,
            })
            continue

        for ch in range(sig.shape[0]):
            try:
                sig_ch = sig[ch, :]
                f, pxx = _compute_welch(sig_ch, fs)
                metrics = _analyze_channel(f, pxx)

                if metrics is None:
                    rows.append({
                        "file": os.path.basename(npz_path),
                        "file_path": npz_path,
                        "signal_key": signal_key,
                        "fs": fs,
                        "channel_idx": ch,
                        "channel_name": str(channel_names[ch]),
                        "status": "no_band_data",
                    })
                    continue

                rows.append({
                    "file": os.path.basename(npz_path),
                    "file_path": npz_path,
                    "signal_key": signal_key,
                    "fs": fs,
                    "channel_idx": ch,
                    "channel_name": str(channel_names[ch]),
                    "status": "ok",
                    **metrics
                })

            except Exception as e:
                rows.append({
                    "file": os.path.basename(npz_path),
                    "file_path": npz_path,
                    "signal_key": signal_key,
                    "fs": fs,
                    "channel_idx": ch,
                    "channel_name": str(channel_names[ch]) if ch < len(channel_names) else f"ch_{ch}",
                    "status": f"channel_error: {e}",
                })

    df = pd.DataFrame(rows)

    df_ok = df[df["status"] == "ok"].copy()

    if len(df_ok) == 0:
        summary = {
            "n_files": len(npz_files),
            "n_valid_rows": 0,
            "message": "No valid channel analyses were produced."
        }
    else:
        summary = {
            "n_files": len(npz_files),
            "n_valid_rows": int(len(df_ok)),
            "n_unique_files_analyzed": int(df_ok["file"].nunique()),
            "n_unique_channels": int(df_ok[["file", "channel_idx"]].drop_duplicates().shape[0]),
            "target_freq_hz": target_freq,
            "tolerance_hz": tolerance_hz,
            "band": band,
            "ratio_threshold": ratio_threshold,
            "pct_near_target": float(100 * df_ok["near_target"].mean()),
            "pct_high_peak": float(100 * df_ok["high_peak"].mean()),
            "pct_near_target_and_high_peak": float(
                100 * ((df_ok["near_target"]) & (df_ok["high_peak"])).mean()
            ),
            "median_peak_freq_hz": float(df_ok["peak_freq_hz"].median()),
            "mean_peak_freq_hz": float(df_ok["peak_freq_hz"].mean()),
            "std_peak_freq_hz": float(df_ok["peak_freq_hz"].std(ddof=1)) if len(df_ok) > 1 else 0.0,
            "median_peak_ratio": float(df_ok["peak_to_baseline_ratio"].median()),
            "mean_peak_ratio": float(df_ok["peak_to_baseline_ratio"].mean()),
            "std_peak_ratio": float(df_ok["peak_to_baseline_ratio"].std(ddof=1)) if len(df_ok) > 1 else 0.0,
        }

    if save_csv_path is not None:
        df.to_csv(save_csv_path, index=False)

    return df, summary




#=================================================================================
#=================================================================================
#=================================================================================

# FUNCTION #14
def sanity_check_global_zscore_npz_1_14(
    original_dir: str,
    normalized_dir: str,
    pattern: str = "*.npz",
    atol: float = 1e-5,
    rtol: float = 1e-5,
):
    """
    Sanity check for globally normalized EEG NPZ files.

    Checks:
    1. Original and normalized files exist with matching names
    2. Required keys exist
    3. Shapes match
    4. Recomputed z-score matches saved normalized X
    5. Reports per-file stats for inspection

    Parameters
    ----------
    original_dir : str
        Directory containing original NPZ files.
    normalized_dir : str
        Directory containing normalized NPZ files.
    pattern : str
        Glob pattern for NPZ files.
    atol : float
        Absolute tolerance for np.allclose.
    rtol : float
        Relative tolerance for np.allclose.

    Returns
    -------
    df : pd.DataFrame
        Summary table with pass/fail and useful stats.
    """
    original_dir = Path(original_dir)
    normalized_dir = Path(normalized_dir)

    original_files = sorted(original_dir.glob(pattern))
    rows = []

    required_norm_keys = {"X", "global_mu", "global_sigma", "eps"}

    for orig_path in original_files:
        norm_path = normalized_dir / orig_path.name

        row = {
            "file": orig_path.name,
            "original_exists": orig_path.exists(),
            "normalized_exists": norm_path.exists(),
            "shape_original": None,
            "shape_normalized": None,
            "same_shape": False,
            "has_required_keys": False,
            "all_finite_original": False,
            "all_finite_normalized": False,
            "zscore_recomputed_match": False,
            "max_abs_diff": np.nan,
            "mean_norm_ch0": np.nan,
            "std_norm_ch0": np.nan,
            "mean_norm_ch1": np.nan,
            "std_norm_ch1": np.nan,
            "status": "fail",
            "reason": "",
        }

        try:
            if not norm_path.exists():
                row["reason"] = "normalized file missing"
                rows.append(row)
                continue

            orig = np.load(orig_path, allow_pickle=True)
            norm = np.load(norm_path, allow_pickle=True)

            if "X" not in orig:
                row["reason"] = "original missing X"
                rows.append(row)
                continue

            norm_keys = set(norm.files)
            row["has_required_keys"] = required_norm_keys.issubset(norm_keys)
            if not row["has_required_keys"]:
                row["reason"] = f"normalized missing keys: {required_norm_keys - norm_keys}"
                rows.append(row)
                continue

            X_orig = orig["X"]
            X_norm = norm["X"]
            global_mu = norm["global_mu"]
            global_sigma = norm["global_sigma"]
            eps = float(norm["eps"])

            row["shape_original"] = tuple(X_orig.shape)
            row["shape_normalized"] = tuple(X_norm.shape)
            row["same_shape"] = X_orig.shape == X_norm.shape

            if not row["same_shape"]:
                row["reason"] = "shape mismatch"
                rows.append(row)
                continue

            row["all_finite_original"] = bool(np.all(np.isfinite(X_orig)))
            row["all_finite_normalized"] = bool(np.all(np.isfinite(X_norm)))

            if global_mu.shape[0] != X_orig.shape[0]:
                row["reason"] = "global_mu length does not match n_channels"
                rows.append(row)
                continue

            if global_sigma.shape[0] != X_orig.shape[0]:
                row["reason"] = "global_sigma length does not match n_channels"
                rows.append(row)
                continue

            # Recompute z-score from original using saved global stats
            X_recomputed = np.empty_like(X_orig, dtype=np.float32)
            for c in range(X_orig.shape[0]):
                X_recomputed[c] = (X_orig[c] - global_mu[c]) / (global_sigma[c] + eps)

            diff = np.abs(X_recomputed - X_norm)
            row["max_abs_diff"] = float(np.max(diff))
            row["zscore_recomputed_match"] = bool(
                np.allclose(X_recomputed, X_norm, atol=atol, rtol=rtol)
            )

            # Useful descriptive stats
            if X_norm.shape[0] > 0:
                row["mean_norm_ch0"] = float(np.mean(X_norm[0]))
                row["std_norm_ch0"] = float(np.std(X_norm[0]))
            if X_norm.shape[0] > 1:
                row["mean_norm_ch1"] = float(np.mean(X_norm[1]))
                row["std_norm_ch1"] = float(np.std(X_norm[1]))

            if row["zscore_recomputed_match"]:
                row["status"] = "pass"
            else:
                row["reason"] = "saved normalized X does not match recomputed z-score"

        except Exception as e:
            row["reason"] = str(e)

        rows.append(row)

    df = pd.DataFrame(rows)

    print("\n=== Sanity Check Summary ===")
    print(df["status"].value_counts(dropna=False))

    n_fail = (df["status"] != "pass").sum()
    if n_fail > 0:
        print("\nFailed files:")
        print(df.loc[df["status"] != "pass", ["file", "reason"]].to_string(index=False))

    return df
#=================================================================================
#=================================================================================
#=================================================================================

# FUNCTION #15
import numpy as np

def compute_global_channel_stats_1_15(all_files, key="X", verbose=True):
    """
    Compute global mean and standard deviation per EEG channel
    across multiple NPZ files.

    Parameters
    ----------
    all_files : list
        List of paths to NPZ files.
    
    key : str
        Key inside each NPZ file containing the EEG signal.
        Expected shape: (n_channels, n_samples).
    
    verbose : bool
        If True, print progress and final results.

    Returns
    -------
    ch_mean : np.ndarray
        Global mean per channel.
    
    ch_std : np.ndarray
        Global standard deviation per channel.
    
    ch_count : np.ndarray
        Number of valid finite samples used per channel.
    
    skipped : list
        List of files that were skipped.
    """

    # Infer number of channels from the first file
    first_data = np.load(all_files[0], allow_pickle=True)
    N_CHANNELS = first_data[key].shape[0]

    if verbose:
        print(f"Number of channels: {N_CHANNELS}")

    # One accumulator per channel
    ch_count = np.zeros(N_CHANNELS, dtype=np.float64)
    ch_mean  = np.zeros(N_CHANNELS, dtype=np.float64)
    ch_M2    = np.zeros(N_CHANNELS, dtype=np.float64)

    skipped = []

    for path in all_files:
        try:
            data = np.load(path, allow_pickle=True)
        except Exception as exc:
            if verbose:
                print(f"[WARNING] Could not load '{path.name}': {exc}")
            skipped.append(path.name)
            continue

        if key not in data:
            if verbose:
                print(f"[WARNING] '{path.name}' has no key '{key}' — skipping.")
            skipped.append(path.name)
            continue

        X = data[key]  # Expected shape: (C, N)

        for c in range(N_CHANNELS):
            row = X[c, :]

            # Keep only valid numeric values
            finite_vals = row[np.isfinite(row)]

            if finite_vals.size == 0:
                continue

            # Statistics for the current file/channel
            b_count = finite_vals.size
            b_mean  = float(np.mean(finite_vals))
            b_M2    = float(np.var(finite_vals, ddof=0)) * b_count

            # Chan's parallel update
            combined_count = ch_count[c] + b_count
            delta = b_mean - ch_mean[c]

            ch_mean[c] = (
                ch_count[c] * ch_mean[c] + b_count * b_mean
            ) / combined_count

            ch_M2[c] += (
                b_M2
                + delta**2 * (ch_count[c] * b_count) / combined_count
            )

            ch_count[c] = combined_count

    # Final standard deviation per channel
    ch_std = np.sqrt(ch_M2 / ch_count)

    if verbose:
        print("\n" + "=" * 45)
        for c in range(N_CHANNELS):
            print(
                f"  Channel {c}  |  mean: {ch_mean[c]:.6f}  "
                f"|  std: {ch_std[c]:.6f}  "
                f"|  samples: {int(ch_count[c]):,}"
            )
        print("=" * 45)
        print(f"Files skipped: {len(skipped)}")

    return ch_mean, ch_std, ch_count, skipped
#=================================================================================
#=================================================================================
#=================================================================================

# FUNCTION #16

from pathlib import Path
import numpy as np

def apply_global_channel_normalization_1_16(
    all_files,
    ch_mean,
    ch_std,
    ch_count,
    output_dir,
    key="X",
    eps=1e-8,
    verbose=True
):
    """
    Apply global channel-wise z-score normalization to multiple NPZ files.

    This function uses global mean and standard deviation values computed
    from all recordings, and saves a normalized version of each NPZ file.

    Parameters
    ----------
    all_files : list
        List of paths to NPZ files.

    ch_mean : np.ndarray
        Global mean per channel.

    ch_std : np.ndarray
        Global standard deviation per channel.

    ch_count : np.ndarray
        Number of samples used to compute global statistics.

    output_dir : str or Path
        Directory where normalized NPZ files will be saved.

    key : str
        Key inside each NPZ file containing the EEG signal.

    eps : float
        Small value added to standard deviation to avoid division by zero.

    verbose : bool
        If True, print progress messages.

    Returns
    -------
    stats_npz_path : Path
        Path to the saved global normalization statistics file.

    summary : dict
        Dictionary with processing summary.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_channels = len(ch_mean)

    metadata_keys = (
        "mu", "sigma", "fs", "channel_names",
        "source_file", "seizure_onsets", "T0", "TF"
    )

    n_ok = 0
    n_skipped = 0

    recording_names = []
    recording_shapes = []
    recording_ok = []

    for path in all_files:
        try:
            data = np.load(path, allow_pickle=True)
            X = data[key]  # Expected shape: (C, N)

            if X.shape[0] != n_channels:
                raise ValueError(
                    f"Expected {n_channels} channels, got {X.shape[0]}"
                )

            # Global channel-wise normalization
            X_norm = np.empty_like(X, dtype=np.float32)

            for c in range(n_channels):
                X_norm[c, :] = (X[c, :] - ch_mean[c]) / (ch_std[c] + eps)

            # Preserve original metadata if available
            metadata = {
                meta_key: data[meta_key]
                for meta_key in metadata_keys
                if meta_key in data
            }

            # Save normalized NPZ
            np.savez_compressed(
                output_dir / path.name,
                X=X_norm,
                global_mu=ch_mean.astype(np.float32),
                global_sigma=ch_std.astype(np.float32),
                normalization_type="global_channel_zscore",
                eps=np.float32(eps),
                **metadata
            )

            recording_names.append(path.name)
            recording_shapes.append(X.shape)
            recording_ok.append(True)

            n_ok += 1

        except Exception as exc:
            if verbose:
                print(f"[WARNING] Failed on '{path.name}': {exc}")

            recording_names.append(path.name)
            recording_shapes.append((-1, -1))
            recording_ok.append(False)

            n_skipped += 1

    # Save global normalization statistics
    stats_npz_path = output_dir / "normalization_stats_global.npz"

    np.savez_compressed(
        stats_npz_path,
        global_mu=ch_mean.astype(np.float32),
        global_sigma=ch_std.astype(np.float32),
        global_sample_count=ch_count.astype(np.int64),
        n_channels=np.int32(n_channels),
        eps=np.float32(eps),
        recording_names=np.array(recording_names, dtype=object),
        recording_shapes=np.array(recording_shapes, dtype=object),
        recording_ok=np.array(recording_ok, dtype=bool),
    )

    summary = {
        "n_ok": n_ok,
        "n_skipped": n_skipped,
        "output_dir": output_dir,
        "stats_npz_path": stats_npz_path,
    }

    if verbose:
        print(f"\nDone. Saved: {n_ok}  |  Skipped: {n_skipped}")
        print(f"Global stats written to: {stats_npz_path}")

    return stats_npz_path, summary

#=================================================================================
#=================================================================================
#=================================================================================
#=================================================================================
#=================================================================================
#=================================================================================
