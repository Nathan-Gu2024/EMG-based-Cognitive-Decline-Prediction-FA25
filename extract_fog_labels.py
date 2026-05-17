"""
extract_fog_labels.py

Aligns Mendeley filtered task .txt files with your raw IMU CSVs using
absolute wall-clock timestamps. No hardcoded GAIT_START needed.

Folder structure assumed:
    Raw/
        001/
            LShank.csv  RShank.csv  Waist.csv  Arm.csv (some may be missing)
            001.vhdr  001.eeg  001.vmrk
    Filtered Data/
        001/
            task_1.txt  task_2.txt  task_3.txt  task_4.txt (count varies)
"""

import os
import glob
import numpy as np
import pandas as pd
from datetime import datetime


TASK_NAMES = {
    "task_1": "Rising, walking, turning, obstacles, sitting",
    "task_2": "Repeat Task 1",
    "task_3": "Rising, walking, turns in limited space",
    "task_4": "Repeat Task 3",
}

SENSORS = ["Waist", "Arm", "LShank", "RShank"]  # Priority: Waist (best for gait), Arm (best coverage), then legs


# -- 1. LOAD RAW SENSOR CSV (UNFUSED ACC + GYRO) -----------------------------

def load_raw_sensor_csv(path: str) -> pd.DataFrame:
    """
    Load one raw sensor CSV (e.g. Waist.csv, LShank.csv).
    This contains UNFUSED accelerometer + gyroscope readings.
    You must apply Prep.fuse_imu_data_vectorized() afterward to get pitch/roll/yaw.

    CSV format (9 columns):
        timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, extra1, extra2

    Returns DataFrame with absolute 'timestamp', numeric 't_sec',
    and columns: acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
    """
    df = pd.read_csv(path, header=None, engine="python")
    
    # Assign column names based on actual structure
    if df.shape[1] == 9:
        df.columns = ["timestamp", "acc_x", "acc_y", "acc_z",
                      "gyro_x", "gyro_y", "gyro_z", "extra1", "extra2"]
    elif df.shape[1] == 8:
        df.columns = ["timestamp", "acc_x", "acc_y", "acc_z",
                      "gyro_x", "gyro_y", "gyro_z", "extra"]
    else:
        raise ValueError(f"Unexpected CSV format: {df.shape[1]} columns in {path}")

    # Clean and parse timestamps
    df["timestamp"] = df["timestamp"].astype(str).str.strip()
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], format="%Y_%m_%d_%H_%M_%S_%f", errors="coerce"
    )
    
    # Drop rows with failed timestamp parsing
    before_count = len(df)
    df = df.dropna(subset=["timestamp"])
    after_count = len(df)
    
    if df.empty:
        raise ValueError(f"All timestamps failed to parse in {path}")
    
    if before_count != after_count:
        print(f"  WARNING: Dropped {before_count - after_count} rows with invalid timestamps")
    
    # Remove duplicates and sort
    df = df.drop_duplicates(subset=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Calculate t_sec relative to first sample
    df["t_sec"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()
    
    return df


# -- 2. LOAD FILTERED TASK .TXT -----------------------------------------------

def load_filtered_txt(path: str, recording_date) -> pd.DataFrame:
    """
    Load one Mendeley filtered task .txt file.

    Attaches recording_date (from the IMU CSV first timestamp) to the
    HH:MM:SS times so both share an absolute datetime for alignment.
    """
    df = pd.read_csv(path, header=None, engine="python")
    df = df.rename(columns={
        0:               "sample_idx",
        1:               "time_str",
        df.columns[-1]:  "label",
    })

    times = pd.to_datetime(
        df["time_str"].astype(str).str.strip(),
        format="%H:%M:%S.%f", errors="coerce"
    )
    mask = times.isna()
    if mask.any():
        times[mask] = pd.to_datetime(
            df.loc[mask, "time_str"].astype(str).str.strip(),
            format="%H:%M:%S", errors="coerce"
        )

    # Attach recording date to get absolute wall-clock datetimes
    df["timestamp"] = times.apply(
        lambda t: datetime(
            recording_date.year, recording_date.month, recording_date.day,
            t.hour, t.minute, t.second, t.microsecond
        ) if pd.notna(t) else pd.NaT
    )

    midnight = datetime(recording_date.year, recording_date.month, recording_date.day)
    df["t_sec"] = (df["timestamp"] - midnight).dt.total_seconds()
    df["label"] = df["label"].astype(int)
    return df


# -- 3. EXTRACT FOG EVENTS ----------------------------------------------------

def extract_fog_events(df: pd.DataFrame, min_duration_sec: float = 0.5) -> list:
    """
    Convert binary label column into list of (start_sec, end_sec) tuples.
    t_sec values use whatever epoch df uses.
    """
    labels = df["label"].to_numpy()
    t      = df["t_sec"].to_numpy()

    diff   = np.diff(labels, prepend=0, append=0)
    starts = np.where(diff ==  1)[0]
    ends   = np.where(diff == -1)[0]

    events = []
    for s, e in zip(starts, ends):
        end_idx  = min(e, len(t) - 1)
        duration = t[end_idx] - t[s]
        if duration >= min_duration_sec:
            events.append((float(t[s]), float(t[end_idx])))
    return events


# -- 4. SLICE RAW SENSOR TO TASK WINDOW ---------------------------------------

def slice_raw_sensor_to_task(sensor_df: pd.DataFrame, task_df: pd.DataFrame) -> pd.DataFrame:
    """
    Crop the continuous raw sensor DataFrame to one task's time window.
    Re-zeros t_sec to task start so fog_events offsets match.
    
    This gives you the raw acc+gyro data for the task window.
    You still need to apply Prep.fuse_imu_data_vectorized() to get pitch/roll/yaw.
    """
    task_start = task_df["timestamp"].iloc[0]
    task_end   = task_df["timestamp"].iloc[-1]

    mask   = (sensor_df["timestamp"] >= task_start) & (sensor_df["timestamp"] <= task_end)
    sliced = sensor_df[mask].copy().reset_index(drop=True)

    if sliced.empty:
        print(f"  WARNING: No sensor samples found between "
              f"{task_start.strftime('%H:%M:%S')} and {task_end.strftime('%H:%M:%S')}")
        return sliced

    sliced["t_sec"] = (sliced["timestamp"] - task_start).dt.total_seconds()
    return sliced


# -- 5. PROCESS ALL SUBJECTS --------------------------------------------------

def process_all_subjects(raw_root: str, filtered_root: str, sensor: str = "Waist") -> dict:
    """
    For every subject + task:
      - Load raw sensor CSV (unfused acc+gyro) for chosen sensor
      - Load filtered task .txt
      - Align via absolute timestamps
      - Extract fog_events as t_sec offsets from task start
      - Slice raw sensor data to task window

    Returns:
    {
        "001": {
            "task_1": {
                "raw_sensor_df": DataFrame,    # sliced raw acc+gyro, t_sec zeroed at task start
                                                # STILL NEEDS Prep.fuse_imu_data_vectorized()
                "fog_events":    [(s, e), ...], # t_sec offsets from task start
                "task_desc":     str
            }, ...
        }, ...
    }
    """
    results = {}

    subject_ids = sorted([
        d for d in os.listdir(raw_root)
        if os.path.isdir(os.path.join(raw_root, d))
    ])

    for subj_id in subject_ids:
        raw_path      = os.path.join(raw_root, subj_id)
        filtered_path = os.path.join(filtered_root, subj_id)

        if not os.path.isdir(filtered_path):
            print(f"[{subj_id}] No filtered folder found, skipping.")
            continue

        # Find sensor CSV -- prefer chosen sensor, fall back to whatever exists
        csv_path = os.path.join(raw_path, f"{sensor}.csv")
        used_sensor = sensor
        if not os.path.exists(csv_path):
            for alt in SENSORS:
                alt_path = os.path.join(raw_path, f"{alt}.csv")
                if os.path.exists(alt_path):
                    csv_path   = alt_path
                    used_sensor = alt
                    break
            else:
                print(f"[{subj_id}] No sensor CSV found, skipping.")
                continue

        print(f"\n[{subj_id}] Loading {used_sensor}.csv ...")
        raw_sensor_df = load_raw_sensor_csv(csv_path)

        # Recording date comes from the sensor CSV itself -- no hardcoding needed
        recording_date = raw_sensor_df["timestamp"].iloc[0].date()
        midnight = datetime(recording_date.year, recording_date.month, recording_date.day)
        raw_sensor_df["t_sec"] = (raw_sensor_df["timestamp"] - midnight).dt.total_seconds()

        print(f"  Raw sensor: {len(raw_sensor_df):,} samples | "
              f"{raw_sensor_df['timestamp'].iloc[0].strftime('%H:%M:%S')} -> "
              f"{raw_sensor_df['timestamp'].iloc[-1].strftime('%H:%M:%S')} | "
              f"date: {recording_date}")

        task_files = sorted(glob.glob(os.path.join(filtered_path, "task_*.txt")))
        results[subj_id] = {}

        for task_path in task_files:
            task_name = os.path.splitext(os.path.basename(task_path))[0]
            task_desc = TASK_NAMES.get(task_name, task_name)

            task_df = load_filtered_txt(task_path, recording_date)

            # FoG events re-zeroed to task start
            task_start_sec  = task_df["t_sec"].iloc[0]
            fog_events_abs  = extract_fog_events(task_df)
            fog_events      = [(s - task_start_sec, e - task_start_sec)
                               for s, e in fog_events_abs]

            # Slice raw sensor data to this task window
            raw_sensor_task = slice_raw_sensor_to_task(raw_sensor_df, task_df)

            n_fog = sum(e - s for s, e in fog_events)
            print(f"  {task_name}: {len(fog_events)} FoG events | "
                  f"{n_fog:.1f}s FoG | "
                  f"{len(raw_sensor_task):,} sensor samples | "
                  f"{task_desc}")

            results[subj_id][task_name] = {
                "raw_sensor_df": raw_sensor_task,  # UNFUSED - needs Prep.fuse_imu_data_vectorized()
                "fog_events":    fog_events,
                "task_desc":     task_desc,
            }

    return results


# -- 6. PLUG INTO YOUR EXISTING PIPELINE --------------------------------------

if __name__ == "__main__":
    RAW_ROOT      = "/path/to/Raw"            # <- change
    FILTERED_ROOT = "/path/to/Filtered Data"  # <- change
    SENSOR        = "Waist"                   # preferred; falls back if missing

    all_data = process_all_subjects(RAW_ROOT, FILTERED_ROOT, sensor=SENSOR)

    # Use in train_cnn.py like this:
    #
    # from FoG_CNN_class import FoG_Class as FC
    # from prep import Prep
    #
    # for subj_id, tasks in all_data.items():
    #     for task_name, task in tasks.items():
    #
    #         raw_sensor_df = task["raw_sensor_df"]  # unfused acc+gyro, t_sec from task start
    #         fog_events    = task["fog_events"]     # [(start_sec, end_sec), ...]
    #
    #         # Step 1: Fuse the raw sensor data using YOUR pipeline
    #         imu_fused = Prep.fuse_imu_data_vectorized(
    #             raw_sensor_df, sfreq=500.0, alpha=0.98
    #         )  # adds pitch, roll, yaw columns
    #
    #         # Step 2: Resample to 128 Hz for CNN
    #         df_128 = FC.resample_imu_d(imu_fused, target_sfreq=128.0)
    #
    #         # Step 3: Build labels and feature matrix
    #         y = FC.build_window_labels(df_128, fog_events)
    #         S = FC.build_S_matrix(df_128)
    #         X = FC.sliding_windows(S)
    #
    #         print(f"{subj_id}/{task_name}: X={X.shape}, y={y.shape}, "
    #               f"fog_events={len(fog_events)}")
    #
    # NOTE: If you need the raw sensor data at 128 Hz before fusion, resample first:
    #       raw_128 = FC.resample_imu_d(raw_sensor_df, target_sfreq=128.0)
    #       imu_fused = Prep.fuse_imu_data_vectorized(raw_128, sfreq=128.0, alpha=0.98)
    #       Then build S matrix, etc.