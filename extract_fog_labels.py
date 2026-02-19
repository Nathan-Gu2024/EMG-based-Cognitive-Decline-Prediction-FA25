import os
import glob
import numpy as np
import pandas as pd
from datetime import *

#Loading from Filtered folder
def load_filtered_txt(path):
    df = pd.read_csv(path, header=None)
    # Name the known bookend columns
    df = df.rename(columns={0: "sample_idx", 1: "timestamp", df.columns[-1]: "label"})
    # Parse timestamp → t_sec relative to first sample
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%H:%M:%S.%f", errors="coerce")

    # Fallback: some rows may lack sub-second part
    mask = df["timestamp"].isna()
    if mask.any():
        df.loc[mask, "timestamp"] = pd.to_datetime(
            df.loc[mask, df.columns[1]], format="%H:%M:%S", errors="coerce"
        )

    df["t_sec"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()
    df["label"] = df["label"].astype(int)

    return df

#Extracting events
def extract_fog_events(df, min_duration_sec=0.5):
    fog_events = []
    in_fog = False
    start_t = None

    for _, row in df.iterrows():
        if row["label"] == 1 and not in_fog:
            in_fog = True
            start_t = row["t_sec"]
        elif row["label"] == 0 and in_fog:
            in_fog = False
            duration = row["t_sec"] - start_t
            if duration >= min_duration_sec:
                fog_events.append((start_t, row["t_sec"]))

    # Close any open FoG at end of recording
    if in_fog:
        duration = df["t_sec"].iloc[-1] - start_t
        if duration >= min_duration_sec:
            fog_events.append((start_t, df["t_sec"].iloc[-1]))

    return fog_events

#Better for long recordings
def extract_fog_events_vectorized(df, min_duration_sec=0.5):
    labels = df["label"].to_numpy()
    t      = df["t_sec"].to_numpy()

    # Detect rising (0→1) and falling (1→0) edges
    diff        = np.diff(labels, prepend=0, append=0)
    starts_idx  = np.where(diff == 1)[0]
    ends_idx    = np.where(diff == -1)[0]

    fog_events = []
    for s, e in zip(starts_idx, ends_idx):
        duration = t[min(e, len(t) - 1)] - t[s]
        if duration >= min_duration_sec:
            fog_events.append((float(t[s]), float(t[min(e, len(t) - 1)])))

    return fog_events

#Stats
def fog_summary(fog_events: list) -> None:
    if not fog_events:
        print("No FoG events found.")
        return

    durations = [e - s for s, e in fog_events]
    total     = sum(durations)

    print(f"FoG episodes  : {len(fog_events)}")
    print(f"Total FoG time: {total:.1f}s  ({total/60:.1f} min)")
    print(f"Mean duration : {np.mean(durations):.2f}s")
    print(f"Min / Max     : {min(durations):.2f}s / {max(durations):.2f}s")
    print()
    for i, (s, e) in enumerate(fog_events):
        print(f"  [{i+1:3d}]  {s:8.2f}s → {e:8.2f}s   ({e-s:.2f}s)")

#Processing subjects
def load_all_fog_events(filtered_root: str, pattern: str = "*.txt") -> dict:
    all_events = {}

    subject_dirs = sorted([
        d for d in os.listdir(filtered_root)
        if os.path.isdir(os.path.join(filtered_root, d))
    ])

    for subj_id in subject_dirs:
        subj_path = os.path.join(filtered_root, subj_id)
        txt_files = sorted(glob.glob(os.path.join(subj_path, "*.txt")))

        if not txt_files:
            print(f"  WARNING: No .txt files found for subject {subj_id}")
            continue

        all_events[subj_id] = {}
        print(f"\nSubject {subj_id}  ({len(txt_files)} tasks)")

        for path in txt_files:
            task_name = os.path.splitext(os.path.basename(path))[0]  # e.g. "task_1"
            try:
                df         = load_filtered_txt(path)
                fog_events = extract_fog_events_vectorized(df)
                all_events[subj_id][task_name] = fog_events

                n_fog = df["label"].sum()
                pct   = 100 * n_fog / len(df)
                dur   = df["t_sec"].iloc[-1]
                print(f"  {task_name}: {len(fog_events)} FoG events, "
                      f"{n_fog} FoG samples ({pct:.1f}% of {dur:.1f}s recording)")
            except Exception as ex:
                print(f"  ERROR loading {path}: {ex}")

    return all_events


#Using Filtered's signals instead of my own
def extract_imu_from_filtered(df: pd.DataFrame,
                               acc_cols:  list = None,
                               gyro_cols: list = None) -> pd.DataFrame:
    if acc_cols is None:
        acc_cols  = [28, 29, 30]   # ← adjust these
    if gyro_cols is None:
        gyro_cols = [31, 32, 33]   # ← adjust these

    imu_df = pd.DataFrame({
        "t_sec":  df["t_sec"].values,
        "acc_x":  df.iloc[:, acc_cols[0]].values,
        "acc_y":  df.iloc[:, acc_cols[1]].values,
        "acc_z":  df.iloc[:, acc_cols[2]].values,
        "gyro_x": df.iloc[:, gyro_cols[0]].values,
        "gyro_y": df.iloc[:, gyro_cols[1]].values,
        "gyro_z": df.iloc[:, gyro_cols[2]].values,
    })
    return imu_df
