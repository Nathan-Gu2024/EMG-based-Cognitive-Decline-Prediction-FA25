import os
from datetime import datetime
import torch
from torch.utils.data import DataLoader
import numpy as np
from parallel_proc import Parallel
from FoG_CNN_class import FoG_Class as FC
from fog_dataset import FoGDataset
from extract_fog_labels import *
import multiprocessing


multiprocessing.set_start_method("spawn", force=True)

if __name__ == "__main__": 
    from datetime import datetime
    import torch
    from torch.utils.data import DataLoader

    from parallel_proc import Parallel
    from FoG_CNN_class import FoG_Class as FC
    from fog_dataset import FoGDataset

    ROOT = "/Users/nathangu/Desktop/Pytorch/NT/t8j8v4hnm4-1/Raw"
    SENSORS = ["LShank", "RShank", "Waist", "Arm"]

    GAIT_START_STR = "2019-12-18 09:28:46.727"
    GAIT_START = datetime.strptime(GAIT_START_STR, "%Y-%m-%d %H:%M:%S.%f")

    USE_KALMAN = False
    os.environ["USE_KALMAN"] = "True" if USE_KALMAN else "False"

    print("Loading data...")
    all_subjects_data = Parallel.run_parallel(
        root_dir=ROOT,
        gait_start=GAIT_START,
        sensors=SENSORS,
        max_workers=4
    )

    # Pick one subject for now
    subj_id = list(all_subjects_data.keys())[0]
    sd = all_subjects_data[subj_id]

    acc_df = sd["acc_aligned"]

    available_sensors = set(c.split("_")[0] for c in acc_df.columns if "_acc_" in c)
    sensor = "Waist" if "Waist" in available_sensors else list(available_sensors)[0]

    print("Using sensor:", sensor)

    # Extract exactly the 6 columns we need
    expected = [
        f"{sensor}_acc_x",
        f"{sensor}_acc_y",
        f"{sensor}_acc_z",
        f"{sensor}_gyro_x",
        f"{sensor}_gyro_y",
        f"{sensor}_gyro_z",
    ]

    # Check they exist
    missing = [c for c in expected if c not in acc_df.columns]
    if len(missing) > 0:
        raise RuntimeError(f"Missing IMU columns: {missing}")

    df_sensor = acc_df[["t_sec"] + expected].copy()

    # Rename cleanly
    df_sensor = df_sensor.rename(columns={
        f"{sensor}_acc_x": "acc_x",
        f"{sensor}_acc_y": "acc_y",
        f"{sensor}_acc_z": "acc_z",
        f"{sensor}_gyro_x": "gyro_x",
        f"{sensor}_gyro_y": "gyro_y",
        f"{sensor}_gyro_z": "gyro_z",
    })

    print("Final df_sensor columns:", df_sensor.columns.tolist())

    df_imu_128 = FC.resample_imu_d(df_sensor, target_sfreq=128.0)

    filtered_df = load_filtered_txt("/Users/nathangu/Desktop/Pytorch/NT/Filtered/Filtered Data")
    fog_events  = extract_fog_events_vectorized(filtered_df)

    y = FC.build_window_labels(df_imu_128, fog_events)

    S = FC.build_S_matrix(df_imu_128)
    X = FC.sliding_windows(S)

    
    print("X:", X.shape)
    print("y:", y.shape)

    dataset = FoGDataset(X, y)

    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=2
    )

    xb, yb = next(iter(loader))
    np.save("X_windows.npy", X)
    np.save("y_windows.npy", y)
    print("Batch X:", xb.shape)
    print("Batch y:", yb.shape)
    print("Any NaNs in X:", np.isnan(X).any())
    print("Any inf in X:", np.isinf(X).any())
    print("Label distribution:", np.unique(y, return_counts=True))
