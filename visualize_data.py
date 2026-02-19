import mne
from mne.io import *
import requests
import glob
import os
import datetime
import numpy as np
from datetime import *
from kalman_filter import KalmanFilter
from prep import Prep
from plots import Plots
from dashboard import Dashboard
from parallel_proc import Parallel
from FoG_CNN_class import FoG_Class as FC

### Find the metrics for the FoG

# 1. Check that our preprocessed data done accordingly and prep it for training
#     1. Fully plot all the data (EMG, EEG, IMU) with the event markers present to make sure they line up
#     2. Find the absolute timestamps for the rest of the 11 subjects
#     3. Segment the data like that of the Github repo we are referencing 
#     4. Incorporate multithreading to make the process of preprocessing the data quicker each run
#     5. Do cross validation on the EEG / EMG data with the IMU data for FoG using an existing model that predicts FoG off of IMU data 
#     6. Get the timeseries plot to be more interactive so its easier to see all the data



# 2. Start training and tweaking the model for baseline XGBoost model
#     1. Data augmentation to get as many features as possible from the data
#     2. Find the precision and accuracy of the model
#     3. Test with the weights for certain features within the model and determine whether or not there is imbalance in the data (then apply SMOTE); probably incorporate cross validation



#Running plots and preproc data
if __name__ == "__main__":
    USE_KALMAN = False  
    ROOT = "/Users/nathangu/Desktop/Pytorch/NT/t8j8v4hnm4-1/Raw"  
    SENSORS = ["LShank", "RShank", "Waist", "Arm"]

    GAIT_START_STR = "2019-12-18 09:28:46.727"
    GAIT_START = datetime.strptime(GAIT_START_STR, "%Y-%m-%d %H:%M:%S.%f")

    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    os.environ["USE_KALMAN"] = "True" if USE_KALMAN else "False"

    print("Running parallel preprocessing...")
    all_subjects_data = Parallel.run_parallel(
        root_dir=ROOT,
        gait_start=GAIT_START,
        sensors=SENSORS,
        max_workers=4
    )
    subj_id = list(all_subjects_data.keys())[0]
    sd = all_subjects_data[subj_id]

    acc_df = sd["acc_aligned"]
    print(acc_df.columns.tolist())

    available_sensors = set(c.split("_")[0] for c in acc_df.columns if "_acc_" in c)
    print("Available sensors:", available_sensors)

    sensor = "Waist" if "Waist" in list(available_sensors) else list(available_sensors)[0]
    print("Using sensor:", sensor)

    # Find this sensor's columns
    sensor_cols = [c for c in acc_df.columns if c.startswith(sensor)]

    print("Found sensor columns:", sensor_cols)

    df_sensor = acc_df[["t_sec"] + sensor_cols].copy()

    rename_map = {
        f"{sensor}_acc_x": "acc_x",
        f"{sensor}_acc_y": "acc_y",
        f"{sensor}_acc_z": "acc_z",
        f"{sensor}_gyro_x": "gyro_x",
        f"{sensor}_gyro_y": "gyro_y",
        f"{sensor}_gyro_z": "gyro_z",
    }

    df_sensor = acc_df[["t_sec"] + list(rename_map.keys())].rename(columns=rename_map)

    print("Renamed columns:", df_sensor.columns.tolist())

    print(df_sensor.head())
    print(df_sensor.dtypes)

    df_imu_128 = FC.resample_imu_d(df_sensor, target_sfreq=128.0)


    #FoG ground truth
    fog_events = [
        (30.0, 35.0),
        (80.0, 83.0),
    ]
    # fog_events = [
    #     (200, 600),
    #     (900, 1300),
    # ]


    fog_labels = FC.build_window_labels(
        df_imu_128=df_imu_128,
        fog_events=fog_events
    )
    print("df_imu_128 columns:", df_imu_128.columns.tolist())
    Plots.plot_fog_windows(
        imu_df=df_imu_128,
        labels=fog_labels,
        sfreq=128
    )
    print(df_imu_128.head())
    print(df_imu_128.describe())
    print(np.unique(fog_labels, return_counts=True))

    S = FC.build_S_matrix(df_imu_128)
    X = FC.sliding_windows(S)

    print("X shape:", X.shape)
    print("y shape:", fog_labels.shape)









# for subj_path in subject_dirs:
#     subj_id = os.path.basename(subj_path)
#     print(f"\nSUBJECT {subj_id}")
    
#     #Loading EEG / EMG
#     vhdr_pattern = os.path.join(subj_path, "*.vhdr")
#     vhdr_files = glob.glob(vhdr_pattern)
#     if len(vhdr_files) == 0:
#         print(f"No .vhdr found in {subj_path}, skipping EEG/EMG preprocessing for this subject.")
#         raw_eeg = raw_emg = ica = raw_emg_filtered = None
#         events = event_id = None
#     else:
#         vhdr_file = vhdr_files[0]  
#         print(f"Found VHDR: {os.path.basename(vhdr_file)}")
#         raw_eeg, raw_emg, ica, raw_emg_filtered = Prep.prep(vhdr_file, run_ica=True)
#         raw_for_events = read_raw_brainvision(vhdr_file, preload=True)
#         events, event_id = mne.events_from_annotations(raw_for_events)
#         print(f"Loaded EEG/EMG. EMG channels: {raw_emg.ch_names if raw_emg is not None else 'None'}")
    

#     #Load ACC / gyro
#     subj_acc_data = {}
#     for sensor in SENSORS:
#         csv_path = os.path.join(subj_path, f"{sensor}.csv")
#         if os.path.exists(csv_path):
#             print(f"Loading ACC CSV: {sensor}.csv")
#             df_acc = Prep.prep_acc_data(subj_path, sensor, GAIT_START)
            
#             if df_acc is not None:
#                 print(f"{sensor}: {len(df_acc)} samples, "
#                       f"t_sec range {df_acc['t_sec'].iloc[0]:.3f} - {df_acc['t_sec'].iloc[-1]:.3f}s")
#                 subj_acc_data[sensor] = df_acc
#             else:
#                 print(f"rep_acc_data returned None for {sensor}")
#         else:
#             print(f"Missing ACC CSV: {sensor}.csv (skipping)")

#     #Aligning sensors from ACC / gyro
#     if len(subj_acc_data) == 0:
#         print("No ACC sensors loaded for this subject.")
#         aligned_data = None
#     else:
#         aligned_data = Prep.merge_all_sensors(subj_acc_data, GAIT_START)
#         print(f"Merged accelerometer dataframe shape: {aligned_data.shape}")


#     #Fusing ACC + gyro for IMU data for each sensor
#     fused_imu_results = {}
#     if aligned_data is not None:
#         for sensor in subj_acc_data.keys():
#             print(f"\nFusing IMU data for {sensor}")
            
#             # Extract just this sensor’s columns
#             sensor_cols = [c for c in aligned_data.columns if c.startswith(sensor)]
#             if not sensor_cols:
#                 print(f"No matching columns found for {sensor}")
#                 continue
            
#             df_sensor = aligned_data[["timestamp", "t_sec"] + sensor_cols].copy()
#             df_sensor.columns = [c.replace(f"{sensor}_", "") for c in df_sensor.columns]  # normalize colnames
            
#             # Apply fusion
#             fused_df = Prep.fuse_imu_data(df_sensor)
#             fused_imu_results[sensor] = fused_df
#             merged_fused_imu = Prep.merge_fused_imu(fused_imu_results)
            
#             # print(f"{sensor} fused IMU shape: {fused_df.shape}")
#             # print(f"Fused IMU preview for {sensor}:")
#             # print(fused_df[['timestamp', 'pitch', 'roll', 'yaw']].head())
#     else:
#         print("No aligned accelerometer data to fuse IMU signals")


#     fused_vec = {}
#     fused_kal = {}

#     for sensor in subj_acc_data.keys():
#         cols = [c for c in aligned_data.columns if c.startswith(sensor)]
#         df_sensor = aligned_data[["t_sec", "timestamp"] + cols].copy()
#         df_sensor.columns = ["t_sec", "timestamp"] + [
#             c.replace(f"{sensor}_", "") for c in cols
#         ]

#         # Always compute complementary (fast)
#         fused_vec[sensor] = Prep.fuse_imu_data_vectorized(
#             df_sensor, sfreq=500.0, alpha=0.98
#         )

#         if USE_KALMAN:
#             fused_kal[sensor] = Prep.fuse_imu_data_kalman(
#                 df_sensor, sfreq=500.0, q_var=1e-4, r_var=1e-2
#             )

#     subject_store = {
#         "raw_eeg": raw_eeg,
#         "raw_emg_filtered": raw_emg_filtered,
#         "imu_fused_vec" : fused_vec,
#         "events": events,
#         "event_id": event_id
#     }
#     if USE_KALMAN:
#         subject_store["imu_fused_kal"] = fused_kal

#     all_subjects_data[subj_id] = subject_store

# app = Dashboard.build_dashboard_all_subjects(all_subjects_data)
# app.run(debug=False, use_reloader=False, port=8050)

    #Store results
    # all_subjects_data[subj_id] = {
    #     "raw_eeg": raw_eeg,
    #     "raw_emg": raw_emg,
    #     "raw_emg_filtered": raw_emg_filtered,
    #     "events": events,
    #     "event_id": event_id,
    #     "acc_dfs": subj_acc_data,
    #     "acc_aligned": aligned_data,
    #     "imu_fused": fused_imu_results,
    #     "imu_fused_by_sensor": fused_imu_results,
    #     "imu_merged": merged_fused_imu
    # }
    # Plots.plot_combined_timeseries(raw_eeg, raw_emg_filtered, fused_imu_results, events, event_id, 30, 0)

    # break
#Single subject data
# merged_df = all_subjects_data["001"]["acc_aligned"]
# imu_df = all_subjects_data["001"]["imu_fused"]
# print("\nSample merged accelerometer data:")
# print(merged_df.head())

# print("\nSample fused IMU data for Waist:")
# if "Waist" in imu_df:
#     print(imu_df["Waist"].head())

# # quick summary
# print(f"\nProcessed {len(all_subjects_data)} subjects. Example keys for subject 001:")
# example = next(iter(all_subjects_data.keys()))
# print(example, list(all_subjects_data[example].keys()))


