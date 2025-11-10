import os
import glob
import mne
import numpy as np
import pandas as pd
from scipy.interpolate import *
from datetime import *
from mne.io import *
from kalman_filter import KalmanFilter

class Prep:
    def get_vhdr_files(root_dir):
        vhdr_files = glob.glob(os.path.join(root_dir, '**', '*.vhdr'), recursive=True)
        print(f"Found {len(vhdr_files)} .vhdr files")
        return vhdr_files

    #Needs to handle subject 8's files still, can't parse since 2 layers of folders
    def prep(file_path, run_ica=True):
        print(f"Loading: {file_path}")
        raw = read_raw_brainvision(file_path, preload=True)
        raw._data *= 1e6  # convert Volts to µV
        emg_channels = [channel for channel in raw.ch_names if channel.startswith('EMG')]
        print(f"EMG channels: {emg_channels}")
        
        raw_emg = raw.copy().pick_channels(emg_channels)
        
        # I think their band pass was 10-500Hz then resample to 500
        raw_emg_filtered = raw_emg.copy().filter(l_freq=10, h_freq=499.99, fir_design = 'firwin', verbose = False)
        raw_emg_filtered.notch_filter(freqs = 50, verbose = False)

        if raw_emg.info['sfreq'] != 499.99:
            raw_emg_filtered = raw_emg_filtered.resample(499.99, npad = "auto")

        raw_eeg, ica = None, None
        if run_ica:
            eeg_channels = [ch for ch in raw.ch_names if ch not in emg_channels and 
                        any(x in ch.lower() for x in ["fp", "f", "c", "p", "o", "t", "z", "eeg"])]
            if eeg_channels:
                raw_eeg = raw.copy().pick_channels(eeg_channels)

                raw_eeg.filter(l_freq = 0.5, h_freq = 100, fir_design = "firwin", verbose = False)
                raw_eeg.notch_filter(freqs = 50, verbose = False)
                
                ica = mne.preprocessing.ICA(n_components=15, random_state=42)
                ica.fit(raw_eeg)     
                raw_eeg = ica.apply(raw_eeg)

                if (raw_eeg.info['sfreq'] != 500):
                    raw_eeg.resample(500, npad = "auto")
        return raw_eeg, raw_emg, ica, raw_emg_filtered


    def check_quality(data, freq, channels):
        for i, channel in enumerate(channels):
            channel_data = data[i]
            
            mean_val = channel_data.mean()
            std_val = channel_data.std()
            max_val = channel_data.max()
            min_val = channel_data.min()
            
            #NAN 
            nan_percentage = np.isnan(channel_data).sum() / len(channel_data) * 100
            
            # Saturation/clipping detection 
            abs_data = np.abs(channel_data)
            saturation_threshold = np.percentile(abs_data, 99.5)
            clipping_percentage = (abs_data > saturation_threshold).sum() / len(channel_data) * 100
            
            print(f"\n{channel}:")
            print(f"  Samples: {len(channel_data):,}")
            print(f"  Duration: {len(channel_data)/freq:.2f}s")
            print(f"  Stats: mean={mean_val:.2f}μV, std={std_val:.2f}μV")
            print(f"  Range: [{min_val:.2f}, {max_val:.2f}]μV")
            print(f"  Quality: {nan_percentage:.2f}% NaN")
            print(f"  Clipping: {clipping_percentage:.2f}% > {saturation_threshold:.2f}μV")



    # Merge preprocessed ACC sensor DataFrames (from prep_acc_data) into one aligned DataFrame
    # Retrun pd.DataFrame: merged accelerometer + gyro data aligned to the reference sensor
    def merge_all_sensors(subject_acc_data, gait_start):
        if not subject_acc_data:
            raise ValueError("No accelerometer data provided to merge_all_sensors()")

        #pick reference for time alignment
        ref_sensor = "Waist" if "Waist" in subject_acc_data else list(subject_acc_data.keys())[0]
        ref_time = subject_acc_data[ref_sensor]["t_sec"].values
        aligned = pd.DataFrame({"t_sec": ref_time})

        #merge interpolated columns for all sensors
        for sensor, df in subject_acc_data.items():
            for col in ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]:
                aligned[f"{sensor}_{col}"] = np.interp(ref_time, df["t_sec"], df[col])

        #reconstruct absolute timestamps
        aligned["timestamp"] = [
            (gait_start + pd.to_timedelta(t, unit="s")).strftime("%H:%M:%S.%f")[:-3]
            for t in aligned["t_sec"]
        ]

        return aligned

    #LShank, RShank, Waist, Arm
    # timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, NC/SC (not important)
    #Sampled at 500Hz => 2ms / sample
    #Missing 1-2 CSVs
    #Likely skips over the 8th subject due to the actual data being 2 folders rather than 1 folder deep
    #Need to align sensors (LShank, RShank, Waist, Arm) 
    # ^^ (maybe make into a singular mne object so it is easier to work with)
    def prep_acc_data(path_acc, sensor_loc, gait_start): 
        file_path = os.path.join(path_acc, f"{sensor_loc}.csv")

        if not os.path.exists(file_path):
            print(f"Missing file for {sensor_loc}: {file_path}")
            return None
        
        df = pd.read_csv(
            file_path,
            header=None,
            names=["timestamp", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "extra"],
            index_col=False,
            engine="python",
        )

        #Clean timestamps column
        df["timestamp"] = df["timestamp"].astype(str).str.strip().str.strip(",")
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y_%m_%d_%H_%M_%S_%f", errors="coerce")
        df = df.dropna(subset=["timestamp"])

        #Convert timestamps to seconds
        t0 = df["timestamp"].iloc[0]
        df["t_sec"] = (df["timestamp"] - t0).dt.total_seconds()

        #Handle duplicates by averaging over duplicates since the interp can't deal with dupes
        df = df.groupby("t_sec", as_index=False).mean(numeric_only=True)

        #Create 500 Hz timeline (0.002s spacing)
        t_interp = np.arange(df["t_sec"].iloc[0], df["t_sec"].iloc[-1], 1/500)

        #Interpolate each column
        interp_df = pd.DataFrame({"t_sec": t_interp})
        for col in ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]:
            f = interp1d(df["t_sec"], df[col], kind="cubic", fill_value="extrapolate")
            interp_df[col] = f(t_interp)

        #Create absolute timestamps relative to gait_start
        interp_df["timestamp"] = [
            (gait_start + timedelta(seconds=t)).strftime("%H:%M:%S.%f")[:-3]
            for t in interp_df["t_sec"]
        ]
        return interp_df

    #1. Convert the gyro to anguler velo
    # subtract the gyroscope's bias (average the readings over a period when the sensor is stationary,
    #  then subtracting this average from all subsequent readings)
    #2. Take gyro readings at discrete time intervals
    # Multiply the angular velocity by the time interval to find the change in angle for that interval.
    # Sum these changes over time to get a cumulative estimate of the sensor's orientation (prone to drift)
    # 3. Use  accelerometer's measurements to determine the orientation relative to gravity
    # Combine  with gyroscope data to create more accurate, stable orientation estimate
    # 4. Apply kalman / complementary filter to combine the data (kalman is in another class)
    def fuse_imu_data(df):
        dt = 0.002  # 500 Hz
        kf_x = KalmanFilter.create_kalman_filter(dt)
        kf_y = KalmanFilter.create_kalman_filter(dt)
        kf_z = KalmanFilter.create_kalman_filter(dt)

        fused_pitch, fused_roll, fused_yaw = [], [], []

        for _, row in df.iterrows():
            # Gyro input (rad/s)
            gyro = np.array([[row["gyro_x"]], [row["gyro_y"]], [row["gyro_z"]]])

            # Acc-derived angles
            acc_x, acc_y, acc_z = row["acc_x"], row["acc_y"], row["acc_z"]
            pitch_meas = np.arctan2(-acc_x, np.sqrt(acc_y**2 + acc_z**2))
            roll_meas  = np.arctan2(acc_y, acc_z)

            # X-axis (pitch)
            kf_x.predict(u=np.array([[gyro[0,0]]]))
            kf_x.update(np.array([[pitch_meas]]))
            fused_pitch.append(kf_x.estimate0[0,0])

            # Y-axis (roll)
            kf_y.predict(u=np.array([[gyro[1,0]]]))
            kf_y.update(np.array([[roll_meas]]))
            fused_roll.append(kf_y.estimate0[0,0])

            # Z-axis (yaw)
            kf_z.predict(u=np.array([[gyro[2,0]]]))
            fused_yaw.append(kf_z.estimate0[0,0])

        fused_df = df.copy()
        fused_df["pitch"] = np.degrees(fused_pitch)
        fused_df["roll"] = np.degrees(fused_roll)
        fused_df["yaw"] = np.degrees(fused_yaw)

        return fused_df
