import mne
import requests
import glob
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mne.preprocessing import ICA
from mne.io import read_raw_brainvision
from scipy.signal import spectrogram
from mne import create_info, EpochsArray
from scipy.interpolate import interp1d
from datetime import timedelta, datetime
from kalman_filter import KalmanFilter

####ACTUAL FUNCTIONS
# EMG: bandpass 10–500 Hz
# EEG: bandpass 0.5–45 Hz
# (They resample to 500 Hz or 1000 Hz depending on the modality)
def get_vhdr_files(root_dir):
    vhdr_files = glob.glob(os.path.join(root_dir, '**', '*.vhdr'), recursive=True)
    print(f"Found {len(vhdr_files)} .vhdr files")
    return vhdr_files


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


def plot_emg(original, filtered, channels, freq):
    n_channels = len(channels)
    fig, axes = plt.subplots(n_channels, 2, figsize=(15, 3 * n_channels))
    time_axis = np.arange(5000) / freq
    
    for i, channel in enumerate(channels):
        # Original signal
        axes[i, 0].plot(time_axis, original[i, :5000])
        axes[i, 0].set_title(f'{channel} - Original')
        axes[i, 0].set_ylabel('Amplitude (μV)')
        axes[i, 0].grid(True, alpha=0.3)  
        # Filtered signal 
        axes[i, 1].plot(time_axis, filtered[i, :5000])
        axes[i, 1].set_title(f'{channel} - Filtered (10-500 Hz)')
        axes[i, 1].set_ylabel('Amplitude (μV)')
        axes[i, 1].grid(True, alpha=0.3)
        
        if i == n_channels - 1:
            axes[i, 0].set_xlabel('Time (s)')
            axes[i, 1].set_xlabel('Time (s)')

    plt.tight_layout()
    plt.show()



def plot_spectrograms(original, filtered, sfreq, channel_names):
    for i, ch in enumerate(channel_names):
        f_orig, t_orig, Sxx_orig = spectrogram(original[i], sfreq, nperseg=1024, noverlap=512)
        f_filt, t_filt, Sxx_filt = spectrogram(filtered[i], sfreq, nperseg=1024, noverlap=512)
        
        #scaling so its not all purple
        Sxx_orig_db = 10 * np.log10(Sxx_orig)
        Sxx_filt_db = 10 * np.log10(Sxx_filt)
        vmin_orig, vmax_orig = np.percentile(Sxx_orig_db, [5, 95])
        vmin_filt, vmax_filt = np.percentile(Sxx_filt_db, [5, 95])

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        im1 = ax1.pcolormesh(t_orig, f_orig, Sxx_orig_db, shading='gouraud',
                             cmap='viridis', vmin=vmin_orig, vmax=vmax_orig)
        ax1.set_title(f'{ch} - Original')
        ax1.set_ylabel('Frequency [Hz]')
        ax1.set_ylim(0, 250)
        plt.colorbar(im1, ax=ax1, label='Power (dB)')


        im2 = ax2.pcolormesh(t_filt, f_filt, Sxx_filt_db, shading='gouraud',
                             cmap='viridis', vmin=vmin_filt, vmax=vmax_filt)
        ax2.set_title(f'{ch} - Filtered (10–500 Hz)')
        ax2.set_ylabel('Frequency [Hz]')
        ax2.set_xlabel('Time [s]')
        ax2.set_ylim(0, 250)
        plt.colorbar(im2, ax=ax2, label='Power (dB)')

        plt.tight_layout()
        plt.show()


# Move the event markers to the bottom, and add in accelerometer signal data for gait context
# Extend the graphs to include the full length of the task (make it interactable)
def plot_combined_timeseries(raw_eeg, raw_emg_filtered, raw_acc, events, event_id, duration, start):
    eeg_data, emg_data, acc_data = None, None, None
    eeg_channels, emg_channels, acc_channels = [], [], []
    sfreq = None

    if raw_eeg is not None:
        eeg_data = raw_eeg.get_data()
        eeg_channels = raw_eeg.ch_names
        sfreq = raw_eeg.info['sfreq']

    if raw_emg_filtered is not None:
        emg_data = raw_emg_filtered.get_data()
        emg_channels = raw_emg_filtered.ch_names
        if sfreq is None:
            sfreq = raw_emg_filtered.info['sfreq']


    if raw_acc is not None:
        acc_data = raw_acc.get_data()
        acc_channels = raw_acc.ch_names
        if sfreq is None:
            sfreq = raw_emg_filtered.info['sfreq']
    

    # Window
    n_samples = int(duration * sfreq)
    start_sample = int(start * sfreq)
    time_axis = np.arange(n_samples) / sfreq  # starts at 0


    min_len = n_samples
    if eeg_data is not None:
        min_len = min(min_len, eeg_data.shape[1] - start_sample)
    if emg_data is not None:
        min_len = min(min_len, emg_data.shape[1] - start_sample)
    if acc_data is not None:
        min_len = min(min_len, acc_data.shape[1] - start_sample)
    n_samples = max(0, min_len)
    duration = n_samples / sfreq
    time_axis = np.arange(n_samples) / sfreq


    n_emg = len(emg_channels)
    total_plots = 2 + n_emg #EEG heatmap, EMG, ACC

    height_ratios = [1.2] + [1.25] * n_emg + [0.6]
    fig, axes = plt.subplots(
    total_plots, 1,
    figsize=(15, 4 + 2.5 * total_plots),
    sharex=True,
    gridspec_kw={'height_ratios': height_ratios},
    constrained_layout=True   # handles legends and colorbars automatically
    )

    # EEG plot (top)
    # if eeg_data is not None:
    #     eeg_segment = eeg_data[:, start_sample:start_sample + n_samples]
    #     offset = 50
    #     for i, ch in enumerate(eeg_channels[:2]):  #only first EEG channel for clarity
    #         axes[0].plot(time_axis, eeg_segment[i] + i * offset, label=ch)
    #     axes[0].set_title("EEG")
    #     axes[0].set_ylabel("Amplitude (μV)")
    #     axes[0].legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=8, framealpha=0.6)
    #     axes[0].grid(alpha=0.3)


    #EEG heatmap 
    eeg_segment = eeg_data[:, start_sample:start_sample + n_samples]
    eeg_z = (eeg_segment - np.mean(eeg_segment, axis=1, keepdims=True)) / np.std(eeg_segment, axis=1, keepdims=True)
    im = axes[0].imshow(eeg_z, aspect='auto', origin='lower',
                            extent=[0, duration, 0, len(eeg_channels)],
                            cmap='RdBu_r', vmin=-3, vmax=3)
    axes[0].set_title("EEG (Z-scored Amplitude Heatmap)")
    axes[0].set_ylabel("Channels")
    axes[0].set_yticks(np.arange(len(eeg_channels)) + 0.5)
    axes[0].set_yticklabels(eeg_channels, fontsize=7)
    fig.colorbar(im, ax=axes[0], label="Z-score", shrink=0.8)



    # EMG spectrograms
    for idx, ch in enumerate(emg_channels):
        ax = axes[idx + 1]
        emg_segment = emg_data[idx, start_sample:start_sample + n_samples]

        # Pad EMG if shorter than expected
        if len(emg_segment) < n_samples:
            pad_len = n_samples - len(emg_segment)
            emg_segment = np.pad(emg_segment, (0, pad_len), mode='constant')

        f_centers, t_centers, Sxx = spectrogram(
            emg_segment, fs=sfreq, nperseg=1024, noverlap=512, mode='psd'
        )
        Sxx_db = 10 * np.log10(Sxx + 1e-12)

        # Generate evenly spaced time edges matching full window duration
        t_edges = np.linspace(0.0, duration, Sxx_db.shape[1] + 1)
        f_edges = np.linspace(f_centers[0], f_centers[-1], Sxx_db.shape[0] + 1)

        im = ax.pcolormesh(
            t_edges, f_edges, Sxx_db, shading='auto', cmap='viridis', rasterized=True
        )
        ax.set_title(f"{ch} - Filtered (10–500 Hz)")
        ax.set_ylabel("Freq [Hz]")
        ax.set_ylim(0, 250)
        ax.set_xlim(0, duration)

        # Add colorbar to the final EMG only
        if idx == n_emg - 1:
            fig.colorbar(im, ax=ax, label="Power (dB)", shrink=0.8)
            if raw_acc is None:
                ax.set_xlabel("Time [s]")
        
    if raw_acc is not None:
        acc_ax = axes[-1]
        acc_segment = acc_data[:, start_sample:start_sample + n_samples]

        # Pad accelerometer if short
        for i in range(acc_segment.shape[0]):
            if acc_segment.shape[1] < n_samples:
                pad_len = n_samples - acc_segment.shape[1]
                acc_segment[i] = np.pad(acc_segment[i], (0, pad_len), mode='constant')

        for i, ch in enumerate(acc_channels):
            acc_ax.plot(time_axis, acc_segment[i] + i * 0.5, label=ch)

        acc_ax.set_title("Accelerometer Signals (Gait Context)")
        acc_ax.set_ylabel("Amplitude (a.u.)")
        acc_ax.set_xlabel("Time [s]")
        acc_ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=7, framealpha=0.5)
        acc_ax.grid(alpha=0.3)
        


    # Event markers
    if events is not None and len(events) > 0:
        for e in events:
            event_time_abs = e[0] / sfreq
            event_time_rel = event_time_abs - start
            if 0 <= event_time_rel <= duration:
                label = [k for k, v in event_id.items() if v == e[2]]
                axes[-1].axvline(event_time_rel, color='red', linestyle='--', alpha=0.8)
                if label:
                    axes[-1].text(event_time_rel, axes[-1].get_ylim()[1] * 0.9,
                                  label[0], rotation=90, color='red', va='top', fontsize=7)

    # layout adjustment
    axes[0].set_xlim(0, duration)
    # fig.subplots_adjust(left=0.07, right=0.92, top=0.95, bottom=0.07, hspace=0.35)
    plt.show()

#1. Convert the gyro to anguler velo
# subtract the gyroscope's bias (average the readings over a period when the sensor is stationary,
#  then subtracting this average from all subsequent readings)
#2. Take gyro readings at discrete time intervals
# Multiply the angular velocity by the time interval to find the change in angle for that interval.
# Sum these changes over time to get a cumulative estimate of the sensor's orientation (prone to drift)
# 3. Use  accelerometer's measurements to determine the orientation relative to gravity
# Combine  with gyroscope data to create more accurate, stable orientation estimate
# 4. Apply kalman / complementary filter to combine the data (kalman is in another class)
def calc_imu_data():
    return None

# Merge preprocessed ACC sensor DataFrames (from prep_acc_data) into one aligned DataFrame
# Retrun pd.DataFrame: merged accelerometer + gyro data aligned to the reference sensor
def merge_all_sensors(subject_acc_data, gait_start):
    if not subject_acc_data:
        raise ValueError("No accelerometer data provided to merge_all_sensors()")

    # pick reference for time alignment
    ref_sensor = "Waist" if "Waist" in subject_acc_data else list(subject_acc_data.keys())[0]
    ref_time = subject_acc_data[ref_sensor]["t_sec"].values
    aligned = pd.DataFrame({"t_sec": ref_time})

    # merge interpolated columns for all sensors
    for sensor, df in subject_acc_data.items():
        for col in ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]:
            aligned[f"{sensor}_{col}"] = np.interp(ref_time, df["t_sec"], df[col])

    # reconstruct absolute timestamps
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
    
    # Load CSV safely
    df = pd.read_csv(
        file_path,
        header=None,
        names=["timestamp", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "extra"],
        index_col=False,
        engine="python",
    )

    # Clean timestamp column
    df["timestamp"] = df["timestamp"].astype(str).str.strip().str.strip(",")
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y_%m_%d_%H_%M_%S_%f", errors="coerce")
    df = df.dropna(subset=["timestamp"])

    # Convert timestamps to seconds since first record
    t0 = df["timestamp"].iloc[0]
    df["t_sec"] = (df["timestamp"] - t0).dt.total_seconds()

    # Handle duplicates by averaging over duplicates
    df = df.groupby("t_sec", as_index=False).mean(numeric_only=True)

    # Create new 500 Hz timeline (0.002 s spacing)
    t_interp = np.arange(df["t_sec"].iloc[0], df["t_sec"].iloc[-1], 1/500)

    # Interpolate each column
    interp_df = pd.DataFrame({"t_sec": t_interp})
    for col in ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]:
        f = interp1d(df["t_sec"], df[col], kind="cubic", fill_value="extrapolate")
        interp_df[col] = f(t_interp)

    # Create absolute timestamps relative to gait_start
    interp_df["timestamp"] = [
        (gait_start + timedelta(seconds=t)).strftime("%H:%M:%S.%f")[:-3]
        for t in interp_df["t_sec"]
    ]
    return interp_df


def create_kalman_filter(dt=0.002):
    # State: [angle, bias]
    F = np.array([[1, -dt],
                  [0,  1]])          # State transition (angle decreases by bias each step)

    B = np.array([[dt],
                  [0]])              # Control input (gyro rate)

    H = np.array([[1, 0]])           # We measure only the angle (from accelerometer)

    Q = np.array([[1e-5, 0],
                  [0, 1e-6]])        # Process noise covariance
    R = np.array([[1e-2]])           # Measurement noise covariance

    x0 = np.zeros((2, 1))            # Initial state [angle=0, bias=0]
    P0 = np.eye(2) * 0.01            # Initial covariance

    return KalmanFilter(F, B, H, Q, R, x0, P0)

def fuse_imu_data(df):
    dt = 0.002  # 500 Hz
    kf_x = create_kalman_filter(dt)
    kf_y = create_kalman_filter(dt)
    kf_z = create_kalman_filter(dt)

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

        # Z-axis (yaw — if magnetometer existed, we’d use it)
        kf_z.predict(u=np.array([[gyro[2,0]]]))
        fused_yaw.append(kf_z.estimate0[0,0])

    fused_df = df.copy()
    fused_df["pitch"] = np.degrees(fused_pitch)
    fused_df["roll"] = np.degrees(fused_roll)
    fused_df["yaw"] = np.degrees(fused_yaw)

    return fused_df


#Running plots and preproc data
ROOT = "/Users/nathangu/Desktop/Pytorch/NT/t8j8v4hnm4-1/Raw"  
SENSORS = ["LShank", "RShank", "Waist", "Arm"]
GAIT_START_STR = "2019-12-18 09:28:46.727"  # replace per-subject for different gait starts
GAIT_START = datetime.strptime(GAIT_START_STR, "%Y-%m-%d %H:%M:%S.%f")

subject_dirs = sorted([
    os.path.join(ROOT, d) for d in os.listdir(ROOT)
    if os.path.isdir(os.path.join(ROOT, d))
])

print(f"Found {len(subject_dirs)} subject folders under {ROOT}:\n", subject_dirs)


# Main loop

# Run this once, should allow you to download the data files

# url = 'https://drive.google.com/uc?export=download&id=1yc1evq9s3N7tfYX_vJbchFgKuwbFJOhJ?'
# response = requests.get(url)
# with open('local_filename.ext', 'wb') as file:
#     file.write(response.content)

root_dir = '/Users/nathangu/Desktop/Pytorch/NT/t8j8v4hnm4-1/Raw'

all_subjects_data = {}   # store results keyed by subject id (folder name)

for subj_path in subject_dirs:
    subj_id = os.path.basename(subj_path)
    print(f"\nSUBJECT {subj_id}")
    
    #Loading EEG / EMG
    vhdr_pattern = os.path.join(subj_path, "*.vhdr")
    vhdr_files = glob.glob(vhdr_pattern)
    if len(vhdr_files) == 0:
        print(f"No .vhdr found in {subj_path}, skipping EEG/EMG preprocessing for this subject.")
        raw_eeg = raw_emg = ica = raw_emg_filtered = None
        events = event_id = None
    else:
        vhdr_file = vhdr_files[0]  
        print(f"Found VHDR: {os.path.basename(vhdr_file)}")
        raw_eeg, raw_emg, ica, raw_emg_filtered = prep(vhdr_file, run_ica=True)
        raw_for_events = read_raw_brainvision(vhdr_file, preload=True)
        events, event_id = mne.events_from_annotations(raw_for_events)
        print(f"Loaded EEG/EMG. EMG channels: {raw_emg.ch_names if raw_emg is not None else 'None'}")
    
    #Load ACC / gyro
    subj_acc_data = {}
    for sensor in SENSORS:
        csv_path = os.path.join(subj_path, f"{sensor}.csv")
        if os.path.exists(csv_path):
            print(f"Loading ACC CSV: {sensor}.csv")
            df_acc = prep_acc_data(subj_path, sensor, GAIT_START)
            
            if df_acc is not None:
                print(f"{sensor}: {len(df_acc)} samples, "
                      f"t_sec range {df_acc['t_sec'].iloc[0]:.3f} - {df_acc['t_sec'].iloc[-1]:.3f}s")
                subj_acc_data[sensor] = df_acc
            else:
                print(f"rep_acc_data returned None for {sensor}")
        else:
            print(f"Missing ACC CSV: {sensor}.csv (skipping)")

    #Aligning sensors from ACC / gyro
    if len(subj_acc_data) == 0:
        print("No ACC sensors loaded for this subject.")
        aligned_data = None
    else:
        aligned_data = merge_all_sensors(subj_acc_data, GAIT_START)
        print(f"Merged accelerometer dataframe shape: {aligned_data.shape}")


    plot_combined_timeseries(raw_eeg, raw_emg_filtered, None, events, event_id, 30, 0)



    #Fusing ACC + gyro for IMU data for each sensor
    fused_imu_results = {}
    if aligned_data is not None:
        for sensor in subj_acc_data.keys():
            print(f"\nFusing IMU data for {sensor}")
            
            # Extract just this sensor’s columns
            sensor_cols = [c for c in aligned_data.columns if c.startswith(sensor)]
            if not sensor_cols:
                print(f"No matching columns found for {sensor}")
                continue
            
            df_sensor = aligned_data[["timestamp"] + sensor_cols].copy()
            df_sensor.columns = [c.replace(f"{sensor}_", "") for c in df_sensor.columns]  # normalize colnames
            
            # Apply fusion
            fused_df = fuse_imu_data(df_sensor)
            fused_imu_results[sensor] = fused_df
            
            print(f"{sensor} fused IMU shape: {fused_df.shape}")
            print(f"Fused IMU preview for {sensor}:")
            print(fused_df[['timestamp', 'pitch', 'roll', 'yaw']].head())
    else:
        print("No aligned accelerometer data to fuse IMU signals")
    #Store results
    all_subjects_data[subj_id] = {
        "raw_eeg": raw_eeg,
        "raw_emg": raw_emg,
        "raw_emg_filtered": raw_emg_filtered,
        "events": events,
        "event_id": event_id,
        "acc_dfs": subj_acc_data,
        "acc_aligned": aligned_data,
        "imu_fused": fused_imu_results
    }
    break

#Single subject data
merged_df = all_subjects_data["001"]["acc_aligned"]
imu_df = all_subjects_data["001"]["imu_fused"]
print("\nSample merged accelerometer data:")
print(merged_df.head())

print("\nSample fused IMU data for Waist:")
if "Waist" in imu_df:
    print(imu_df["Waist"].head())

# # quick summary
# print(f"\nProcessed {len(all_subjects_data)} subjects. Example keys for subject 001:")
# example = next(iter(all_subjects_data.keys()))
# print(example, list(all_subjects_data[example].keys()))


