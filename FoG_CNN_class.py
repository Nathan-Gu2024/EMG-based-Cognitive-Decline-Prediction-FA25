# These data are fed to the CNN classifier to detect FoG episodes (Class 1) from Walking-with-Turns (Class 2) and Stops (Class 3) 
# in a sliding window manner. Sensitivity and specificity scores from the CNN implementation are calculated and 
# compared with the respective ones from traditional Machine Learning classifiers. 

# Sample at 128 Hz
# N x 6 matrix, where N is the samples
# Structre: 𝑆𝑗 = [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
# three-second (𝑊𝑙 = 3 s) sliding window with 0.25 s overlap is being selected for processing the 𝑆𝑗 data
# The resulted 𝑆𝑗 matrix is being reshaped to 𝐾×𝑊𝑙×6, where 𝐾= 𝑁−𝑊𝑙 / 𝑆 +1 is the number of segments created
# segments used to construct the training set for the CNN
# labeled as “FoG” whether FoG episode exists inside window and whether FoG event starts and ends inside the window
# captures even short FoG events (≤𝑊𝑙/2)
# If FoG event is not completely inside a window, segment labeled as “FoG” only when the duration of the FoG is ≥𝑊𝑙/2
# ^^ basically will be labeled if over half of the window is FoG

#Get IMU data -> 100 : 10, max pool, 40:10, gloabal average pooling layer -> fully-connected layer -> classes probability

import pandas as pd
import numpy as np

class FoG_Class:
    FOG, STOP, WALK = 1, 2, 3

    def resample_imu_d(df, target_sfreq=128.0):
        """
        Resample IMU dataframe to target sampling frequency.
        Works with acc-only OR acc+gyro data.
        """
        t = df["t_sec"].to_numpy()
        t_new = np.arange(t[0], t[-1], 1.0 / target_sfreq)

        out = {"t_sec": t_new}

        for col in df.columns:
            if col == "t_sec":
                continue
            out[col] = np.interp(t_new, t, df[col].to_numpy())

        return pd.DataFrame(out)
        
    def build_S_matrix(df):
        """
        Returns S ∈ R^{N x 6}
        """
        return df[[
            "acc_x", "acc_y", "acc_z",
            "gyro_x", "gyro_y", "gyro_z"
        ]].to_numpy(dtype=np.float32)
    

    def label_sliding_windows_multiclass(t_sec, fog_mask, stop_mask, sfreq=128, window_sec=3.0, overlap_sec=0.25):
        W_l = int(window_sec * sfreq)
        step = int(overlap_sec * sfreq)
        K = (len(t_sec) - W_l) // step + 1

        y = np.zeros(K, dtype=np.int64)
        half = W_l // 2

        for k in range(K):
            s = k * step
            e = s + W_l

            fog_count  = fog_mask[s:e].sum()
            stop_count = stop_mask[s:e].sum()

            if fog_count >= half:
                y[k] = FoG_Class.FOG
            elif stop_count >= half:
                y[k] = FoG_Class.STOP
            else:
                y[k] = FoG_Class.WALK

        return y


    def build_cnn_input_from_imu(df_imu):
        """
        df_imu: fused or raw IMU DataFrame for ONE sensor
        """
        df_128 = FoG_Class.resample_imu_df(df_imu, target_sfreq=128.0)
        S = FoG_Class.build_S_matrix(df_128)          # (N, 6)
        X = FoG_Class.sliding_windows(S)              # (K, 384, 6)
        return X
    

    def build_fog_mask(t_sec, fog_events):
        """
        t_sec: (N,) array of timestamps in seconds
        fog_events: list of (start_sec, end_sec)
        Returns:
            fog_mask: (N,) binary array
        """
        fog_mask = np.zeros(len(t_sec), dtype=np.int8)

        for start, end in fog_events:
            fog_mask |= ((t_sec >= start) & (t_sec <= end)).astype(np.int8)

        return fog_mask

    def label_sliding_windows(t_sec, fog_mask, sfreq=128, window_sec=3.0, overlap_sec=0.25, fog_class=1, nonfog_class=2):
        """
        Returns:
            y: (K,) window labels
        """
        W_l = int(window_sec * sfreq)     # 384
        step = int(overlap_sec * sfreq)   # 32
        N = len(t_sec)

        K = (N - W_l) // step + 1
        y = np.zeros(K, dtype=np.int64)

        min_fog_samples = W_l // 2        # >= 1.5s

        for k in range(K):
            start = k * step
            end = start + W_l

            fog_count = fog_mask[start:end].sum()

            if fog_count >= min_fog_samples:
                y[k] = fog_class
            else:
                y[k] = nonfog_class

        return y

    def build_window_labels(df_imu_128, fog_events):
        """
        df_imu_128: resampled IMU dataframe (128 Hz)
        fog_events: list of (start_sec, end_sec)
        """
        t_sec = df_imu_128["t_sec"].to_numpy()

        fog_mask = FoG_Class.build_fog_mask(t_sec, fog_events)
        y = FoG_Class.label_sliding_windows(t_sec, fog_mask)

        return y
    

