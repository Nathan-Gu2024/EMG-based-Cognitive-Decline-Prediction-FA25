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
from scipy import signal

class FoG_Class:
    FOG, STOP, WALK = 1, 2, 3

    # ── Resampling ────────────────────────────────────────────────────────────

    def resample_imu_d(df, target_sfreq=128.0):
        """
        Resample IMU dataframe to target sampling frequency.
        Skips non-numeric columns (timestamp etc).
        """
        t = df["t_sec"].to_numpy()
        t_new = np.arange(t[0], t[-1], 1.0 / target_sfreq)
        out = {"t_sec": t_new}

        for col in df.columns:
            if col == "t_sec":
                continue
            if df[col].dtype.kind in ['O', 'M', 'U']:
                continue
            try:
                out[col] = np.interp(t_new, t, df[col].to_numpy())
            except (TypeError, ValueError):
                continue

        return pd.DataFrame(out)

    # ── Signal Matrix ─────────────────────────────────────────────────────────

    def build_S_matrix(df):
        """
        Returns S ∈ R^{N x 6}: raw acc + gyro (no fusion needed).
        Converts raw ADC values to physical units:
          acc  → g   (divide by 8192, assuming ±4g range / 16-bit)
          gyro → deg/s (divide by 131, assuming ±250 dps range / 16-bit)
        NOTE: Adjust ACC_SENS and GYRO_SENS if your sensor uses different ranges.
        """
        ACC_SENS  = 8192.0   # LSB/g   — ±4g range, 16-bit  (8000 ADC ≈ 1g at rest confirms this)
        GYRO_SENS = 131.0    # LSB/(deg/s) — ±250 dps range, 16-bit

        S = df[["acc_x", "acc_y", "acc_z",
                "gyro_x", "gyro_y", "gyro_z"]].to_numpy(dtype=np.float32)

        S[:, 0:3] /= ACC_SENS   # acc channels → g
        S[:, 3:6] /= GYRO_SENS  # gyro channels → deg/s

        return S

    # ── Per-window Z-score normalization ─────────────────────────────────────

    def normalize_windows(X):
        """
        Per-window, per-channel z-score normalization.
        X: (K, 384, 6) → same shape, each channel in each window has mean=0, std=1.
        This is critical for CNN stability when channels have different scales.
        """
        mean = X.mean(axis=1, keepdims=True)   # (K, 1, 6)
        std  = X.std(axis=1, keepdims=True)    # (K, 1, 6)
        std  = np.where(std < 1e-8, 1e-8, std) # avoid divide-by-zero
        return (X - mean) / std

    # ── Sliding Windows ───────────────────────────────────────────────────────

    def sliding_windows(S, window_len=384, step=32):
        """
        S: (N, 6) → X: (K, 384, 6)
        """
        N = S.shape[0]
        K = (N - window_len) // step + 1
        X = np.zeros((K, window_len, 6), dtype=np.float32)
        for k in range(K):
            s = k * step
            X[k] = S[s:s + window_len]
        return X

    # ── Labeling ─────────────────────────────────────────────────────────────

    def build_fog_mask(t_sec, fog_events):
        """
        t_sec: (N,) array of timestamps in seconds
        fog_events: list of (start_sec, end_sec)
        Returns fog_mask: (N,) binary array
        """
        fog_mask = np.zeros(len(t_sec), dtype=np.int8)
        for start, end in fog_events:
            fog_mask |= ((t_sec >= start) & (t_sec <= end)).astype(np.int8)
        return fog_mask

    def label_sliding_windows(t_sec, fog_mask, sfreq=128, window_sec=3.0,
                               overlap_sec=0.25, fog_class=1, nonfog_class=0):
        """
        Returns y: (K,) window labels (0-indexed: 0=NonFoG, 1=FoG)
        """
        W_l  = int(window_sec * sfreq)     # 384
        step = int(overlap_sec * sfreq)    # 32
        N    = len(t_sec)
        K    = (N - W_l) // step + 1
        y    = np.zeros(K, dtype=np.int64)
        min_fog_samples = W_l // 2         # ≥ 1.5s

        for k in range(K):
            start = k * step
            end   = start + W_l
            if fog_mask[start:end].sum() >= min_fog_samples:
                y[k] = fog_class
            else:
                y[k] = nonfog_class

        return y

    def build_window_labels(df_imu_128, fog_events):
        """
        df_imu_128: resampled IMU dataframe (128 Hz)
        fog_events: list of (start_sec, end_sec)
        Returns y: (K,) with 0=NonFoG, 1=FoG
        """
        t_sec    = df_imu_128["t_sec"].to_numpy()
        fog_mask = FoG_Class.build_fog_mask(t_sec, fog_events)
        y        = FoG_Class.label_sliding_windows(t_sec, fog_mask)
        return y

    # ── Feature extraction (for traditional ML baseline) ─────────────────────

    def extract_features(X):
        feats = []
        for w in X:
            f  = w.mean(axis=0).tolist()
            f += w.std(axis=0).tolist()
            f += np.max(w, axis=0).tolist()
            f += np.min(w, axis=0).tolist()
            feats.append(f)
        return np.array(feats)