import pandas as pd
import numpy as np
from scipy import signal

class FoG_Class:
    FOG, STOP, WALK = 1, 2, 3

    # ── Resampling ────────────────────────────────────────────────────────────

    def resample_imu_d(df, target_sfreq=128.0):
        """Resample IMU dataframe. Skips non-numeric columns."""
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
        Returns S ∈ R^{N x C} where:
          C = 6  if pitch/roll are absent (acc_x/y/z + gyro_x/y/z)
          C = 8  if pitch/roll are present (appended as channels 7-8)

        Acc:  divide by 8192 → g     (z-axis at rest ≈ 8000 ADC ≈ 1g)
        Gyro: divide by 131  → deg/s (±250 dps range, 16-bit)
        Pitch/Roll: already in degrees from fuse_imu_data_vectorized — kept as-is.
        Yaw is excluded (drifts on 6-DoF IMU without magnetometer).
        """
        ACC_SENS  = 8192.0
        GYRO_SENS = 131.0

        # Core 6 channels
        S = df[["acc_x", "acc_y", "acc_z",
                "gyro_x", "gyro_y", "gyro_z"]].to_numpy(dtype=np.float32)
        S[:, 0:3] /= ACC_SENS   # → g
        S[:, 3:6] /= GYRO_SENS  # → deg/s

        # Append pitch and roll if available (already in degrees, no conversion needed)
        if "pitch" in df.columns and "roll" in df.columns:
            pitch_roll = df[["pitch", "roll"]].to_numpy(dtype=np.float32)
            S = np.concatenate([S, pitch_roll], axis=1)  # (N, 8)

        return S

    # ── Sliding Windows ───────────────────────────────────────────────────────

    def sliding_windows(S, window_len=384, step=96):
        """
        S: (N, 6) → X: (K, window_len, 6)

        step=96 gives 0.75s step (75% overlap) at 128 Hz.
        This reduces correlation between consecutive windows vs the original
        step=32 (0.25s, 91.7% overlap) while still capturing enough context.

        For reference:
          step=32  → 91.7% overlap (original — very redundant)
          step=64  → 83.3% overlap
          step=96  → 75.0% overlap  ← recommended
          step=128 → 66.7% overlap
        """
        N = S.shape[0]
        K = (N - window_len) // step + 1
        X = np.zeros((K, window_len, S.shape[1]), dtype=np.float32)
        for k in range(K):
            s = k * step
            X[k] = S[s:s + window_len]
        return X

    # ── Labeling ─────────────────────────────────────────────────────────────

    def build_fog_mask(t_sec, fog_events):
        fog_mask = np.zeros(len(t_sec), dtype=np.int8)
        for start, end in fog_events:
            fog_mask |= ((t_sec >= start) & (t_sec <= end)).astype(np.int8)
        return fog_mask

    def label_sliding_windows(t_sec, fog_mask, sfreq=128, window_sec=3.0,
                               overlap_sec=0.25, fog_class=1, nonfog_class=0):
        """
        Returns y: (K,) — 0=NonFoG, 1=FoG.
        overlap_sec should match the step used in sliding_windows.
        """
        W_l  = int(window_sec * sfreq)        # 384
        step = int(overlap_sec * sfreq)        # 96 at 0.75s
        N    = len(t_sec)
        K    = (N - W_l) // step + 1
        y    = np.zeros(K, dtype=np.int64)
        min_fog_samples = W_l // 2             # ≥ 1.5s of FoG in window

        for k in range(K):
            start = k * step
            end   = start + W_l
            if fog_mask[start:end].sum() >= min_fog_samples:
                y[k] = fog_class
            else:
                y[k] = nonfog_class
        return y

    def build_window_labels(df_imu_128, fog_events, overlap_sec=0.75):
        t_sec    = df_imu_128["t_sec"].to_numpy()
        fog_mask = FoG_Class.build_fog_mask(t_sec, fog_events)
        y        = FoG_Class.label_sliding_windows(
                       t_sec, fog_mask, overlap_sec=overlap_sec)
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