"""
train_cnn.py

Preprocessing pipeline fixes:
- Skips fusion step entirely (pitch/roll/yaw not used by CNN)
- Converts raw ADC values to physical units in build_S_matrix
- Applies per-window z-score normalization
- Labels are already 0-indexed (0=NonFoG, 1=FoG)
- Removes DEBUG prints
"""

import os
import json
import numpy as np
import torch
import argparse
from torch.utils.data import DataLoader

from FoG_CNN_class import FoG_Class as FC
from fog_dataset import FoGDataset
from extract_fog_labels import process_all_subjects

import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_root', type=str, required=True, help="Path to raw data")
    parser.add_argument('--filtered_root', type=str, required=True, help="Path to filtered data")
    parser.add_argument('--save_dir', type=str, default='.', help="Where to save the .npy files")
    args = parser.parse_args()

    RAW_ROOT      = args.raw_root
    FILTERED_ROOT = args.filtered_root
    SENSOR        = "Waist"



    # ── Load all subjects + tasks ─────────────────────────────────────────────
    print("Loading and aligning data...")
    all_data = process_all_subjects(RAW_ROOT, FILTERED_ROOT, sensor=SENSOR)

    # ── Build X and y per subject ─────────────────────────────────────────────
    all_X = []
    all_y = []
    subject_indices = []

    for subj_id, tasks in all_data.items():
        subj_X = []
        subj_y = []

        for task_name, task in tasks.items():
            print(f"\n[{subj_id}/{task_name}]")

            raw_sensor_df = task["raw_sensor_df"]
            fog_events    = task["fog_events"]

            if raw_sensor_df.empty:
                print(f"  Skipping — no sensor data in task window")
                continue

            # Step 1: Resample raw acc+gyro to 128 Hz
            # NOTE: No fusion needed — CNN uses acc+gyro directly, not pitch/roll/yaw
            df_128 = FC.resample_imu_d(raw_sensor_df, target_sfreq=128.0)

            # Step 2: Build signal matrix (converts ADC → physical units internally)
            S = FC.build_S_matrix(df_128)   # (N, 6) in g and deg/s

            # Step 3: Sliding windows
            X = FC.sliding_windows(S)       # (K, 384, 6)

            # Step 4: Per-window z-score normalization
            X = FC.normalize_windows(X)     # (K, 384, 6) mean=0, std=1 per channel per window

            # Step 5: Build labels (0=NonFoG, 1=FoG)
            y = FC.build_window_labels(df_128, fog_events)

            print(f"  X: {X.shape}, y: {y.shape}")
            print(f"  FoG events: {len(fog_events)}")
            print(f"  Label dist: {dict(zip(*np.unique(y, return_counts=True)))}")

            subj_X.append(X)
            subj_y.append(y)

        if not subj_X:
            continue

        subj_X_all = np.concatenate(subj_X, axis=0)
        subj_y_all = np.concatenate(subj_y, axis=0)

        start_idx = sum(len(a) for a in all_X)
        end_idx   = start_idx + len(subj_X_all)

        all_X.append(subj_X_all)
        all_y.append(subj_y_all)

        subject_indices.append({
            "subject_id": subj_id,
            "num_windows": len(subj_X_all),
            "start_idx": start_idx,
            "end_idx": end_idx
        })

        print(f"\n[{subj_id}] Total: {len(subj_X_all)} windows | "
              f"FoG: {(subj_y_all == 1).sum()} | "
              f"NonFoG: {(subj_y_all == 0).sum()}")

    # ── Combine ───────────────────────────────────────────────────────────────
    X_combined = np.concatenate(all_X, axis=0)
    y_combined = np.concatenate(all_y, axis=0)

    # ── 1. Apply Subject-Level Normalization ──
    print("Applying subject-level normalization...")
    X_norm = X_combined.copy()
    for subj in subject_indices:
        s, e = subj['start_idx'], subj['end_idx']
        subj_data = X_combined[s:e]
        mean = subj_data.mean(axis=(0,1), keepdims=True)  
        std  = subj_data.std(axis=(0,1),  keepdims=True)  
        std  = np.where(std < 1e-8, 1e-8, std)
        X_norm[s:e] = (subj_data - mean) / std

    # ── 2. Add Pre-FoG Class (3-Class Labels) ──
    print("Applying Pre-FoG 3-class labeling...")
    PRE_FOG_STEPS = 8 # ~2 seconds
    y_new = np.zeros_like(y_combined)
    
    for subj in subject_indices:
        s, e = subj['start_idx'], subj['end_idx']
        y_subj = y_combined[s:e]
        y_subj_new = np.zeros_like(y_subj)
        y_subj_new[y_subj == 1] = 2   # FoG -> class 2

        # Find onsets
        diff = np.diff(y_subj, prepend=0)
        onsets = np.where(diff == 1)[0]
        
        for onset in onsets:
            start = max(0, onset - PRE_FOG_STEPS)
            for i in range(start, onset):
                if y_subj_new[i] == 0:
                    y_subj_new[i] = 1 # Pre-FoG -> class 1
        
        y_new[s:e] = y_subj_new

    # ── Save the updated arrays ──
    save_dir = args.save_dir if hasattr(args, 'save_dir') else '.'
    
    np.save(os.path.join(save_dir, "X_windows_all_subjects.npy"), X_norm)
    np.save(os.path.join(save_dir, "y_windows_all_subjects.npy"), y_new)
    
    with open(os.path.join(save_dir, "subject_indices.json"), "w") as f:
        json.dump(subject_indices, f, indent=2)