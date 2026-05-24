"""
prep_daphnet.py
Processes the pre-downloaded Daphnet dataset from Google Drive into overlapping 
windows ready for train_fog_cnn.py via command-line arguments.
"""

import os
import glob
import json
import numpy as np
import pandas as pd
import argparse

def process_daphnet(raw_dir, save_dir, window_sec=2.0, overlap_sec=0.25, sfreq=64):
    print(f"\nLooking for Daphnet files in: {raw_dir}")
    txt_files = sorted(glob.glob(os.path.join(raw_dir, "*.txt")))
    
    if not txt_files:
        print("ERROR: No .txt files found! Please double check your --raw_dir path.")
        return

    print(f"Found {len(txt_files)} files. Processing...\n")
    
    W_l = int(window_sec * sfreq)       # 128 samples at 64Hz
    step = int(overlap_sec * sfreq)     # 16 samples
    
    all_X = []
    all_y = []
    subject_indices = []
    
    # Group by Subject (S01, S02...)
    files_by_subject = {}
    for f in txt_files:
        subj_id = os.path.basename(f)[:3]
        if subj_id not in files_by_subject:
            files_by_subject[subj_id] = []
        files_by_subject[subj_id].append(f)
        
    for subj_id, files in files_by_subject.items():
        subj_X = []
        subj_y = []
        
        for file in files:
            # Columns: Time, Ankle(x,y,z), Thigh(x,y,z), Trunk(x,y,z), Annotation
            df = pd.read_csv(file, sep=' ', header=None)
            df.columns = ['Time', 'A_x', 'A_y', 'A_z', 'Th_x', 'Th_y', 'Th_z', 'Tr_x', 'Tr_y', 'Tr_z', 'Label']
            
            # Extract sensors and convert to milli-g (Daphnet data is in mg)
            S = df[['A_x', 'A_y', 'A_z', 'Th_x', 'Th_y', 'Th_z', 'Tr_x', 'Tr_y', 'Tr_z']].to_numpy() / 1000.0
            labels = df['Label'].to_numpy()
            
            # Sliding Windows
            N = len(S)
            K = (N - W_l) // step + 1
            
            if K <= 0: continue
            
            for k in range(K):
                start = k * step
                end = start + W_l
                win_labels = labels[start:end]
                
                # Skip windows not part of the experiment protocol (Label 0)
                if 0 in win_labels:
                    continue
                    
                # If window has >= 50% FoG (Daphnet class 2), label it 1. Else 0.
                is_fog = (win_labels == 2).sum() >= (W_l // 2)
                
                subj_X.append(S[start:end])
                subj_y.append(1 if is_fog else 0)
                
        if len(subj_X) == 0:
            print(f"Skipping {subj_id} - no valid protocol data found.")
            continue
            
        subj_X_np = np.stack(subj_X)
        subj_y_np = np.array(subj_y)
        
        # Z-Score normalization per subject
        mean = subj_X_np.mean(axis=(0,1), keepdims=True)
        std = subj_X_np.std(axis=(0,1), keepdims=True) + 1e-8
        subj_X_np = (subj_X_np - mean) / std
        
        start_idx = sum(len(x) for x in all_X)
        end_idx = start_idx + len(subj_X_np)
        
        all_X.append(subj_X_np)
        all_y.append(subj_y_np)
        
        subject_indices.append({
            "subject_id": subj_id,
            "num_windows": len(subj_X_np),
            "start_idx": start_idx,
            "end_idx": end_idx
        })
        print(f"[{subj_id}] {len(subj_X_np)} windows | FoG: {(subj_y_np==1).sum()} | NonFoG: {(subj_y_np==0).sum()}")
        
    X_out = np.concatenate(all_X, axis=0)
    y_out = np.concatenate(all_y, axis=0)
    
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "X_windows_all_subjects.npy"), X_out)
    np.save(os.path.join(save_dir, "y_windows_all_subjects.npy"), y_out)
    with open(os.path.join(save_dir, "subject_indices.json"), "w") as f:
        json.dump(subject_indices, f, indent=2)
        
    print(f"\nSuccess! Saved matrices to {save_dir}")
    print(f"Final Combined Shape: X={X_out.shape}, y={y_out.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Daphnet dataset into overlapping windows.")
    parser.add_argument('--raw_dir', type=str, required=True, help="Path to the raw Daphnet .txt files")
    parser.add_argument('--save_dir', type=str, required=True, help="Directory to save the processed .npy arrays")
    parser.add_argument('--window_sec', type=float, default=2.0, help="Sliding window length in seconds")
    parser.add_argument('--overlap_sec', type=float, default=0.25, help="Overlap length in seconds")
    
    args = parser.parse_args()
    
    process_daphnet(
        raw_dir=args.raw_dir, 
        save_dir=args.save_dir, 
        window_sec=args.window_sec, 
        overlap_sec=args.overlap_sec
    )