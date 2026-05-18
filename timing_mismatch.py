"""
diagnose_timing_mismatches.py

Investigates why some tasks have "no sensor data in task window".
Shows the time gap between raw IMU recording and filtered task times.
"""

import os
import pandas as pd
from datetime import datetime
from extract_fog_labels import load_raw_sensor_csv, load_filtered_txt

# RAW_ROOT      = "/content/drive/MyDrive/t8j8vhnm4-1/Raw"
# FILTERED_ROOT = "/content/drive/MyDrive/Filtered_Data"
SENSOR        = "Waist"

def diagnose_subject(subj_id: str):
    """
    Compare raw sensor timeline vs filtered task timelines for one subject.
    Checks all available sensors.
    """
    raw_path      = os.path.join(RAW_ROOT, subj_id)
    filtered_path = os.path.join(FILTERED_ROOT, subj_id)
    
    if not os.path.isdir(filtered_path):
        print(f"[{subj_id}] No filtered folder")
        return
    
    # Find any available sensor CSV
    available_sensors = []
    for sensor in ["Waist", "LShank", "RShank", "Arm"]:
        csv_path = os.path.join(raw_path, f"{sensor}.csv")
        if os.path.exists(csv_path):
            available_sensors.append((sensor, csv_path))
    
    if not available_sensors:
        print(f"[{subj_id}] No sensor CSVs found at all")
        return
    
    print(f"\n{'='*70}")
    print(f"SUBJECT {subj_id}")
    print(f"{'='*70}")
    print(f"Available sensors: {', '.join([s for s, _ in available_sensors])}")
    
    # Check each sensor
    for sensor_name, csv_path in available_sensors:
        print(f"\n--- {sensor_name} ---")
        
        raw_sensor_df = load_raw_sensor_csv(csv_path)
        recording_date = raw_sensor_df["timestamp"].iloc[0].date()
        
        raw_start = raw_sensor_df["timestamp"].iloc[0]
        raw_end   = raw_sensor_df["timestamp"].iloc[-1]
        
        print(f"  Recording: {raw_start.strftime('%H:%M:%S')} → {raw_end.strftime('%H:%M:%S')} ({(raw_end - raw_start).total_seconds() / 60:.1f} min)")
        
        # Check each task
        import glob
        task_files = sorted(glob.glob(os.path.join(filtered_path, "task_*.txt")))
        
        for task_path in task_files:
            task_name = os.path.splitext(os.path.basename(task_path))[0]
            task_df = load_filtered_txt(task_path, recording_date)
            
            task_start = task_df["timestamp"].iloc[0]
            task_end   = task_df["timestamp"].iloc[-1]
            
            # Check overlap
            overlap = (task_start <= raw_end) and (task_end >= raw_start)
            
            if overlap:
                status = "✓"
            else:
                status = "✗"
            
            print(f"    {task_name}: {status}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_root', type=str, required=True, help="Path to raw data")
    parser.add_argument('--filtered_root', type=str, required=True, help="Path to filtered data")
    args = parser.parse_args()

    # Update global variables so the diagnose_subject function uses them
    RAW_ROOT = args.raw_root
    FILTERED_ROOT = args.filtered_root

    # Get all subjects in the filtered folder
    import os
    if os.path.exists(FILTERED_ROOT):
        subjects = sorted([d for d in os.listdir(FILTERED_ROOT) if not d.startswith('.')])
        for subj in subjects:
            diagnose_subject(subj)
    else:
        print(f"Could not find filtered root: {FILTERED_ROOT}")