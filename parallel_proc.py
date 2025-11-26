import os
import glob
import concurrent.futures
import multiprocessing
import traceback
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm

from prep import Prep    
from kalman_filter import KalmanFilter

class Parallel: 
    def process_subject(subj_path, gait_start_iso, sensors):
        """
        Processes one subject folder and returns a picklable dict:
        {
        "subj_id": "<id>",
        "raw_eeg": { "data": np.array, "ch_names": [...], "sfreq": float }  OR None,
        "raw_emg_filtered": { "data": np.array, "ch_names": [...], "sfreq": float } OR None,
        "events": events_array OR None,
        "event_id": event_id_dict OR None,
        "acc_aligned": pandas.DataFrame OR None,
        "imu_fused_vec": { sensor_name: pandas.DataFrame, ... },
        "imu_fused_kal": { sensor_name: pandas.DataFrame, ... }
        }
        """
        try:
            # prevent MKL/OpenMP oversubscription inside workers
            os.environ["OMP_NUM_THREADS"] = "1"
            os.environ["MKL_NUM_THREADS"] = "1"
            os.environ["OPENBLAS_NUM_THREADS"] = "1"

            # imports local to worker so top-level script runs fast to spawn workers
            import mne
            from mne.io import read_raw_brainvision

            # convert gait_start string to datetime
            gait_start = datetime.fromisoformat(gait_start_iso)

            subj_id = os.path.basename(subj_path)
            vhdr_files = glob.glob(os.path.join(subj_path, "*.vhdr"))

            # default outputs
            raw_eeg_picklable = None
            raw_emg_filtered_picklable = None
            events = None
            event_id = None

            if len(vhdr_files) > 0:
                vhdr = vhdr_files[0]
                raw_eeg, raw_emg, ica, raw_emg_filtered = Prep.prep(vhdr, run_ica=True)

                #Convert Raw -> picklable summary (arrays + meta)
                if raw_eeg is not None:
                    raw_eeg_picklable = {
                        "data": raw_eeg.get_data(),
                        "ch_names": list(raw_eeg.ch_names),
                        "sfreq": float(raw_eeg.info["sfreq"])
                    }
                if raw_emg_filtered is not None:
                    raw_emg_filtered_picklable = {
                        "data": raw_emg_filtered.get_data(),
                        "ch_names": list(raw_emg_filtered.ch_names),
                        "sfreq": float(raw_emg_filtered.info["sfreq"])
                    }

                #events
                raw_for_events = read_raw_brainvision(vhdr, preload=False)
                events, event_id = mne.events_from_annotations(raw_for_events)

            #ACC processing
            subj_acc_data = {}
            for sensor in sensors:
                csv_path = os.path.join(subj_path, f"{sensor}.csv")
                if os.path.exists(csv_path):
                    df_acc = Prep.prep_acc_data(subj_path, sensor, gait_start)
                    if df_acc is not None:
                        subj_acc_data[sensor] = df_acc

            acc_aligned = None
            if subj_acc_data:
                acc_aligned = Prep.merge_all_sensors(subj_acc_data, gait_start)

            #IMU fusion (vectorized and/or kalman) per sensor
            imu_fused_vec = {}
            imu_fused_kal = {}
            if acc_aligned is not None:
                for sensor, df_sensor in subj_acc_data.items():
                    #reconstruct the sensor-specific DataFrame from acc_aligned
                    cols = [c for c in acc_aligned.columns if c.startswith(sensor)]
                    if not cols:
                        continue
                    df_for_fuse = acc_aligned[["t_sec", "timestamp"] + cols].copy()
                    #rename to plain column names for fuse functions
                    df_for_fuse.columns = ["t_sec", "timestamp"] + [c.replace(f"{sensor}_", "") for c in cols]

                    #vectorized fusion
                    try:
                        vec_df = Prep.fuse_imu_data_vectorized(df_for_fuse, sfreq=500.0, alpha=0.98)
                    except Exception:
                        vec_df = Prep.fuse_imu_data(df_for_fuse)  #fallback (per-row)
                    imu_fused_vec[sensor] = vec_df

                    # Kalman variant
                    # Only compute Kalman if enabled
                    if os.environ.get("USE_KALMAN", "False") == "True":
                        try:
                            kal_df = Prep.fuse_imu_data_kalman(...)
                            imu_fused_kal[sensor] = kal_df
                        except:
                            pass

            # prepare result
            result = {
                "subj_id": subj_id,
                "raw_eeg": raw_eeg_picklable,
                "raw_emg_filtered": raw_emg_filtered_picklable,
                "events": events,
                "event_id": event_id,
                "acc_aligned": acc_aligned,
                "imu_fused_vec": imu_fused_vec,
                "imu_fused_kal": imu_fused_kal,
            }
            return {"ok": True, "result": result}

        except Exception as e:
            tb = traceback.format_exc()
            return {"ok": False, "error": str(e), "traceback": tb, "subj": subj_path}
        except BaseException as e:
            return {"ok": False, "error": f"FATAL: {e}", "traceback": "Non-Python crash", "subj": subj_path}



    #Parallel runner
    def run_parallel(root_dir, gait_start, sensors, max_workers=None):
        subject_dirs = sorted([
            os.path.join(root_dir, d) for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])
        if not subject_dirs:
            raise ValueError("No subject folders found under: " + root_dir)

        #sensible default for workers
        cpu_count = multiprocessing.cpu_count()
        if max_workers is None:
            max_workers = max(1, min(cpu_count - 1, 6))

        all_results = {}
        print(f"Launching processing for {len(subject_dirs)} subjects with {max_workers} workers...")
        tasks = []
        #use ProcessPoolExecutor
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as exe:
            futures = {
                exe.submit(Parallel.process_subject, subj_path, gait_start.isoformat(), sensors): subj_path
                for subj_path in subject_dirs
            }
            #progress bar
            for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                subj_path = futures[fut]
                res = fut.result()
                if res.get("ok"):
                    r = res["result"]
                    all_results[r["subj_id"]] = r
                else:
                    print("ERROR processing", subj_path)
                    print(res.get("error"))
                    print(res.get("traceback"))
        return all_results
