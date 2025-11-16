import numpy as np
import mne
import matplotlib.pyplot as plt
from scipy.signal import *


class Plots:
    def __init__(self):
        pass

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
    #imu_sensors: stores the abosolute timestamp, pitch, roll, and yaw for each sensor
    #IMU: pitch, roll, yaw
    #Fix the scaling on the IMU
    def plot_combined_timeseries(raw_eeg, raw_emg_filtered, imu_dict, events, event_id, duration, start):
        eeg_data = None
        eeg_channels = []
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

        if sfreq is None:
            raise ValueError("No EEG/EMG sampling rate found.")

        n_samples = int(duration * sfreq)
        start_sample = int(start * sfreq)
        time_axis = np.arange(n_samples) / sfreq



        n_emg = len(emg_channels)
        total_plots = 2 + n_emg
        height_ratios = [1.2] + [1.25] * n_emg + [1.0]

        fig, axes = plt.subplots(
            total_plots, 1,
            figsize=(16, 3 + 2.2 * total_plots),
            sharex=True,
            gridspec_kw={"height_ratios": height_ratios},
            constrained_layout=True
        )


        #Heamap for EEG
        eeg_segment = eeg_data[:, start_sample:start_sample + n_samples]
        eeg_z = (eeg_segment - eeg_segment.mean(axis=1, keepdims=True)) / eeg_segment.std(axis=1, keepdims=True)

        im = axes[0].imshow(
            eeg_z, aspect="auto", origin="lower",
            extent=[0, duration, 0, len(eeg_channels)],
            cmap="RdBu_r", vmin=-3, vmax=3
        )
        axes[0].set_title("EEG (z-scored heatmap)")
        axes[0].set_ylabel("Channels")
        axes[0].set_yticks(np.arange(len(eeg_channels)) + 0.5)
        axes[0].set_yticklabels(eeg_channels, fontsize=6)
        fig.colorbar(im, ax=axes[0], shrink=0.8)

        #Spectro for EMG
        for idx, ch in enumerate(emg_channels):
            ax = axes[idx + 1]
            emg_segment = emg_data[idx, start_sample:start_sample + n_samples]

            f, t, Sxx = spectrogram(
                emg_segment, fs=sfreq,
                nperseg=1024, noverlap=512
            )
            Sxx_db = 10 * np.log10(Sxx + 1e-12)

            t_edges = np.linspace(0, duration, Sxx_db.shape[1] + 1)
            f_edges = np.linspace(f[0], f[-1], Sxx_db.shape[0] + 1)

            im = ax.pcolormesh(t_edges, f_edges, Sxx_db, shading="auto", cmap="viridis")
            ax.set_title(f"EMG: {ch}")
            ax.set_ylabel("Hz")
            ax.set_ylim(0, 250)

            if idx == len(emg_channels) - 1:
                fig.colorbar(im, ax=ax, shrink=0.7)

        #IMU plot
        imu_ax = axes[-1]
        
        for sensor_name, df in imu_dict.items():
            if not isinstance(df, dict) and hasattr(df, "columns"):

                if "t_sec" in df.columns:
                    time_sec = df["t_sec"].to_numpy()
                elif "timestamp" in df.columns:
                    time_sec = df["timestamp"].to_numpy() - df["timestamp"].iloc[0]
                else:
                    print(f"Skipping {sensor_name}: no valid time column")
                    continue

                # window mask
                mask = (time_sec >= start) & (time_sec <= start + duration)

                # which channels exist
                channels = [c for c in ["pitch", "roll", "yaw"] if c in df.columns]
                if not channels:
                    continue

                segment = df.loc[mask, channels].to_numpy().T
                time_rel = time_sec[mask]

                # offset each line
                for i, ch in enumerate(channels):
                    imu_ax.plot(time_rel - start, segment[i] + i * 0.5,
                                label=f"{sensor_name} - {ch}")

        imu_ax.set_title("IMU Fused Orientation (Pitch, Roll, Yaw)")
        imu_ax.set_ylabel("Angle")
        imu_ax.set_xlabel("Time (s)")
        imu_ax.grid(alpha=0.3)
        imu_ax.legend(fontsize=6, loc="upper left")

        if events is not None:
            for e in events:
                t_abs = e[0] / sfreq
                t_rel = t_abs - start
                if 0 <= t_rel <= duration:
                    label = [k for k, v in event_id.items() if v == e[2]]
                    imu_ax.axvline(t_rel, color="red", linestyle="--")
                    if label:
                        imu_ax.text(t_rel, imu_ax.get_ylim()[1]*0.9,
                                    label[0], fontsize=6, rotation=90)

        plt.show()


        # min_len = n_samples
        # if eeg_data is not None:
        #     min_len = min(min_len, eeg_data.shape[1] - start_sample)
        # if emg_data is not None:
        #     min_len = min(min_len, emg_data.shape[1] - start_sample)
        # if acc_data is not None:
        #     min_len = min(min_len, acc_data.shape[1] - start_sample)
        # n_samples = max(0, min_len)
        # duration = n_samples / sfreq
        # time_axis = np.arange(n_samples) / sfreq


        # n_emg = len(emg_channels)
        # total_plots = 2 + n_emg #EEG heatmap, EMG, ACC

        # height_ratios = [1.2] + [1.25] * n_emg + [0.6]
        # fig, axes = plt.subplots(
        # total_plots, 1,
        # figsize=(15, 4 + 2.5 * total_plots),
        # sharex=True,
        # gridspec_kw={'height_ratios': height_ratios},
        # constrained_layout=True   # handles legends and colorbars automatically
        # )

        # # EEG plot (top)
        # # if eeg_data is not None:
        # #     eeg_segment = eeg_data[:, start_sample:start_sample + n_samples]
        # #     offset = 50
        # #     for i, ch in enumerate(eeg_channels[:2]):  #only first EEG channel for clarity
        # #         axes[0].plot(time_axis, eeg_segment[i] + i * offset, label=ch)
        # #     axes[0].set_title("EEG")
        # #     axes[0].set_ylabel("Amplitude (μV)")
        # #     axes[0].legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=8, framealpha=0.6)
        # #     axes[0].grid(alpha=0.3)


        # #EEG heatmap 
        # eeg_segment = eeg_data[:, start_sample:start_sample + n_samples]
        # eeg_z = (eeg_segment - np.mean(eeg_segment, axis=1, keepdims=True)) / np.std(eeg_segment, axis=1, keepdims=True)
        # im = axes[0].imshow(eeg_z, aspect='auto', origin='lower',
        #                         extent=[0, duration, 0, len(eeg_channels)],
        #                         cmap='RdBu_r', vmin=-3, vmax=3)
        # axes[0].set_title("EEG (Z-scored Amplitude Heatmap)")
        # axes[0].set_ylabel("Channels")
        # axes[0].set_yticks(np.arange(len(eeg_channels)) + 0.5)
        # axes[0].set_yticklabels(eeg_channels, fontsize=7)
        # fig.colorbar(im, ax=axes[0], label="Z-score", shrink=0.8)



        # # EMG spectrograms
        # for idx, ch in enumerate(emg_channels):
        #     ax = axes[idx + 1]
        #     emg_segment = emg_data[idx, start_sample:start_sample + n_samples]

        #     # Pad EMG if shorter than expected
        #     if len(emg_segment) < n_samples:
        #         pad_len = n_samples - len(emg_segment)
        #         emg_segment = np.pad(emg_segment, (0, pad_len), mode='constant')

        #     f_centers, t_centers, Sxx = spectrogram(
        #         emg_segment, fs=sfreq, nperseg=1024, noverlap=512, mode='psd'
        #     )
        #     Sxx_db = 10 * np.log10(Sxx + 1e-12)

        #     # Generate evenly spaced time edges matching full window duration
        #     t_edges = np.linspace(0.0, duration, Sxx_db.shape[1] + 1)
        #     f_edges = np.linspace(f_centers[0], f_centers[-1], Sxx_db.shape[0] + 1)

        #     im = ax.pcolormesh(
        #         t_edges, f_edges, Sxx_db, shading='auto', cmap='viridis', rasterized=True
        #     )
        #     ax.set_title(f"{ch} - Filtered (10–500 Hz)")
        #     ax.set_ylabel("Freq [Hz]")
        #     ax.set_ylim(0, 250)
        #     ax.set_xlim(0, duration)

        #     # Add colorbar to the final EMG only
        #     if idx == n_emg - 1:
        #         fig.colorbar(im, ax=ax, label="Power (dB)", shrink=0.8)
        #         if raw_acc is None:
        #             ax.set_xlabel("Time [s]")
            
        # if raw_acc is not None:
        #     acc_ax = axes[-1]
        #     acc_segment = acc_data[:, start_sample:start_sample + n_samples]

        #     # Pad accelerometer if short
        #     for i in range(acc_segment.shape[0]):
        #         if acc_segment.shape[1] < n_samples:
        #             pad_len = n_samples - acc_segment.shape[1]
        #             acc_segment[i] = np.pad(acc_segment[i], (0, pad_len), mode='constant')


        #     #Instead of channel, need each sensor, (probably can get from the PD df)
        #     for i, ch in enumerate(acc_channels):
        #         acc_ax.plot(time_axis, acc_segment[i] + i * 0.5, label=ch)

        #     acc_ax.set_title("IMU (pitch roll, yaw)")
        #     acc_ax.set_ylabel("Angular velocity [rad / s]")
        #     acc_ax.set_xlabel("Time [s]")
        #     acc_ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=7, framealpha=0.5)
        #     acc_ax.grid(alpha=0.3)
            


        # # Event markers
        # if events is not None and len(events) > 0:
        #     for e in events:
        #         event_time_abs = e[0] / sfreq
        #         event_time_rel = event_time_abs - start
        #         if 0 <= event_time_rel <= duration:
        #             label = [k for k, v in event_id.items() if v == e[2]]
        #             axes[-1].axvline(event_time_rel, color='red', linestyle='--', alpha=0.8)
        #             if label:
        #                 axes[-1].text(event_time_rel, axes[-1].get_ylim()[1] * 0.9,
        #                             label[0], rotation=90, color='red', va='top', fontsize=7)

        # # layout adjustment
        # axes[0].set_xlim(0, duration)
        # # fig.subplots_adjust(left=0.07, right=0.92, top=0.95, bottom=0.07, hspace=0.35)
        # plt.show()


