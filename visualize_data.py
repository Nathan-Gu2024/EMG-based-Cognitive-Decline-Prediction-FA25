import mne
import requests
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from mne.preprocessing import ICA
from mne.io import read_raw_brainvision
from scipy.signal import spectrogram
from mne import create_info, EpochsArray


# Run this once, should allow you to download the data files

# url = 'https://drive.google.com/uc?export=download&id=1yc1evq9s3N7tfYX_vJbchFgKuwbFJOhJ?'
# response = requests.get(url)
# with open('local_filename.ext', 'wb') as file:
#     file.write(response.content)

#Change your directory accordingly otherwise this won't run
root_dir = '/Users/nathangu/Desktop/Pytorch/NT/t8j8v4hnm4-1/Raw'

####ACTUAL FUNCTIONS
# EMG: bandpass 10–500 Hz
# EEG: bandpass 0.5–45 Hz
# (They resample to 500 Hz or 1000 Hz depending on the modality)

def get_vhdr_files(root_dir):
    vhdr_files = glob.glob(os.path.join(root_dir, '**', '*.vhdr'), recursive=True)
    print(f"Found {len(vhdr_files)} .vhdr files")
    return vhdr_files


# Parse the accelerometer data
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



#Running plots and preproc data
vhdr_files = get_vhdr_files(root_dir)
for file in vhdr_files:
    raw_eeg, raw_emg, ica, raw_emg_filtered = prep(file)
    # Load event info
    raw = mne.io.read_raw_brainvision(file, preload=True)
    # print(raw.info['ch_names'])
    events, event_id = mne.events_from_annotations(raw)
#     # print(events)
#     # print(event_id)
    

#     #Tasks 1, 2: 2.5 - 3 mins long
#     #Tasks 3, 4: 30 - 60 sec
    plot_combined_timeseries(raw_eeg, raw_emg_filtered, None, events, event_id, duration=30, start=0)

    # raw = mne.io.read_raw_brainvision(file, preload=True)
    # print(raw.ch_names)

    #EMG portion
    # if raw_emg is not None:
    #     data_original = raw_emg.get_data()
    #     data_filtered = raw_emg_filtered.get_data()
    #     sfreq = raw_emg.info['sfreq']
    #     channel_names = raw_emg.ch_names
    #     check_quality(data_original, sfreq, channel_names)
    #     plot_emg(data_original, data_filtered, channel_names, sfreq)
        # plot_spectrograms(data_original, data_filtered, sfreq, channel_names)
    #     events, event_id = mne.events_from_annotations(raw_emg)
    #     print(event_id)
    #     print(events[:10])
    #     print(events)
    
    #EEG portion
    # if raw_eeg is not None:
    #     data_eeg = raw_eeg.get_data()
    #     sfreq_eeg = raw_eeg.info['sfreq']
    #     channel_names_eeg = raw_eeg.ch_names
    #     check_quality(data_eeg, sfreq_eeg, channel_names_eeg)
    #     #Not a refined plot, super basic
    #     raw_eeg.plot_psd(fmax=60)
    #     raw_eeg.plot(duration=5, n_channels=10)
    #     plt.tight_layout()
    #     plt.show()
    #     raw_eeg.compute_psd(fmax=50).plot(picks="data", exclude="bads", amplitude=False)
    #     raw_eeg.plot(duration=5, n_channels=30)
    #     plt.show()

    # epochs = mne.Epochs(raw_eeg, events, event_id, tmin=-0.2, tmax=1.0, preload=True)
    # times = np.arange(0, 1, 0.1)
    # epochs.average().plot_topomap(times, ch_type='eeg')

    # data_original = raw_emg.get_data()
    # data_filtered = raw_emg_filtered.get_data()
    # sfreq = raw_emg_filtered.info['sfreq']
    # channel_names = raw_emg.ch_names

    # print(f"\nFinal sampling rate: {sfreq} Hz")
    # print(f"Data shape: {data_filtered.shape}")

#debug
# raw = mne.io.read_raw_brainvision(file, preload=True)

#event markers
# events, event_id = mne.events_from_annotations(raw)
# print(event_id)
# print(events[:10])
# print(events)

#epoching
# epochs = mne.Epochs(raw, events, event_id, tmin=-0.2, tmax=1.0, preload=True)


# for ch in raw.ch_names:
#     print(ch)
# raw.plot(duration=5, n_channels=10)

# print(raw)
# print(raw.get_data().shape)
# print(raw.get_data()[:, :10])
# print(raw._filenames)
# print(np.ptp(raw.get_data()))




