import mne
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from mne.preprocessing import ICA
from mne.io import read_raw_brainvision
from scipy.signal import spectrogram

# VHDR
#.eeg and .vmrk files are being linked correctly (implcity via these outputs)
root_dir = '/Users/nathangu/Desktop/Pytorch/NT/t8j8v4hnm4-1/Raw'

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



#Figure out VMRK files still and event markers (likely need to get outside VMRK files like the paper)
def plot_combined_timeseries(raw_eeg, raw_emg_filtered, events, event_id, duration, start):
    eeg_data, emg_data = None, None
    eeg_channels, emg_channels = [], []
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

    # Window
    n_samples = int(duration * sfreq)
    start_sample = int(start * sfreq)
    time_axis = np.arange(n_samples) / sfreq  # starts at 0

    n_emg = len(emg_channels)
    total_plots = 1 + n_emg

    height_ratios = [1] + [1.25] * n_emg
    fig, axes = plt.subplots(total_plots, 1, figsize=(15, 4 + 3 * n_emg), sharex=True,
                             gridspec_kw={'height_ratios': height_ratios})

    # EEG plot (top)
    if eeg_data is not None:
        eeg_segment = eeg_data[:, start_sample:start_sample + n_samples]
        offset = 50
        for i, ch in enumerate(eeg_channels[:1]):  #only first EEG channel for clarity
            axes[0].plot(time_axis, eeg_segment[i] + i * offset, label=ch)
        axes[0].set_title("EEG")
        axes[0].set_ylabel("Amplitude (μV)")
        axes[0].legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=8, framealpha=0.6)
        axes[0].grid(alpha=0.3)

    # EMG spectrograms
    for idx, ch in enumerate(emg_channels):
        ax = axes[idx + 1]
        emg_segment = emg_data[idx, start_sample:start_sample + n_samples]
        f_centers, t_centers, Sxx = spectrogram(emg_segment, fs=sfreq, nperseg=1024, noverlap=512, mode='psd')
        Sxx_db = 10 * np.log10(Sxx + 1e-12)

        # alignment
        f_edges = np.concatenate([
            [f_centers[0] - (f_centers[1] - f_centers[0]) / 2],
            (f_centers[:-1] + f_centers[1:]) / 2,
            [f_centers[-1] + (f_centers[-1] - f_centers[-2]) / 2]
        ])
        t_edges = np.linspace(0.0, duration, Sxx_db.shape[1] + 1)

        # Plot spectrogram
        try:
            im = ax.pcolormesh(t_edges, f_edges, Sxx_db, shading='gouraud')
        except Exception:
            t_centers = np.linspace(0.0, duration, Sxx_db.shape[1])
            im = ax.pcolormesh(t_centers, f_centers, Sxx_db, shading='auto')

        ax.set_title(f"{ch} - Filtered (10–500 Hz)")
        ax.set_ylabel("Frequency [Hz]")
        ax.set_ylim(0, 250)
        ax.set_xlim(0, duration)
        if idx == n_emg - 1:
            ax.set_xlabel("Time [s]")

        #colorbar for each spectrogram
        cax = fig.add_axes([0.88, 0.12 + (0.36 * (n_emg - idx - 1) / n_emg), 0.02, 0.36 / n_emg])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label("Power (dB)")

    # Event markers
    if events is not None and len(events) > 0:
        for e in events:
            event_time_abs = e[0] / sfreq
            event_time_rel = event_time_abs - start
            if 0 <= event_time_rel <= duration:
                label = [k for k, v in event_id.items() if v == e[2]]
                for ax in axes:
                    ax.axvline(event_time_rel, color='red', linestyle='--', alpha=0.8)
                if label:
                    axes[0].text(event_time_rel, axes[0].get_ylim()[1] * 0.92,
                                 label[0], rotation=90, color='red', va='top', fontsize=8)

    fig.subplots_adjust(left=0.06, right=0.86, top=0.95, bottom=0.07, hspace=0.32)
    plt.show()

    # EMG plot
    # if emg_data is not None:
    #     emg_segment = emg_data[:, start : start + n_samples]
    #     for i, ch in enumerate(emg_channels):
    #         axes[1].plot(time_axis, emg_segment[i] + i * 200, label=ch) 
    #     axes[1].set_title("EMG (10–500 Hz)")
    #     axes[1].set_ylabel("Amplitude (μV)")
    #     axes[1].legend(loc='upper right', fontsize=8)
    #     axes[1].grid(alpha=0.3)


#Running plots and preproc data
vhdr_files = get_vhdr_files(root_dir)
for file in vhdr_files:
    raw_eeg, raw_emg, ica, raw_emg_filtered = prep(file)
    # Load event info
    raw = mne.io.read_raw_brainvision(file, preload=True)
    events, event_id = mne.events_from_annotations(raw)
    # print(events)
    # print(event_id)
    

    #Tasks 1, 2: 2.5 - 3 mins long
    #Tasks 3, 4: 30 - 60 sec
    plot_combined_timeseries(raw_eeg, raw_emg_filtered, events, event_id, duration=60, start=0)

    # raw = mne.io.read_raw_brainvision(file, preload=True)
    # print(raw.ch_names)
    #event markers
    # events, event_id = mne.events_from_annotations(raw)
    # print(event_id)
    # print(events[:10])
    # print(events)

    #EMG portion
    # if raw_emg is not None:
    #     data_original = raw_emg.get_data()
    #     data_filtered = raw_emg_filtered.get_data()
    #     sfreq = raw_emg.info['sfreq']
    #     channel_names = raw_emg.ch_names
    #     check_quality(data_original, sfreq, channel_names)
    #     plot_emg(data_original, data_filtered, channel_names, sfreq)
    #     plot_spectrograms(data_original, data_filtered, sfreq, channel_names)
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

    # epochs = mne.Epochs(raw_emg, events, event_id, tmin=-0.2, tmax=1.0, preload=True)

    # data_original = raw_emg.get_data()
    # data_filtered = raw_emg_filtered.get_data()
    # sfreq = raw_emg_filtered.info['sfreq']
    # channel_names = raw_emg.ch_names

    # print(f"\nFinal sampling rate: {sfreq} Hz")
    # print(f"Data shape: {data_filtered.shape}")

    # check_quality(data_original, sfreq, channel_names)
    # plot_emg(data_original, data_filtered, channel_names, sfreq)
    # plot_spectrograms(data_original, data_filtered, sfreq, channel_names)


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










# Input: CAS9 sequence
# Have a DP to track the current proteins and what sequences it corresponds to (String to String map)
# Run a sliding window algorithm on the sequence (window size tbd)
# Get the current substring of amino acids
# edit distance from how far it is from the thing (DP) Smith–Waterman algorithm
#  --> Find a protein (in humans?) containing that specific sequence (not sure what to use, maybe BLAST)
#  --> if != null, put it into the DP
#  --> else (not found) skip an amino acid and run the checks again
# (Try to account for structure if possible, not sure what check I can do but 
# maybe parse 2 at once (structure information and the sequnce along with it))
# process the DP (not sure what to process yet, but maybe protein stability and other metrics of the proteins)
# probably somehow score the DP overalls
# return the best DP based off of highest score, 
# which would have the proteins corresponding to their respective sequences

#Scoring through alphafold / tmalign