import os
import numpy as np
import mne
from mne.preprocessing import ICA
from mne_icalabel import label_components
import matplotlib.pyplot as plt
from mne_bids import BIDSPath, read_raw_bids

event_id = {
    'normal': 1,   
    'conflict': 2  
}

def process_subject(subject_id, session, bids_root):
    
    # Define file paths
    bids_path = BIDSPath(
        subject=subject_id, session=session, task="PredictionError", suffix="eeg", extension=".vhdr", root=bids_root
    )
    
    # Load raw data
    raw = read_raw_bids(bids_path)

    # Preprocessing steps
    raw.annotations.onset -= 0.063  # Adjust for EEG setup delay
    raw_resampled = raw.copy().resample(sfreq=250, npad="auto")  # Resample
    raw_filtered = raw_resampled.filter(l_freq=1.0, h_freq=124.0).notch_filter(freqs=50)  # Bandpass + Notch filter
    raw_referenced = raw_filtered.set_eeg_reference(ref_channels="average").set_montage("standard_1020")  # Re-reference
    
    # Step 5: Extract events based on annotations
    events = []
    for annot in raw_referenced.annotations:
        print(f"Processing annotation: {annot['description']}")
        if 'normal_or_conflict:normal' in annot['description']:
            events.append([int(annot['onset'] * raw_referenced.info['sfreq']), 0, event_id['normal']])
        elif 'normal_or_conflict:conflict' in annot['description']:
            events.append([int(annot['onset'] * raw_referenced.info['sfreq']), 0, event_id['conflict']])
        else:
            print("Skipping irrelevant annotation:", annot['description'])
    events = np.array(events, dtype=int)

    # Extract epochs
    epochs = mne.Epochs(
        raw_referenced, events, event_id=event_id, tmin=-0.3, tmax=0.7,
        baseline=(-0.3, 0), preload=True, event_repeated='merge'
    )
    print(f"Total epochs: {len(epochs)}")

    # Compute Mean Absolute Amplitude for Each Epoch
    epoch_data = epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)
    mean_amplitudes = np.mean(np.abs(epoch_data), axis=(1, 2))  # Mean over channels and time

    # Rank Epochs by Mean Amplitude
    ranked_indices = np.argsort(mean_amplitudes)  

    # Keep 85% of the Cleanest Epochs
    percentage = 85
    n_epochs_to_keep = int(len(epochs) * (percentage / 100))
    selected_indices = ranked_indices[:n_epochs_to_keep]  # Select top 85% clean epochs

    # Create Clean Epochs and Apply ICA
    clean_epochs = epochs[selected_indices]

    ica = ICA(n_components=10, method='fastica', random_state=42, max_iter=5000)
    ica.fit(clean_epochs)

    ica.plot_components()

    # Step 6: Use ICLabel for automatic component classification
    labels = label_components(raw_referenced, ica, method='iclabel')

    print("ICLabel Results:")
    for idx, (label, prob) in enumerate(zip(labels['labels'], labels['y_pred_proba'])):
        print(f"Component {idx}: {label} (Probability: {prob:.2f})")

    # Automatically mark bad components for exclusion
    bad_ics = [idx for idx, label in enumerate(labels['labels'])
            if label in ('eye blink', 'muscle artifact', 'line_noise')]

    ica.exclude = bad_ics  # Mark components for exclusion

    ica.apply(raw_referenced)

    raw_referenced.set_meas_date(None)

    # Step 7: Filter ERP data with 0.2 Hz high-pass and 35 Hz low-pass
    raw_referenced = raw_referenced.copy().filter(l_freq=0.2, h_freq=35.0)

    # Step 9: Reject 10% of the noisiest epochs based on signal amplitude
    epoch_data = epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)
    mean_amplitudes = np.mean(np.abs(epoch_data), axis=(1, 2))  # Compute mean amplitude for each epoch
    threshold = np.percentile(mean_amplitudes, 90)  # Top 10% noisy epochs
    clean_epochs = epochs[mean_amplitudes < threshold]

    # Step 10: Focus analyses on selected electrodes 
    frontal_channels = ['Fz', 'Cz', 'Fp1', 'FC1', 'FC2']
    clean_epochs.pick_channels(frontal_channels)

    # Step 11: Extract ERP negativity peaks (minimum peak) in the 100–300 ms time window
    time_window = (0.1, 0.3)  # 100–300 ms
    negativity_peaks = {}

    for ch_name in frontal_channels:
        channel_idx = clean_epochs.ch_names.index(ch_name)
        erp_data = clean_epochs.average().data[channel_idx]
        times = clean_epochs.times

        # Extract the data within the time window
        mask = (times >= time_window[0]) & (times <= time_window[1])
        time_window_data = erp_data[mask]
        time_window_times = times[mask]

        # Find the minimum (negative) peak
        peak_idx = np.argmin(time_window_data)
        peak_time = time_window_times[peak_idx]
        peak_amplitude = time_window_data[peak_idx]

        negativity_peaks[ch_name] = (peak_time, peak_amplitude)
        print(f"Channel {ch_name}: Negativity peak at {peak_time:.3f} s with amplitude {peak_amplitude:.3f} µV")


    epochs_match = epochs['normal']
    epochs_mismatch = epochs['conflict']
    print(f"Match trials: {len(epochs_match)}, Mismatch trials: {len(epochs_mismatch)}")


    erp_normal = epochs_match.average()
    erp_conflict = epochs_mismatch.average()

    return erp_normal.data, erp_conflict.data, raw_referenced

bids_root = "/home/st/st_us-053000/st_st190561/EEG" # path to the bids_root folder
subjects = ["02", "03", "06", "07", "08", "11", "12", "13", "14", "15", "16"] # Hard-coded valid subjects for experiment
sessions = ["EMS", "Vibro", "Visual"]

all_normal_ems = []
all_conflict_ems = []
all_normal_vibro = []
all_conflict_vibro = []
all_normal_visual = []
all_conflict_visual = []

for subject in subjects:
    for session in sessions:
        try:
            normal_data, conflict_data, raw_referenced = process_subject(subject, session, bids_root)
            print(f"Shape of normal_data for subject {subject}, session {session}: {normal_data.shape}")
            print(f"Shape of conflict_data for subject {subject}, session {session}: {conflict_data.shape}")
            print("The channels are:",raw_referenced.ch_names)

            if session == "EMS":
                all_normal_ems.append(normal_data)
                all_conflict_ems.append(conflict_data)
            elif session == "Vibro":
                all_normal_vibro.append(normal_data)
                all_conflict_vibro.append(conflict_data)
            elif session == "Visual":
                all_normal_visual.append(normal_data)
                all_conflict_visual.append(conflict_data)

        except FileNotFoundError:
            print(f"File not found for subject {subject}, session {session}. Skipping this session.")
            continue

avg_normal_ems = np.mean(all_normal_ems, axis=0)
avg_conflict_ems = np.mean(all_conflict_ems, axis=0)
avg_normal_vibro = np.mean(all_normal_vibro, axis=0)
avg_conflict_vibro = np.mean(all_conflict_vibro, axis=0)
avg_normal_visual = np.mean(all_normal_visual, axis=0)
avg_conflict_visual = np.mean(all_conflict_visual, axis=0)

channel = 'Fz'

raw_example = read_raw_bids(BIDSPath(
    subject=subjects[0], session=sessions[0], task="PredictionError", suffix="eeg", extension=".vhdr", root=bids_root
))
channel_idx = raw_referenced.info['ch_names'].index(channel)

erp_normal_ems = avg_normal_ems[channel_idx, :]  # Shape: (n_times,)
erp_conflict_ems = avg_conflict_ems[channel_idx, :]  # Shape: (n_times,)
erp_normal_vibro = avg_normal_vibro[channel_idx, :]  # Shape: (n_times,)
erp_conflict_vibro = avg_conflict_vibro[channel_idx, :]  # Shape: (n_times,)
erp_normal_visual = avg_normal_visual[channel_idx, :]  # Shape: (n_times,)
erp_conflict_visual = avg_conflict_visual[channel_idx, :]  # Shape: (n_times,)

erp_difference_ems = erp_conflict_ems - erp_normal_ems
erp_difference_vibro = erp_conflict_vibro - erp_normal_vibro
erp_difference_visual = erp_conflict_visual - erp_normal_visual

times = np.linspace(-0.3, 0.7, erp_normal_ems.shape[0])  # Adjust based on your sampling rate


fig, axs = plt.subplots(3, 1, figsize=(12, 36), sharex=True)

axs[0].plot(times, erp_normal_ems * 1e6, label=f'EMS', color='red')
axs[0].plot(times, erp_normal_vibro * 1e6, label=f'Vibro', color='yellow')
axs[0].plot(times, erp_normal_visual * 1e6, label=f'Visual', color='blue')
axs[0].axvspan(0.1, 0.3, color='gray', alpha=0.2, label='Peak Search Window')
axs[0].axhline(0, color='black', linestyle='--')
axs[0].set_title("ERP Match Across Sessions")
axs[0].set_ylabel("Amplitude (µV)")
axs[0].legend()

axs[1].plot(times, erp_conflict_ems * 1e6, label=f'EMS', color='red')
axs[1].plot(times, erp_conflict_vibro * 1e6, label=f'Vibro', color='yellow')
axs[1].plot(times, erp_conflict_visual * 1e6, label=f'Visual', color='blue')
axs[1].axvspan(0.1, 0.3, color='gray', alpha=0.2, label='Peak Search Window')
axs[1].axhline(0, color='black', linestyle='--')
axs[1].set_title("ERP Mismatch Across Sessions")
axs[1].set_ylabel("Amplitude (µV)")
axs[1].legend()

axs[2].plot(times, erp_difference_ems * 1e6, label=f'EMS', color='red')
axs[2].plot(times, erp_difference_vibro * 1e6, label=f'Vibro', color='yellow')
axs[2].plot(times, erp_difference_visual * 1e6, label=f'Visual', color='blue')
axs[2].axvspan(0.1, 0.3, color='gray', alpha=0.2, label='Peak Search Window')
axs[2].axhline(0, color='black', linestyle='--')
axs[2].set_title("ERP Difference (Mismatch - Match) Across Sessions")
axs[2].set_xlabel("Time (s)")
axs[2].set_ylabel("Amplitude (µV)")
axs[2].legend()

plt.tight_layout()
plt.savefig("/home/st/st_us-053000/st_st190561/EEG/fz.png")
plt.show()
