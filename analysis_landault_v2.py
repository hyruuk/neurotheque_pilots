import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mne.viz import plot_topomap

# ----------------------------- Configuration ----------------------------- #

# Path to the preprocessed raw EEG data file
DATA_PATH = "../data/pilot_data/sub-01_ses-001_raw_preprocessed_LandoitC_noEpoched.fif"

# Event trigger codes
NO_LOAD_TRIGGERS = [17, 34]
LOW_LOAD_TRIGGERS = [2, 4, 6, 8, 19, 21, 23, 25, 36, 38, 40, 42]
HIGH_LOAD_TRIGGERS = [10, 12, 14, 16, 27, 29, 31, 33, 44, 46, 48, 50]

# Response triggers
CORRECT_TRIGGER = 51
INCORRECT_TRIGGER = 52
NO_RESPONSE_TRIGGER = 54

# Mapping of conditions to new event IDs
EVENT_ID_MAPPING = {
    'no_load': 100,
    'low_load': 200,
    'high_load': 300
}

# Time windows for topographical maps (in seconds)
TOPO_WINDOWS = [(0.2, 0.3), (0.4, 0.8)]

# CDA analysis time window (in seconds)
CDA_WINDOW = (0.4, 0.8)

# Electrode groups for contralateral and ipsilateral hemispheres
LEFT_HEMI = ['P3', 'T5', 'O1']
RIGHT_HEMI = ['P4', 'T6', 'O2']

# ----------------------------- Data Loading ----------------------------- #

# Load the preprocessed raw EEG data
raw = mne.io.read_raw_fif(DATA_PATH, preload=True)

# Apply band-pass filter between 0.1 Hz and 30 Hz
raw.filter(0.1, 30.0)

# Detect events in the raw data
events = mne.find_events(raw)
print("Unique event codes:", np.unique(events[:, 2]))

# ----------------------------- Event Recode ----------------------------- #

# Recode memory array onset events to standardized event IDs
for event in events:
    trigger = event[2]
    if trigger in NO_LOAD_TRIGGERS:
        event[2] = EVENT_ID_MAPPING['no_load']
    elif trigger in LOW_LOAD_TRIGGERS:
        event[2] = EVENT_ID_MAPPING['low_load']
    elif trigger in HIGH_LOAD_TRIGGERS:
        event[2] = EVENT_ID_MAPPING['high_load']
    # Non-memory events remain unchanged

# ----------------------------- Epoching ----------------------------- #

# Define epoch parameters
TMIN, TMAX = -0.2, 1.0  # Epoch time window in seconds

# Create epochs for each condition
epochs = mne.Epochs(
    raw, events, event_id=EVENT_ID_MAPPING,
    tmin=TMIN, tmax=TMAX,
    baseline=(None, 0),
    preload=True
)

# ----------------------------- Trial Count Reporting ----------------------------- #

print("\nNumber of trials per condition after epoching:")
for condition in EVENT_ID_MAPPING.keys():
    n_trials = len(epochs[condition])
    print(f"{condition}: {n_trials} trials")

# ----------------------------- Averaging Evoked Responses ----------------------------- #

# Compute average evoked responses for each condition
evoked_no = epochs['no_load'].average()
evoked_low = epochs['low_load'].average()
evoked_high = epochs['high_load'].average()

# Extract time points from the evoked data
times = evoked_no.times

# ----------------------------- Topographical Maps ----------------------------- #

def plot_all_topoplots(evokeds, conditions, windows, fig_title="Topographical Maps"):
    """
    Generates a unified figure containing topographical maps for multiple conditions and time windows.

    Parameters:
    - evokeds: List of Evoked objects.
    - conditions: List of condition names corresponding to the Evoked objects.
    - windows: List of tuples specifying time windows (start_time, end_time) in seconds.
    - fig_title: Title for the entire figure.
    """
    n_conditions = len(evokeds)
    n_windows = len(windows)
    fig, axes = plt.subplots(n_conditions, n_windows, figsize=(5 * n_windows, 4 * n_conditions))
    fig.suptitle(fig_title, fontsize=16)

    # Ensure axes is a 2D array for consistent indexing
    if n_conditions == 1 and n_windows == 1:
        axes = np.array([[axes]])
    elif n_conditions == 1 or n_windows == 1:
        axes = axes.reshape(n_conditions, n_windows)

    for i, (evoked, condition) in enumerate(zip(evokeds, conditions)):
        for j, window in enumerate(windows):
            start, end = window
            evoked_window = evoked.copy().crop(tmin=start, tmax=end)
            mean_data = evoked_window.data.mean(axis=1)
            ax = axes[i, j]
            im, _ = plot_topomap(mean_data, evoked.info, axes=ax, show=False)
            ax.set_title(f"{condition} - {int(start*1000)}-{int(end*1000)} ms", fontsize=12)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# Define conditions and corresponding Evoked objects
evokeds = [evoked_no, evoked_low, evoked_high]
conditions = ['No-load', 'Low-load', 'High-load']

# Plot topographical maps
plot_all_topoplots(evokeds, conditions, TOPO_WINDOWS, fig_title="Topographical Maps for Different Conditions and Time Windows")

# ----------------------------- CDA Computation ----------------------------- #

def compute_cda(evoked, contra_channels, ipsi_channels):
    """
    Computes the Contralateral Delay Activity (CDA) by subtracting ipsilateral from contralateral channel data.

    Parameters:
    - evoked: Evoked object.
    - contra_channels: List of channel names in the contralateral hemisphere.
    - ipsi_channels: List of channel names in the ipsilateral hemisphere.

    Returns:
    - CDA difference wave as a NumPy array.
    """
    contra_data = evoked.copy().pick_channels(contra_channels).data.mean(axis=0)
    ipsi_data = evoked.copy().pick_channels(ipsi_channels).data.mean(axis=0)
    return contra_data - ipsi_data

# Calculate CDA for each condition
diff_no = compute_cda(evoked_no, RIGHT_HEMI, LEFT_HEMI)
diff_low = compute_cda(evoked_low, RIGHT_HEMI, LEFT_HEMI)
diff_high = compute_cda(evoked_high, RIGHT_HEMI, LEFT_HEMI)

# ----------------------------- CDA Amplitude Extraction ----------------------------- #

def extract_mean_amplitude(data, times, window):
    """
    Extracts the mean amplitude within a specified time window.

    Parameters:
    - data: NumPy array of signal data.
    - times: NumPy array of time points.
    - window: Tuple specifying the time window (start_time, end_time) in seconds.

    Returns:
    - Mean amplitude within the window.
    """
    start, end = window
    mask = (times >= start) & (times <= end)
    return data[mask].mean()

# Extract CDA amplitudes within the defined window
CDA_no = extract_mean_amplitude(diff_no, times, CDA_WINDOW)
CDA_low = extract_mean_amplitude(diff_low, times, CDA_WINDOW)
CDA_high = extract_mean_amplitude(diff_high, times, CDA_WINDOW)

print("\nCDA Amplitudes (µV):")
print(f"No-load: {CDA_no * 1e6:.2f} µV")
print(f"Low-load: {CDA_low * 1e6:.2f} µV")
print(f"High-load: {CDA_high * 1e6:.2f} µV")

# ----------------------------- CDA Difference Waves Plot ----------------------------- #

plt.figure(figsize=(10, 6))
plt.gca().invert_yaxis()
plt.plot(times * 1000, diff_no * 1e6, label='No-load')
plt.plot(times * 1000, diff_low * 1e6, label='Low-load')
plt.plot(times * 1000, diff_high * 1e6, label='High-load')
plt.axvspan(CDA_WINDOW[0] * 1000, CDA_WINDOW[1] * 1000, color='pink', alpha=0.3, label='CDA Window')
plt.axhline(0, color='k', linestyle='--')
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude (µV)")
plt.title("CDA Difference Waves")
plt.legend()
plt.tight_layout()
plt.show()

# ----------------------------- Performance Analysis ----------------------------- #

# Mapping of event codes to condition names
CONDITIONS = {100: 'no_load', 200: 'low_load', 300: 'high_load'}

data_records = []
event_codes = events[:, 2]
event_times = events[:, 0] / raw.info['sfreq']

for idx, trigger in enumerate(event_codes):
    if trigger in CONDITIONS:
        condition = CONDITIONS[trigger]
        rt = np.nan
        accuracy = np.nan
        # Search for the next response event
        for resp_trigger in event_codes[idx + 1:]:
            if resp_trigger in [CORRECT_TRIGGER, INCORRECT_TRIGGER, NO_RESPONSE_TRIGGER]:
                if resp_trigger == CORRECT_TRIGGER:
                    accuracy = 1
                else:
                    accuracy = 0
                # Calculate reaction time
                resp_time = event_times[idx + 1 + np.where(event_codes[idx + 1:] == resp_trigger)[0][0]]
                onset_time = event_times[idx]
                rt = resp_time - onset_time
                break
        data_records.append([condition, rt, accuracy])

# Create DataFrame from recorded data
perf_df = pd.DataFrame(data_records, columns=['Condition', 'RT', 'Accuracy'])

# Exclude trials without a valid response
perf_df.dropna(subset=['RT'], inplace=True)

# ----------------------------- Trial Counts ----------------------------- #

clean_trials = perf_df['Condition'].value_counts().to_dict()
print("\nNumber of clean trials per condition:")
for condition, count in clean_trials.items():
    print(f"{condition}: {count} trials")

# ----------------------------- Performance Summary ----------------------------- #

perf_summary = perf_df.groupby('Condition').agg(
    mean_RT=('RT', 'mean'),
    mean_Acc=('Accuracy', 'mean'),
    trial_count=('RT', 'count')
).reset_index()

print("\nPerformance Summary:")
print(perf_summary)

# ----------------------------- Performance Visualization ----------------------------- #

# Define the order of conditions for plotting
conditions_order = ['low_load', 'high_load']

# Initialize lists to store performance metrics
mean_RT = []
mean_Acc = []

# Populate performance metrics, handling missing conditions
for condition in conditions_order:
    if condition in perf_summary['Condition'].values:
        mean_RT.append(perf_summary.loc[perf_summary.Condition == condition, 'mean_RT'].values[0])
        mean_Acc.append(perf_summary.loc[perf_summary.Condition == condition, 'mean_Acc'].values[0])
    else:
        mean_RT.append(np.nan)
        mean_Acc.append(np.nan)

# Create bar plots for Reaction Times and Accuracy
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Reaction Time Plot
axes[0].bar(conditions_order, mean_RT, color=['skyblue', 'salmon'])
axes[0].set_ylabel("Mean RT (s)")
axes[0].set_title("Reaction Times by Condition")
axes[0].set_xlabel("Condition")
axes[0].set_xticklabels([c.replace('_', '-').capitalize() for c in conditions_order], rotation=45)

# Accuracy Plot
axes[1].bar(conditions_order, mean_Acc, color=['skyblue', 'salmon'])
axes[1].set_ylabel("Mean Accuracy")
axes[1].set_ylim([0, 1])
axes[1].set_title("Accuracy by Condition")
axes[1].set_xlabel("Condition")
axes[1].set_xticklabels([c.replace('_', '-').capitalize() for c in conditions_order], rotation=45)

plt.tight_layout()
plt.show()
