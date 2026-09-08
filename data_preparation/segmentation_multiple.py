from pathlib import Path

import wfdb
import numpy as np
import matplotlib.pyplot as plt

# Import preprocessing function
from preprocess import preprocess_ecg
from labeling import convert_label, is_heartbeat


# ============================================================
# PATHS AND SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "mit-bih-arrhythmia-database-1.0.0"

RECORD_NAME = "100"

# Number of samples before and after the annotated heartbeat
SAMPLES_BEFORE = 100
SAMPLES_AFTER = 100

# Number of heartbeats to inspect
NUMBER_OF_BEATS = 10


# ============================================================
# LOAD ECG RECORD AND ANNOTATIONS
# ============================================================


def load_record(record_name):
    """
    Load ECG signal and annotations from an MIT-BIH record.
    """

    record_path = DATASET_DIR / record_name

    # Load ECG signal
    record = wfdb.rdrecord(str(record_path))

    # Load annotations
    annotation = wfdb.rdann(str(record_path), "atr")

    # Channel 0 = MLII
    ecg = record.p_signal[:, 0]

    fs = record.fs

    return ecg, fs, annotation


# ============================================================
# EXTRACT ONE HEARTBEAT
# ============================================================


def extract_heartbeat(ecg, sample_index):
    """
    Extract a fixed-size window around an annotated heartbeat.
    """

    start = sample_index - SAMPLES_BEFORE
    end = sample_index + SAMPLES_AFTER

    # Check signal boundaries
    if start < 0 or end > len(ecg):
        return None

    heartbeat = ecg[start:end]

    return heartbeat


# ============================================================
# FIND VALID BEAT ANNOTATIONS
# ============================================================


def get_beat_annotations(annotation):
    """
    Find annotation indices that represent heartbeats.

    The decision is made by labeling.py so that
    heartbeat selection and class conversion use
    the same label definition.
    """

    beat_indices = []

    for i, symbol in enumerate(annotation.symbol):
        if is_heartbeat(symbol):
            beat_indices.append(i)

    return beat_indices


# ============================================================
# PLOT RAW VS PROCESSED ECG
# ============================================================


def plot_raw_vs_processed(raw_ecg, processed_ecg, fs):
    """
    Compare raw and preprocessed ECG for the first 10 seconds.
    """

    number_of_samples = int(10 * fs)

    raw = raw_ecg[:number_of_samples]
    processed = processed_ecg[:number_of_samples]

    time = np.arange(len(raw)) / fs

    plt.figure(figsize=(12, 7))

    plt.subplot(2, 1, 1)

    plt.plot(time, raw)

    plt.title("Raw ECG - MLII")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude (mV)")
    plt.grid(True)

    plt.subplot(2, 1, 2)

    plt.plot(time, processed)

    plt.title("Preprocessed ECG - MLII")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude (mV)")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# ============================================================
# PLOT INDIVIDUAL HEARTBEATS
# ============================================================


def plot_heartbeats(heartbeats, labels, fs):
    """
    Plot extracted preprocessed heartbeats.
    """

    number_of_beats = len(heartbeats)

    if number_of_beats == 0:
        print("No heartbeats to plot.")
        return

    time = (np.arange(len(heartbeats[0])) - SAMPLES_BEFORE) / fs

    fig, axes = plt.subplots(number_of_beats, 1, figsize=(10, 2 * number_of_beats))

    if number_of_beats == 1:
        axes = [axes]

    for i, heartbeat in enumerate(heartbeats):
        axes[i].plot(time, heartbeat)

        # Annotated heartbeat position
        axes[i].axvline(x=0, linestyle="--")

        axes[i].set_title(f"Beat {i + 1} | Class: {labels[i]}")

        axes[i].set_xlabel("Time (seconds)")
        axes[i].set_ylabel("Amplitude (mV)")

        axes[i].grid(True)

    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN
# ============================================================


def main():

    print("==========================================")
    print("MIT-BIH ECG PREPROCESSING + SEGMENTATION")
    print("==========================================")

    # --------------------------------------------------------
    # 1. Load ECG and annotations
    # --------------------------------------------------------

    raw_ecg, fs, annotation = load_record(RECORD_NAME)

    print(f"Record       : {RECORD_NAME}")
    print(f"Sampling rate: {fs} Hz")
    print(f"ECG samples  : {len(raw_ecg)}")
    print()

    # --------------------------------------------------------
    # 2. Preprocess ECG
    # --------------------------------------------------------

    print("Preprocessing ECG...")

    processed_ecg = preprocess_ecg(raw_ecg, fs)

    print("Preprocessing completed.")
    print()

    # --------------------------------------------------------
    # 3. Compare raw and processed ECG
    # --------------------------------------------------------

    plot_raw_vs_processed(raw_ecg, processed_ecg, fs)

    # --------------------------------------------------------
    # 4. Find heartbeat annotations
    # --------------------------------------------------------

    beat_indices = get_beat_annotations(annotation)

    print(f"Total beat annotations: {len(beat_indices)}")

    print()

    # --------------------------------------------------------
    # 5. Extract heartbeats from PREPROCESSED ECG
    # --------------------------------------------------------

    heartbeats = []
    labels = []

    for beat_number, annotation_index in enumerate(beat_indices[:NUMBER_OF_BEATS]):
        sample_index = annotation.sample[annotation_index]

        symbol = annotation.symbol[annotation_index]

        # Convert MIT-BIH symbol to 5-class label
        label = convert_label(symbol)

        heartbeat = extract_heartbeat(processed_ecg, sample_index)

        if heartbeat is None:
            print(f"Skipping sample {sample_index} (outside signal boundary)")
            continue

        heartbeats.append(heartbeat)

        # Store the converted 5-class label
        labels.append(label)

        time = sample_index / fs

        print(
            f"Beat {beat_number + 1:2d} | "
            f"Sample: {sample_index:6d} | "
            f"Time: {time:8.3f} s | "
            f"Symbol: {symbol} | "
            f"Class: {label} | "
            f"Shape: {heartbeat.shape}"
        )
    # --------------------------------------------------------
    # 6. Print final result
    # --------------------------------------------------------

    print()
    print(f"Successfully extracted: {len(heartbeats)} beats")

    # --------------------------------------------------------
    # 7. Plot extracted heartbeats
    # --------------------------------------------------------

    plot_heartbeats(heartbeats, labels, fs)


# ============================================================
# PROGRAM ENTRY
# ============================================================

if __name__ == "__main__":
    main()
