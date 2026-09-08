from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wfdb


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "mit-bih-arrhythmia-database-1.0.0"

RECORD_NAME = "100"

# Number of samples before and after the annotated beat
SAMPLES_BEFORE = 100
SAMPLES_AFTER = 100


# --------------------------------------------------
# Load ECG and annotations
# --------------------------------------------------


def load_record(record_name):
    """Load ECG signal and annotations."""

    record_path = DATASET_DIR / record_name

    record = wfdb.rdrecord(str(record_path))
    annotation = wfdb.rdann(str(record_path), "atr")

    # MLII = channel 0
    ecg = record.p_signal[:, 0]

    return record, ecg, annotation


# --------------------------------------------------
# Extract one heartbeat
# --------------------------------------------------


def extract_heartbeat(ecg, sample_index):
    """Extract a fixed-size ECG segment around an annotated sample."""

    start = sample_index - SAMPLES_BEFORE
    end = sample_index + SAMPLES_AFTER

    # Make sure the window stays inside the ECG signal
    if start < 0 or end > len(ecg):
        return None

    heartbeat = ecg[start:end]

    return heartbeat


# --------------------------------------------------
# Plot heartbeat
# --------------------------------------------------


def plot_heartbeat(heartbeat, sample_index, fs):
    """Plot one extracted heartbeat."""

    number_of_samples = len(heartbeat)

    # Time relative to the annotated point
    time = (np.arange(number_of_samples) - SAMPLES_BEFORE) / fs

    plt.figure(figsize=(10, 5))

    plt.plot(time, heartbeat)

    plt.axvline(x=0, linestyle="--", label="Annotated point")

    plt.xlabel("Time relative to annotation (seconds)")
    plt.ylabel("Amplitude (mV)")
    plt.title(f"Extracted Heartbeat - Record {RECORD_NAME}, Sample {sample_index}")

    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# Main
# --------------------------------------------------


def main():

    print(f"Loading record: {RECORD_NAME}")

    record, ecg, annotation = load_record(RECORD_NAME)

    fs = record.fs

    print("\n========== RECORD ==========")
    print(f"Sampling frequency : {fs} Hz")
    print(f"ECG samples       : {len(ecg)}")
    print(f"Number annotations: {len(annotation.sample)}")

    # Select the first beat annotation
    sample_index = annotation.sample[2]
    symbol = annotation.symbol[2]

    print("\n========== SELECTED BEAT ==========")
    print(f"Sample             : {sample_index}")
    print(f"Time               : {sample_index / fs:.3f} s")
    print(f"Annotation         : {symbol}")

    # Extract heartbeat
    heartbeat = extract_heartbeat(ecg, sample_index)

    if heartbeat is None:
        print("Could not extract heartbeat.")
        return

    print("\n========== HEARTBEAT ==========")
    print(f"Number of samples  : {len(heartbeat)}")
    print(f"Duration           : {len(heartbeat) / fs:.3f} seconds")

    # Plot
    plot_heartbeat(heartbeat, sample_index, fs)


if __name__ == "__main__":
    main()
