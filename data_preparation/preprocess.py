from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wfdb
from scipy import signal


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "mit-bih-arrhythmia-database-1.0.0"

RECORD_NAME = "100"

LOWPASS_CUTOFF = 40.0
NOTCH_FREQUENCY = 50.0
NOTCH_Q = 30.0

PLOT_DURATION = 10.0


def load_ecg(record_name):

    record_path = DATASET_DIR / record_name

    record = wfdb.rdrecord(str(record_path))

    ecg = record.p_signal[:, 0]  # every row but only the column 0

    return record, ecg


def remove_baseline_wander(ecg, fs):

    cutoff = 0.5

    b, a = signal.butter(N=2, Wn=cutoff, btype="highpass", fs=fs)

    filtered_ecg = signal.filtfilt(b, a, ecg)

    return filtered_ecg


def apply_notch_filter(ecg, fs):

    b, a = signal.iirnotch(w0=NOTCH_FREQUENCY, Q=NOTCH_Q, fs=fs)

    filtered_ecg = signal.filtfilt(b, a, ecg)

    return filtered_ecg


def apply_lowpass_filter(ecg, fs):

    b, a = signal.butter(N=4, Wn=LOWPASS_CUTOFF, btype="lowpass", fs=fs)

    filtered_ecg = signal.filtfilt(b, a, ecg)

    return filtered_ecg


def preprocess_ecg(ecg, fs):

    ecg = remove_baseline_wander(ecg, fs)

    ecg = apply_notch_filter(ecg, fs)

    ecg = apply_lowpass_filter(ecg, fs)

    return ecg


def plot_comparison(raw_ecg, processed_ecg, fs):

    number_of_samples = int(PLOT_DURATION * fs)

    time = np.arange(number_of_samples) / fs

    plt.figure(figsize=(14, 7))

    plt.subplot(2, 1, 1)

    plt.plot(time, raw_ecg[:number_of_samples])

    plt.title("Raw ECG - MLII")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude (mV)")
    plt.grid(True)

    plt.subplot(2, 1, 2)

    plt.plot(time, processed_ecg[:number_of_samples])

    plt.title("Preprocessed ECG - MLII")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude (mV)")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def main():

    print(f"Loading record: {RECORD_NAME}")

    record, raw_ecg = load_ecg(RECORD_NAME)

    fs = record.fs

    print("\n========== ECG INFORMATION ==========")
    print(f"Sampling frequency : {fs} Hz")
    print(f"Channel            : MLII")
    print(f"Number of samples  : {len(raw_ecg)}")
    print(f"Duration           : {len(raw_ecg) / fs:.2f} seconds")

    print("\nApplying preprocessing...")

    processed_ecg = preprocess_ecg(raw_ecg, fs)

    print("Preprocessing complete.")

    plot_comparison(raw_ecg, processed_ecg, fs)


if __name__ == "__main__":
    main()
