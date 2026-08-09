from pathlib import Path
import matplotlib.pyplot as plt
import wfdb


# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "mit-bih-arrhythmia-database-1.0.0"

RECORD_NAME = "100"


def plot_ecg(record, duration_seconds=10):
    """Plot the MLII ECG signal for a selected duration."""

    sampling_frequency = record.fs

    # MLII is channel 0
    ecg_signal = record.p_signal[:, 0]

    # Calculate the number of samples to display
    number_of_samples = int(duration_seconds * sampling_frequency)

    # Limit to available samples
    number_of_samples = min(number_of_samples, len(ecg_signal))

    # Create time axis
    time = range(number_of_samples)

    time_seconds = [sample / sampling_frequency for sample in time]

    plt.figure(figsize=(14, 5))

    plt.plot(time_seconds, ecg_signal[:number_of_samples])

    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude (mV)")
    plt.title("MIT-BIH Record 100 - MLII ECG")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def load_ecg_record(record_name: str):
    """
    Load an ECG recording and its annotations from the local
    MIT-BIH Arrhythmia Database.
    """

    record_path = DATASET_DIR / record_name

    # Load ECG signal
    record = wfdb.rdrecord(str(record_path))

    # Load beat annotations
    annotation = wfdb.rdann(str(record_path), extension="atr")

    return record, annotation


def print_record_information(record, annotation):
    """Display basic information about the ECG recording."""

    print("\n========== ECG RECORD ==========")

    print(f"Record name       : {record.record_name}")
    print(f"Sampling frequency: {record.fs} Hz")
    print(f"Number of samples : {record.sig_len}")
    print(f"Number of channels: {record.n_sig}")

    print("\n========== CHANNELS ==========")

    for index, channel_name in enumerate(record.sig_name):
        print(f"Channel {index}: {channel_name}")

    print("\n========== SIGNAL ==========")

    print(f"Signal shape      : {record.p_signal.shape}")
    print(f"Units             : {record.units}")

    print("\n========== ANNOTATIONS ==========")

    print(f"Number of annotations: {len(annotation.sample)}")

    print("\nFirst 20 annotations:")

    for i in range(min(20, len(annotation.sample))):
        sample = annotation.sample[i]
        symbol = annotation.symbol[i]

        print(
            f"{i:2d}: "
            f"sample={sample:6d}, "
            f"time={sample / record.fs:8.3f} s, "
            f"symbol={symbol}"
        )


def main():
    print(f"Dataset directory: {DATASET_DIR}")
    print(f"Loading record: {RECORD_NAME}")

    record, annotation = load_ecg_record(RECORD_NAME)

    print_record_information(record, annotation)

    plot_ecg(record, duration_seconds=10)


if __name__ == "__main__":
    main()
