from pathlib import Path
import wfdb
import matplotlib.pyplot as plt


project_root = Path(__file__).resolve().parent.parent
dataset_dir = project_root / "mit-bih-arrhythmia-database-1.0.0"
record_name = "101"


def plot(record, duration):
    sampling_frequency = record.fs

    ecg_signal = record.p_signal[:, 0]

    number_of_samples = int(duration * sampling_frequency)

    number_of_samples = min(number_of_samples, len(ecg_signal))

    time = range(number_of_samples)

    time_sec = [sample / sampling_frequency for sample in time]

    plt.figure(figsize=(14, 7))
    plt.plot(time_sec, ecg_signal[:number_of_samples])

    plt.xlabel("Time (sec)")
    plt.ylabel("Amplitude (mV)")
    plt.title("ECG signal")
    plt.grid(True)

    plt.show()


def load_record(record_name: str):

    record_path = dataset_dir / record_name

    record = wfdb.rdrecord(str(record_path))

    annotation = wfdb.rdann(str(record_path), extension="atr")

    return record, annotation


def print_record(record, annotation):
    print("\n")
    print(f"Record name        : {record.record_name}")
    print(f"Sampling Frquency  : {record.fs}Hz")
    print(f"Number of channels : {record.n_sig}")
    print("\n")

    for i, channel in enumerate(record.sig_name):
        print(f"channel {i} : {channel}")

    print("\n")
    print(f"Signal shape : {record.p_signal.shape}")
    print(f"Units        : {record.units[0]},{record.units[1]}")

    print("\n")

    print(f"Number of annotation : {len(annotation.sample)}")
    print("\nfirst 20 annotation:")
    print(" ")

    for i in range(min(30, len(annotation.sample))):
        sample = annotation.sample[i]
        symbol = annotation.symbol[i]

        print(
            f"| {i:2d}: | sample = {sample:6d} | time = {sample / record.fs:8.3f}s | symbol = {symbol} |"
        )


def main():
    global record_name
    print(f"LOADING Record = {record_name}")
    record, annotation = load_record(record_name)
    print_record(record, annotation)
    plot(record, duration=10)


if __name__ == "__main__":
    main()
