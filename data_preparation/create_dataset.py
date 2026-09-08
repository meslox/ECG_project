from pathlib import Path
import json

import wfdb
import numpy as np

from preprocess import preprocess_ecg
from labeling import convert_label, is_heartbeat


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "mit-bih-arrhythmia-database-1.0.0"

SPLIT_FILE = PROJECT_ROOT / "data_preparation" / "optimized_split.json"

OUTPUT_DIR = PROJECT_ROOT / "dataset"


# ============================================================
# ECG SEGMENT SETTINGS
# ============================================================

# 100 samples before R-peak
SAMPLES_BEFORE = 100

# 100 samples after R-peak
SAMPLES_AFTER = 100

# Total samples in each heartbeat
SEGMENT_LENGTH = SAMPLES_BEFORE + SAMPLES_AFTER


# ============================================================
# LOAD SPLIT
# ============================================================


def load_split():

    if not SPLIT_FILE.exists():
        raise FileNotFoundError(f"Split file not found:\n{SPLIT_FILE}")

    with open(SPLIT_FILE, "r", encoding="utf-8") as file:
        split = json.load(file)

    required_keys = ["training", "validation", "testing"]

    for key in required_keys:
        if key not in split:
            raise KeyError(f"'{key}' not found in {SPLIT_FILE}")

    return split


# ============================================================
# CHECK RECORD OVERLAP
# ============================================================


def check_record_overlap(split):

    training = set(split["training"])
    validation = set(split["validation"])
    testing = set(split["testing"])

    train_validation = training & validation
    train_testing = training & testing
    validation_testing = validation & testing

    if train_validation:
        raise ValueError(f"Training/validation overlap: {sorted(train_validation)}")

    if train_testing:
        raise ValueError(f"Training/testing overlap: {sorted(train_testing)}")

    if validation_testing:
        raise ValueError(f"Validation/testing overlap: {sorted(validation_testing)}")

    print("✓ No record overlap between splits")


# ============================================================
# LOAD ECG RECORD
# ============================================================


def load_record(record_name):

    record_path = str(DATASET_DIR / record_name)

    record = wfdb.rdrecord(record_path)

    annotation = wfdb.rdann(record_path, "atr")

    # Channel 0 = MLII
    ecg = record.p_signal[:, 0]

    sampling_frequency = record.fs

    return (ecg, sampling_frequency, annotation)


# ============================================================
# EXTRACT HEARTBEAT SEGMENTS
# ============================================================


def extract_segments(ecg, annotation):

    segments = []
    labels = []

    skipped_boundary = 0
    skipped_non_heartbeat = 0
    skipped_unknown = 0

    for sample_index, symbol in zip(annotation.sample, annotation.symbol):
        # ----------------------------------------------------
        # Ignore annotations that are not heartbeat classes
        # ----------------------------------------------------

        if not is_heartbeat(symbol):
            skipped_non_heartbeat += 1

            continue

        # ----------------------------------------------------
        # Convert annotation to our class
        # ----------------------------------------------------

        label = convert_label(symbol)

        if label is None:
            skipped_unknown += 1

            continue

        # ----------------------------------------------------
        # Calculate segment boundaries
        # ----------------------------------------------------

        start = int(sample_index) - SAMPLES_BEFORE

        end = int(sample_index) + SAMPLES_AFTER

        # ----------------------------------------------------
        # Skip beats too close to signal boundaries
        # ----------------------------------------------------

        if start < 0 or end > len(ecg):
            skipped_boundary += 1

            continue

        # ----------------------------------------------------
        # Extract heartbeat
        # ----------------------------------------------------

        segment = ecg[start:end]

        # Safety check
        if len(segment) != SEGMENT_LENGTH:
            skipped_boundary += 1

            continue

        segments.append(segment)
        labels.append(label)

    return (segments, labels, skipped_boundary, skipped_non_heartbeat, skipped_unknown)


# ============================================================
# PROCESS ONE RECORD
# ============================================================


def process_record(record_name):

    print(f"\nProcessing record {record_name}...")

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    ecg, fs, annotation = load_record(record_name)

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    processed_ecg = preprocess_ecg(ecg, fs)

    # --------------------------------------------------------
    # Segment
    # --------------------------------------------------------

    (segments, labels, skipped_boundary, skipped_non_heartbeat, skipped_unknown) = (
        extract_segments(processed_ecg, annotation)
    )

    print(f"  Sampling frequency : {fs} Hz")

    print(f"  Original samples    : {len(ecg)}")

    print(f"  Heartbeats extracted: {len(segments)}")

    print(f"  Boundary skipped    : {skipped_boundary}")

    print(f"  Non-heartbeat       : {skipped_non_heartbeat}")

    print(f"  Unknown labels      : {skipped_unknown}")

    return segments, labels


# ============================================================
# PROCESS SPLIT
# ============================================================


def process_split(record_list, split_name):

    all_segments = []
    all_labels = []

    print("\n")
    print("=" * 70)
    print(f"CREATING {split_name.upper()} DATASET")
    print("=" * 70)

    for record_name in record_list:
        segments, labels = process_record(record_name)

        all_segments.extend(segments)
        all_labels.extend(labels)

    # --------------------------------------------------------
    # Convert to NumPy arrays
    # --------------------------------------------------------

    X = np.asarray(all_segments, dtype=np.float32)

    y = np.asarray(all_labels, dtype="<U1")

    # --------------------------------------------------------
    # Verify shape
    # --------------------------------------------------------

    if len(X) != len(y):
        raise ValueError(f"{split_name}: X and y have different lengths")

    if len(X) > 0 and X.shape[1] != SEGMENT_LENGTH:
        raise ValueError(
            f"{split_name}: Expected segment length {SEGMENT_LENGTH}, got {X.shape[1]}"
        )

    print("\n")
    print(f"{split_name.upper()} DATASET CREATED")

    print(f"X shape: {X.shape}")

    print(f"y shape: {y.shape}")

    return X, y


# ============================================================
# CLASS DISTRIBUTION
# ============================================================


def print_class_distribution(y, split_name):

    classes = ["N", "S", "V", "F", "Q"]

    print("\n")
    print(f"{split_name.upper()} CLASS DISTRIBUTION")

    print("-" * 40)

    for class_name in classes:
        count = np.sum(y == class_name)

        print(f"{class_name:<10}: {count}")

    print("-" * 40)

    print(f"{'TOTAL':<10}: {len(y)}")


# ============================================================
# SAVE DATASET
# ============================================================


def save_dataset(X, y, split_name):

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X_file = OUTPUT_DIR / f"X_{split_name}.npy"

    y_file = OUTPUT_DIR / f"y_{split_name}.npy"

    np.save(X_file, X)

    np.save(y_file, y)

    print("\nSaved:")

    print(f"  {X_file}")

    print(f"  {y_file}")


# ============================================================
# SAVE DATASET METADATA
# ============================================================


def save_metadata(split, dataset_information):

    metadata = {
        "dataset": "MIT-BIH Arrhythmia Database",
        "signal_channel": "MLII (channel 0)",
        "segment_length": SEGMENT_LENGTH,
        "samples_before": SAMPLES_BEFORE,
        "samples_after": SAMPLES_AFTER,
        "classes": ["N", "S", "V", "F", "Q"],
        "label_mapping": {
            "N": "N",
            "L": "N",
            "R": "N",
            "e": "N",
            "j": "N",
            "A": "S",
            "a": "S",
            "J": "S",
            "S": "S",
            "V": "V",
            "E": "V",
            "F": "F",
            "/": "Q",
            "f": "Q",
            "Q": "Q",
        },
        "excluded_records": split.get("excluded", []),
        "training_records": split["training"],
        "validation_records": split["validation"],
        "testing_records": split["testing"],
        "dataset_shapes": dataset_information,
    }

    metadata_file = OUTPUT_DIR / "dataset_metadata.json"

    with open(metadata_file, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print(f"\nMetadata saved to:\n{metadata_file}")


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 70)
    print("MIT-BIH ECG DATASET CREATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Check dataset directory
    # --------------------------------------------------------

    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset directory not found:\n{DATASET_DIR}")

    print(f"\nDataset directory:\n{DATASET_DIR}")

    print(f"\nSplit file:\n{SPLIT_FILE}")

    print(f"\nSegment length: {SEGMENT_LENGTH} samples")

    # --------------------------------------------------------
    # Load optimized split
    # --------------------------------------------------------

    split = load_split()

    # --------------------------------------------------------
    # Verify no overlap
    # --------------------------------------------------------

    check_record_overlap(split)

    # --------------------------------------------------------
    # Process training
    # --------------------------------------------------------

    X_train, y_train = process_split(split["training"], "train")

    # --------------------------------------------------------
    # Process validation
    # --------------------------------------------------------

    X_val, y_val = process_split(split["validation"], "val")

    # --------------------------------------------------------
    # Process testing
    # --------------------------------------------------------

    X_test, y_test = process_split(split["testing"], "test")

    # --------------------------------------------------------
    # Print class distributions
    # --------------------------------------------------------

    print_class_distribution(y_train, "training")

    print_class_distribution(y_val, "validation")

    print_class_distribution(y_test, "testing")

    # --------------------------------------------------------
    # Save datasets
    # --------------------------------------------------------

    save_dataset(X_train, y_train, "train")

    save_dataset(X_val, y_val, "val")

    save_dataset(X_test, y_test, "test")

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    dataset_information = {
        "X_train": list(X_train.shape),
        "y_train": list(y_train.shape),
        "X_val": list(X_val.shape),
        "y_val": list(y_val.shape),
        "X_test": list(X_test.shape),
        "y_test": list(y_test.shape),
    }

    save_metadata(split, dataset_information)

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("DATASET CREATION COMPLETE")
    print("=" * 70)

    print(f"\nTraining   : {X_train.shape}")

    print(f"Validation : {X_val.shape}")

    print(f"Testing    : {X_test.shape}")

    print(f"\nDataset files are stored in:\n{OUTPUT_DIR}")

    print("\n✓ No original MIT-BIH files were modified")
    print("✓ Record-level split preserved")
    print("✓ MLII channel used")
    print("✓ 200-sample heartbeat windows created")
    print("✓ Preprocessing applied before segmentation")
    print("✓ Labels generated from annotations")


if __name__ == "__main__":
    main()
