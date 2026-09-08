from pathlib import Path
import json
from collections import Counter

import wfdb

from labeling import convert_label, is_heartbeat


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "mit-bih-arrhythmia-database-1.0.0"

SPLIT_FILE = PROJECT_ROOT / "data_preparation" / "optimized_split.json"

CLASSES = ["N", "S", "V", "F", "Q"]


# ============================================================
# LOAD RECORD SPLIT
# ============================================================


def load_split():
    """
    Load training, validation and testing record names
    from dataset_split.json.
    """

    if not SPLIT_FILE.exists():
        raise FileNotFoundError(
            f"Split file not found:\n{SPLIT_FILE}\n\nRun split_dataset.py first."
        )

    with open(SPLIT_FILE, "r") as file:
        split_data = json.load(file)

    return (split_data["training"], split_data["validation"], split_data["testing"])


# ============================================================
# COUNT CLASSES FOR ONE RECORD
# ============================================================


def count_record_classes(record_name):
    """
    Read the annotations of one MIT-BIH record
    and count the five ECG classes.
    """

    record_path = DATASET_DIR / record_name

    annotation = wfdb.rdann(str(record_path), "atr")

    class_counts = Counter()

    for symbol in annotation.symbol:
        if not is_heartbeat(symbol):
            continue

        label = convert_label(symbol)

        if label in CLASSES:
            class_counts[label] += 1

    return class_counts


# ============================================================
# COUNT CLASSES FOR MULTIPLE RECORDS
# ============================================================


def count_split_classes(record_names):
    """
    Count ECG classes across all records in one split.
    """

    total_counts = Counter()

    for record_name in record_names:
        counts = count_record_classes(record_name)

        total_counts.update(counts)

    return total_counts


# ============================================================
# PRINT DISTRIBUTION
# ============================================================


def print_distribution(training_counts, validation_counts, testing_counts):
    """
    Print the class distribution for all three splits.
    """

    print()
    print("=" * 70)
    print("MIT-BIH ECG CLASS DISTRIBUTION")
    print("=" * 70)

    print()
    print(f"{'Class':<10}{'Training':>15}{'Validation':>15}{'Testing':>15}")

    print("-" * 70)

    for class_name in CLASSES:
        train = training_counts[class_name]
        validation = validation_counts[class_name]
        test = testing_counts[class_name]

        print(f"{class_name:<10}{train:>15}{validation:>15}{test:>15}")

    print("-" * 70)

    total_train = sum(training_counts.values())
    total_validation = sum(validation_counts.values())
    total_test = sum(testing_counts.values())

    print(f"{'TOTAL':<10}{total_train:>15}{total_validation:>15}{total_test:>15}")


# ============================================================
# CHECK MISSING CLASSES
# ============================================================


def check_missing_classes(training_counts, validation_counts, testing_counts):
    """
    Check whether any class is missing from a split.
    """

    print()
    print("=" * 70)
    print("CLASS AVAILABILITY CHECK")
    print("=" * 70)

    splits = {
        "Training": training_counts,
        "Validation": validation_counts,
        "Testing": testing_counts,
    }

    for split_name, counts in splits.items():
        missing = [class_name for class_name in CLASSES if counts[class_name] == 0]

        if missing:
            print(f"⚠ {split_name}: Missing classes -> {', '.join(missing)}")
        else:
            print(f"✓ {split_name}: All five classes are present")


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 70)
    print("MIT-BIH ECG CLASS DISTRIBUTION CHECK")
    print("=" * 70)

    # Load record split
    training_records, validation_records, testing_records = load_split()

    print()
    print(f"Training records   : {len(training_records)}")
    print(f"Validation records : {len(validation_records)}")
    print(f"Testing records    : {len(testing_records)}")

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print()
    print("Counting training classes...")

    training_counts = count_split_classes(training_records)

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print("Counting validation classes...")

    validation_counts = count_split_classes(validation_records)

    # --------------------------------------------------------
    # Testing
    # --------------------------------------------------------

    print("Counting testing classes...")

    testing_counts = count_split_classes(testing_records)

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print_distribution(training_counts, validation_counts, testing_counts)

    # --------------------------------------------------------
    # Check missing classes
    # --------------------------------------------------------

    check_missing_classes(training_counts, validation_counts, testing_counts)


if __name__ == "__main__":
    main()
