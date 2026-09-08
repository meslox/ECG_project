from pathlib import Path
from collections import Counter

import wfdb

from labeling import convert_label, is_heartbeat


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "mit-bih-arrhythmia-database-1.0.0"

CLASSES = ["N", "S", "V", "F", "Q"]


# ============================================================
# FIND RECORDS
# ============================================================


def get_record_names():
    """
    Find all MIT-BIH record names using .hea files.
    """

    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset directory not found:\n{DATASET_DIR}")

    record_names = []

    for header_file in DATASET_DIR.glob("*.hea"):
        record_names.append(header_file.stem)

    record_names.sort()

    return record_names


# ============================================================
# COUNT CLASSES IN ONE RECORD
# ============================================================


def count_record_classes(record_name):
    """
    Count N, S, V, F and Q beats in one MIT-BIH record.
    """

    record_path = DATASET_DIR / record_name

    annotation = wfdb.rdann(str(record_path), "atr")

    counts = Counter()

    for symbol in annotation.symbol:
        # Ignore non-heartbeat annotations
        if not is_heartbeat(symbol):
            continue

        # Convert MIT-BIH symbol to our 5 classes
        label = convert_label(symbol)

        if label in CLASSES:
            counts[label] += 1

    return counts


# ============================================================
# PRINT RECORD TABLE
# ============================================================


def print_record_distribution(record_counts):
    """
    Print the class distribution for every record.
    """

    print()
    print("=" * 85)
    print("MIT-BIH CLASS DISTRIBUTION PER RECORD")
    print("=" * 85)

    print()

    print(f"{'Record':<10}{'N':>12}{'S':>12}{'V':>12}{'F':>12}{'Q':>12}")

    print("-" * 85)

    for record_name, counts in record_counts.items():
        print(
            f"{record_name:<10}"
            f"{counts['N']:>12}"
            f"{counts['S']:>12}"
            f"{counts['V']:>12}"
            f"{counts['F']:>12}"
            f"{counts['Q']:>12}"
        )


# ============================================================
# PRINT TOTAL
# ============================================================


def print_total(record_counts):
    """
    Print the total number of beats across all records.
    """

    total = Counter()

    for counts in record_counts.values():
        total.update(counts)

    print("-" * 85)

    print(
        f"{'TOTAL':<10}"
        f"{total['N']:>12}"
        f"{total['S']:>12}"
        f"{total['V']:>12}"
        f"{total['F']:>12}"
        f"{total['Q']:>12}"
    )

    print()

    print("=" * 85)
    print("TOTAL BEATS:", sum(total.values()))
    print("=" * 85)


# ============================================================
# FIND RECORDS CONTAINING EACH CLASS
# ============================================================


def print_class_records(record_counts):
    """
    Show which records contain each ECG class.
    """

    print()
    print("=" * 85)
    print("RECORDS CONTAINING EACH CLASS")
    print("=" * 85)

    for class_name in CLASSES:
        records = []

        for record_name, counts in record_counts.items():
            if counts[class_name] > 0:
                records.append(record_name)

        print()
        print(f"{class_name} class ({len(records)} records):")

        print(", ".join(records))


# ============================================================
# FIND RARE CLASS RECORDS
# ============================================================


def print_f_class_records(record_counts):
    """
    Show detailed information about records containing F beats.

    F is usually one of the rarest classes in this mapping,
    so these records are important when designing the split.
    """

    print()
    print("=" * 85)
    print("F CLASS RECORDS")
    print("=" * 85)

    f_records = []

    for record_name, counts in record_counts.items():
        if counts["F"] > 0:
            f_records.append((record_name, counts["F"]))

    if not f_records:
        print("No F-class beats found.")

        return

    for record_name, count in f_records:
        print(f"Record {record_name:<5} F beats: {count}")


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 85)
    print("MIT-BIH ECG RECORD CLASS DISTRIBUTION")
    print("=" * 85)

    print()
    print("Dataset directory:")
    print(DATASET_DIR)

    # --------------------------------------------------------
    # Find records
    # --------------------------------------------------------

    record_names = get_record_names()

    print()
    print(f"Records found: {len(record_names)}")

    if len(record_names) == 0:
        raise RuntimeError("No MIT-BIH records were found.")

    # --------------------------------------------------------
    # Count classes
    # --------------------------------------------------------

    record_counts = {}

    print()
    print("Reading annotations...")

    for record_name in record_names:
        print(f"Processing record {record_name}...")

        record_counts[record_name] = count_record_classes(record_name)

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print_record_distribution(record_counts)

    print_total(record_counts)

    print_class_records(record_counts)

    print_f_class_records(record_counts)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
