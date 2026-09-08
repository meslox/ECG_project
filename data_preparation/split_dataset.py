from pathlib import Path
import json


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "mit-bih-arrhythmia-database-1.0.0"

OUTPUT_FILE = PROJECT_ROOT / "dataset_split.json"


# ============================================================
# RECORDS EXCLUDED FROM THE EXPERIMENT
# ============================================================
#
# These records remain in the original MIT-BIH directory.
# They are simply not used for this classification experiment.
#
# ============================================================

EXCLUDED_RECORDS = {"102", "104", "107", "217"}


# ============================================================
# MIT-BIH INTER-PATIENT SPLIT
# ============================================================
#
# DS1 = development/training group
# DS2 = independent testing group
#
# The testing records are kept completely separate from
# training and validation.
#
# ============================================================

DS1_RECORDS = [
    "101",
    "106",
    "108",
    "109",
    "112",
    "114",
    "115",
    "116",
    "118",
    "119",
    "122",
    "124",
    "201",
    "203",
    "205",
    "207",
    "208",
    "209",
    "215",
    "220",
    "223",
    "230",
]


DS2_RECORDS = [
    "100",
    "103",
    "105",
    "111",
    "113",
    "117",
    "121",
    "123",
    "200",
    "202",
    "210",
    "212",
    "213",
    "214",
    "219",
    "221",
    "222",
    "228",
    "231",
    "232",
    "233",
    "234",
]


# ============================================================
# VALIDATION RECORDS
# ============================================================
#
# We take a portion of DS1 for validation.
#
# IMPORTANT:
# DS2 is NEVER used for validation.
# DS2 remains the final independent test set.
#
# ============================================================

VALIDATION_RECORDS = ["108", "114", "119", "201", "208"]


# ============================================================
# CREATE TRAINING RECORDS
# ============================================================


def create_training_records():
    """
    Training records are the DS1 records that are not
    assigned to validation.
    """

    training_records = [
        record for record in DS1_RECORDS if record not in VALIDATION_RECORDS
    ]

    return training_records


# ============================================================
# CHECK DATASET RECORDS
# ============================================================


def check_records(training_records):
    """
    Check that the train, validation and test sets are valid.

    Ensures:
        - No excluded records are used.
        - No record occurs in more than one split.
        - All records actually exist.
    """

    all_splits = {
        "training": training_records,
        "validation": VALIDATION_RECORDS,
        "testing": DS2_RECORDS,
    }

    # --------------------------------------------------------
    # Check excluded records
    # --------------------------------------------------------

    for split_name, records in all_splits.items():
        for record in records:
            if record in EXCLUDED_RECORDS:
                raise ValueError(
                    f"Excluded record {record} found in {split_name} split."
                )

    # --------------------------------------------------------
    # Check for overlapping records
    # --------------------------------------------------------

    training_set = set(training_records)
    validation_set = set(VALIDATION_RECORDS)
    testing_set = set(DS2_RECORDS)

    if training_set & validation_set:
        raise ValueError("Training and validation records overlap.")

    if training_set & testing_set:
        raise ValueError("Training and testing records overlap.")

    if validation_set & testing_set:
        raise ValueError("Validation and testing records overlap.")

    # --------------------------------------------------------
    # Check that records exist
    # --------------------------------------------------------

    all_records = training_records + VALIDATION_RECORDS + DS2_RECORDS

    for record in all_records:
        header_file = DATASET_DIR / f"{record}.hea"

        if not header_file.exists():
            raise FileNotFoundError(f"Record {record} not found:\n{header_file}")


# ============================================================
# SAVE SPLIT
# ============================================================


def save_split(training_records):
    """
    Save the final record split to dataset_split.json.
    """

    split_data = {
        "description": "MIT-BIH inter-patient record-level split",
        "excluded_records": sorted(EXCLUDED_RECORDS),
        "development_set": sorted(DS1_RECORDS),
        "testing_set": sorted(DS2_RECORDS),
        "training": sorted(training_records),
        "validation": sorted(VALIDATION_RECORDS),
        "testing": sorted(DS2_RECORDS),
    }

    with open(OUTPUT_FILE, "w") as file:
        json.dump(split_data, file, indent=4)


# ============================================================
# PRINT RESULTS
# ============================================================


def print_results(training_records):
    """
    Display the final record split.
    """

    print()
    print("=" * 70)
    print("MIT-BIH INTER-PATIENT DATASET SPLIT")
    print("=" * 70)

    print()

    print(
        f"Total original records : "
        f"{len(DS1_RECORDS) + len(DS2_RECORDS) + len(EXCLUDED_RECORDS)}"
    )

    print(f"Excluded records       : {len(EXCLUDED_RECORDS)}")

    print(f"Development records    : {len(DS1_RECORDS)}")

    print(f"Training records       : {len(training_records)}")

    print(f"Validation records     : {len(VALIDATION_RECORDS)}")

    print(f"Testing records        : {len(DS2_RECORDS)}")

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("TRAINING RECORDS")
    print("-" * 70)

    print(", ".join(training_records))

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("VALIDATION RECORDS")
    print("-" * 70)

    print(", ".join(VALIDATION_RECORDS))

    # --------------------------------------------------------
    # Testing
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("TESTING RECORDS")
    print("-" * 70)

    print(", ".join(DS2_RECORDS))

    # --------------------------------------------------------
    # Excluded
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("EXCLUDED RECORDS")
    print("-" * 70)

    print(", ".join(sorted(EXCLUDED_RECORDS)))

    # --------------------------------------------------------
    # Output file
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SPLIT SUCCESSFULLY CREATED")
    print("=" * 70)

    print()
    print("Saved to:")
    print(OUTPUT_FILE)


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 70)
    print("MIT-BIH ECG DATASET SPLITTING")
    print("=" * 70)

    # Check dataset directory
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset directory not found:\n{DATASET_DIR}")

    # Create training records
    training_records = create_training_records()

    # Validate the split
    check_records(training_records)

    # Save JSON
    save_split(training_records)

    # Display results
    print_results(training_records)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
