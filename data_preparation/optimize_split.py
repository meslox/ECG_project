from pathlib import Path
from itertools import combinations
import json

import wfdb
from labeling import convert_label, is_heartbeat


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "mit-bih-arrhythmia-database-1.0.0"

OUTPUT_FILE = PROJECT_ROOT / "data_preparation" / "optimized_split.json"


# ============================================================
# DATASET SETTINGS
# ============================================================

CLASS_NAMES = ["N", "S", "V", "F", "Q"]

# Paced-beat records excluded from this experiment
EXCLUDED_RECORDS = {"102", "104", "107", "217"}


# Standard development set (DS1)
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


# Standard independent test set (DS2)
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


# Number of records to use for validation
MIN_VALIDATION_RECORDS = 4
MAX_VALIDATION_RECORDS = 6


# ============================================================
# LOAD CLASS COUNTS FOR ONE RECORD
# ============================================================


def get_record_class_counts(record_name):
    """
    Read one MIT-BIH record and count the five classes.
    """

    annotation_path = str(DATASET_DIR / record_name)

    annotation = wfdb.rdann(annotation_path, "atr")

    counts = {class_name: 0 for class_name in CLASS_NAMES}

    for symbol in annotation.symbol:
        if not is_heartbeat(symbol):
            continue

        label = convert_label(symbol)

        if label in counts:
            counts[label] += 1

    return counts


# ============================================================
# LOAD ALL RECORD COUNTS
# ============================================================


def load_all_counts(records):

    all_counts = {}

    print("\nReading record annotations...\n")

    for record in records:
        print(f"Processing record {record} ...", end=" ")

        counts = get_record_class_counts(record)

        all_counts[record] = counts

        print("done")

    return all_counts


# ============================================================
# ADD COUNTS
# ============================================================


def calculate_total_counts(records, all_counts):

    total = {class_name: 0 for class_name in CLASS_NAMES}

    for record in records:
        for class_name in CLASS_NAMES:
            total[class_name] += all_counts[record][class_name]

    return total


# ============================================================
# CALCULATE TOTAL BEATS
# ============================================================


def total_beats(counts):

    return sum(counts.values())


# ============================================================
# CALCULATE CLASS PROPORTIONS
# ============================================================


def class_proportions(counts):

    beats = total_beats(counts)

    if beats == 0:
        return {class_name: 0.0 for class_name in CLASS_NAMES}

    return {class_name: counts[class_name] / beats for class_name in CLASS_NAMES}


# ============================================================
# SCORE VALIDATION SPLIT
# ============================================================


def score_validation_split(validation_records, all_counts, ds1_counts):
    """
    Score a candidate validation set.

    The objective tries to:

    1. Keep validation size around 20% of DS1.
    2. Preserve all classes in both training and validation.
    3. Keep validation class proportions reasonably
       close to DS1 proportions.
    4. Give some preference to having enough F and Q
       samples in validation.

    Lower score = better split.
    """

    validation_counts = calculate_total_counts(validation_records, all_counts)

    training_records = [
        record for record in DS1_RECORDS if record not in validation_records
    ]

    training_counts = calculate_total_counts(training_records, all_counts)

    ds1_proportions = class_proportions(ds1_counts)
    validation_proportions = class_proportions(validation_counts)
    training_proportions = class_proportions(training_counts)

    score = 0.0

    # --------------------------------------------------------
    # 1. Validation size
    # --------------------------------------------------------

    ds1_total_beats = total_beats(ds1_counts)
    validation_total_beats = total_beats(validation_counts)

    validation_ratio = validation_total_beats / ds1_total_beats

    # Target approximately 20%
    score += abs(validation_ratio - 0.20) * 100

    # --------------------------------------------------------
    # 2. Missing-class penalty
    # --------------------------------------------------------

    for class_name in CLASS_NAMES:
        if validation_counts[class_name] == 0:
            score += 100

        if training_counts[class_name] == 0:
            score += 100

    # --------------------------------------------------------
    # 3. Class distribution similarity
    # --------------------------------------------------------

    for class_name in CLASS_NAMES:
        difference = abs(
            validation_proportions[class_name] - ds1_proportions[class_name]
        )

        score += difference * 100

        difference_training = abs(
            training_proportions[class_name] - ds1_proportions[class_name]
        )

        score += difference_training * 50

    # --------------------------------------------------------
    # 4. Rare-class minimums
    # --------------------------------------------------------

    # F should preferably have at least 10 beats
    if validation_counts["F"] < 10:
        score += (10 - validation_counts["F"]) * 2

    # Q should preferably have at least 2 beats
    if validation_counts["Q"] < 2:
        score += (2 - validation_counts["Q"]) * 10

    # --------------------------------------------------------
    # 5. Avoid huge concentration of F in validation
    # --------------------------------------------------------

    if ds1_counts["F"] > 0:
        f_ratio = validation_counts["F"] / ds1_counts["F"]

        # Ideally don't put more than 25% of F in validation
        if f_ratio > 0.25:
            score += (f_ratio - 0.25) * 100

    # --------------------------------------------------------
    # 6. Avoid huge concentration of V in validation
    # --------------------------------------------------------

    if ds1_counts["V"] > 0:
        v_ratio = validation_counts["V"] / ds1_counts["V"]

        if v_ratio > 0.30:
            score += (v_ratio - 0.30) * 50

    return score


# ============================================================
# FIND BEST VALIDATION SPLIT
# ============================================================


def find_best_validation_split(all_counts):

    ds1_counts = calculate_total_counts(DS1_RECORDS, all_counts)

    print("\n" + "=" * 70)
    print("DS1 CLASS DISTRIBUTION")
    print("=" * 70)

    print(f"{'Class':<10}{'Count':>12}{'Percentage':>15}")

    ds1_total = total_beats(ds1_counts)

    for class_name in CLASS_NAMES:
        percentage = ds1_counts[class_name] / ds1_total * 100

        print(f"{class_name:<10}{ds1_counts[class_name]:>12}{percentage:>14.3f}%")

    print(f"\nTotal DS1 beats: {ds1_total}")

    # --------------------------------------------------------
    # Search every combination
    # --------------------------------------------------------

    best_score = float("inf")
    best_validation_records = None

    number_tested = 0

    print("\n")
    print("=" * 70)
    print("SEARCHING FOR BEST VALIDATION SPLIT")
    print("=" * 70)

    for size in range(MIN_VALIDATION_RECORDS, MAX_VALIDATION_RECORDS + 1):
        print(f"\nTesting validation sets with {size} records...")

        for combination in combinations(DS1_RECORDS, size):
            number_tested += 1

            score = score_validation_split(combination, all_counts, ds1_counts)

            if score < best_score:
                best_score = score

                best_validation_records = list(combination)

    print(f"\nCandidate splits tested: {number_tested}")

    return (best_validation_records, best_score)


# ============================================================
# PRINT FINAL SPLIT
# ============================================================


def print_final_split(validation_records, all_counts):

    training_records = [
        record for record in DS1_RECORDS if record not in validation_records
    ]

    validation_counts = calculate_total_counts(validation_records, all_counts)

    training_counts = calculate_total_counts(training_records, all_counts)

    testing_counts = calculate_total_counts(DS2_RECORDS, all_counts)

    # --------------------------------------------------------
    # Record lists
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("OPTIMIZED DATASET SPLIT")
    print("=" * 70)

    print("\nTRAINING RECORDS:")
    print(", ".join(training_records))

    print(f"\nTraining records: {len(training_records)}")

    print("\nVALIDATION RECORDS:")
    print(", ".join(validation_records))

    print(f"\nValidation records: {len(validation_records)}")

    print("\nTESTING RECORDS:")
    print(", ".join(DS2_RECORDS))

    print(f"\nTesting records: {len(DS2_RECORDS)}")

    print("\nEXCLUDED RECORDS:")
    print(", ".join(sorted(EXCLUDED_RECORDS)))

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("CLASS DISTRIBUTION")
    print("=" * 70)

    print(f"{'Class':<10}{'Training':>12}{'Validation':>14}{'Testing':>12}")

    print("-" * 70)

    for class_name in CLASS_NAMES:
        print(
            f"{class_name:<10}"
            f"{training_counts[class_name]:>12}"
            f"{validation_counts[class_name]:>14}"
            f"{testing_counts[class_name]:>12}"
        )

    print("-" * 70)

    print(
        f"{'TOTAL':<10}"
        f"{total_beats(training_counts):>12}"
        f"{total_beats(validation_counts):>14}"
        f"{total_beats(testing_counts):>12}"
    )

    # --------------------------------------------------------
    # Percentages
    # --------------------------------------------------------

    print("\nCLASS PERCENTAGES")

    training_proportions = class_proportions(training_counts)

    validation_proportions = class_proportions(validation_counts)

    testing_proportions = class_proportions(testing_counts)

    print(f"{'Class':<10}{'Training':>12}{'Validation':>14}{'Testing':>12}")

    for class_name in CLASS_NAMES:
        print(
            f"{class_name:<10}"
            f"{training_proportions[class_name] * 100:>11.3f}%"
            f"{validation_proportions[class_name] * 100:>13.3f}%"
            f"{testing_proportions[class_name] * 100:>11.3f}%"
        )

    return {
        "training": training_records,
        "validation": validation_records,
        "testing": DS2_RECORDS,
        "excluded": sorted(EXCLUDED_RECORDS),
        "class_distribution": {
            "training": training_counts,
            "validation": validation_counts,
            "testing": testing_counts,
        },
    }


# ============================================================
# SAVE SPLIT
# ============================================================


def save_split(split_data):

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(split_data, file, indent=4)

    print("\n")
    print("=" * 70)
    print(f"Split saved to:")
    print(OUTPUT_FILE)
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================


def main():

    # --------------------------------------------------------
    # Check dataset directory
    # --------------------------------------------------------

    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset directory not found:\n{DATASET_DIR}")

    # --------------------------------------------------------
    # Check record lists
    # --------------------------------------------------------

    all_required_records = DS1_RECORDS + DS2_RECORDS + list(EXCLUDED_RECORDS)

    missing_records = []

    for record in all_required_records:
        hea_file = DATASET_DIR / f"{record}.hea"

        if not hea_file.exists():
            missing_records.append(record)

    if missing_records:
        raise FileNotFoundError(
            "The following records were not found:\n" + ", ".join(missing_records)
        )

    # --------------------------------------------------------
    # Load counts
    # --------------------------------------------------------

    all_records = DS1_RECORDS + DS2_RECORDS

    all_counts = load_all_counts(all_records)

    # --------------------------------------------------------
    # Find best validation set
    # --------------------------------------------------------

    validation_records, best_score = find_best_validation_split(all_counts)

    # --------------------------------------------------------
    # Print result
    # --------------------------------------------------------

    print(f"\nBest split score: {best_score:.4f}")

    split_data = print_final_split(validation_records, all_counts)

    # --------------------------------------------------------
    # Add metadata
    # --------------------------------------------------------

    split_data["method"] = (
        "Record-level validation optimization "
        "within DS1; DS2 kept fixed as independent test set"
    )

    split_data["validation_score"] = best_score

    split_data["settings"] = {
        "min_validation_records": MIN_VALIDATION_RECORDS,
        "max_validation_records": MAX_VALIDATION_RECORDS,
        "excluded_records": sorted(EXCLUDED_RECORDS),
        "classes": CLASS_NAMES,
    }

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_split(split_data)


if __name__ == "__main__":
    main()
