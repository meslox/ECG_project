from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset"

OUTPUT_DIR = DATASET_DIR / "analysis"


# ============================================================
# SETTINGS
# ============================================================

CLASS_NAMES = ["N", "S", "V", "F", "Q"]

# Number of examples to display for each class
EXAMPLES_PER_CLASS = 1


# ============================================================
# LOAD DATASET
# ============================================================


def load_dataset():

    print("=" * 70)
    print("LOADING DATASET")
    print("=" * 70)

    files = {
        "X_train": DATASET_DIR / "X_train.npy",
        "y_train": DATASET_DIR / "y_train.npy",
        "X_val": DATASET_DIR / "X_val.npy",
        "y_val": DATASET_DIR / "y_val.npy",
        "X_test": DATASET_DIR / "X_test.npy",
        "y_test": DATASET_DIR / "y_test.npy",
    }

    for name, path in files.items():
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found:\n{path}")

    X_train = np.load(files["X_train"])
    y_train = np.load(files["y_train"])

    X_val = np.load(files["X_val"])
    y_val = np.load(files["y_val"])

    X_test = np.load(files["X_test"])
    y_test = np.load(files["y_test"])

    print("\nDataset loaded successfully.")

    return (X_train, y_train, X_val, y_val, X_test, y_test)


# ============================================================
# BASIC DATASET INFORMATION
# ============================================================


def print_dataset_information(X_train, y_train, X_val, y_val, X_test, y_test):

    print("\n")
    print("=" * 70)
    print("DATASET INFORMATION")
    print("=" * 70)

    print("\nTraining:")
    print(f"  X shape       : {X_train.shape}")
    print(f"  y shape       : {y_train.shape}")
    print(f"  Data type     : {X_train.dtype}")
    print(f"  Label type    : {y_train.dtype}")

    print("\nValidation:")
    print(f"  X shape       : {X_val.shape}")
    print(f"  y shape       : {y_val.shape}")
    print(f"  Data type     : {X_val.dtype}")
    print(f"  Label type    : {y_val.dtype}")

    print("\nTesting:")
    print(f"  X shape       : {X_test.shape}")
    print(f"  y shape       : {y_test.shape}")
    print(f"  Data type     : {X_test.dtype}")
    print(f"  Label type    : {y_test.dtype}")


# ============================================================
# CHECK X/Y LENGTH
# ============================================================


def check_lengths(X_train, y_train, X_val, y_val, X_test, y_test):

    print("\n")
    print("=" * 70)
    print("X / Y LENGTH CHECK")
    print("=" * 70)

    checks = [
        ("Training", X_train, y_train),
        ("Validation", X_val, y_val),
        ("Testing", X_test, y_test),
    ]

    all_ok = True

    for name, X, y in checks:
        if len(X) == len(y):
            print(f"✓ {name}: {len(X)} samples = {len(y)} labels")

        else:
            print(f"✗ {name}: {len(X)} samples != {len(y)} labels")

            all_ok = False

    if not all_ok:
        raise ValueError("X/y length mismatch detected.")


# ============================================================
# CHECK SHAPE
# ============================================================


def check_shape(X_train, X_val, X_test):

    print("\n")
    print("=" * 70)
    print("SEGMENT SHAPE CHECK")
    print("=" * 70)

    expected_dimensions = 2
    expected_length = 200

    datasets = [
        ("Training", X_train),
        ("Validation", X_val),
        ("Testing", X_test),
    ]

    for name, X in datasets:
        if X.ndim != expected_dimensions:
            raise ValueError(f"{name}: Expected 2D array, got {X.ndim}D")

        if X.shape[1] != expected_length:
            raise ValueError(
                f"{name}: Expected "
                f"{expected_length} samples per heartbeat, "
                f"got {X.shape[1]}"
            )

        print(f"✓ {name}: {X.shape[0]} × {X.shape[1]}")


# ============================================================
# CHECK NaN / INF
# ============================================================


def check_invalid_values(X_train, X_val, X_test):

    print("\n")
    print("=" * 70)
    print("NaN / INF CHECK")
    print("=" * 70)

    datasets = [
        ("Training", X_train),
        ("Validation", X_val),
        ("Testing", X_test),
    ]

    for name, X in datasets:
        nan_count = np.isnan(X).sum()
        inf_count = np.isinf(X).sum()

        if nan_count == 0 and inf_count == 0:
            print(f"✓ {name}: No NaN or Inf values")

        else:
            print(f"✗ {name}: NaN={nan_count}, Inf={inf_count}")


# ============================================================
# SIGNAL STATISTICS
# ============================================================


def print_signal_statistics(X_train, X_val, X_test):

    print("\n")
    print("=" * 70)
    print("SIGNAL STATISTICS")
    print("=" * 70)

    datasets = [
        ("Training", X_train),
        ("Validation", X_val),
        ("Testing", X_test),
    ]

    print(f"{'Dataset':<15}{'Min':>15}{'Max':>15}{'Mean':>15}{'Std':>15}")

    print("-" * 75)

    for name, X in datasets:
        print(
            f"{name:<15}"
            f"{np.min(X):>15.6f}"
            f"{np.max(X):>15.6f}"
            f"{np.mean(X):>15.6f}"
            f"{np.std(X):>15.6f}"
        )


# ============================================================
# CLASS DISTRIBUTION
# ============================================================


def get_class_counts(y):

    counts = {}

    for class_name in CLASS_NAMES:
        counts[class_name] = int(np.sum(y == class_name))

    return counts


def print_class_distribution(y_train, y_val, y_test):

    print("\n")
    print("=" * 70)
    print("CLASS DISTRIBUTION")
    print("=" * 70)

    train_counts = get_class_counts(y_train)
    val_counts = get_class_counts(y_val)
    test_counts = get_class_counts(y_test)

    print(f"{'Class':<10}{'Training':>12}{'Validation':>14}{'Testing':>12}")

    print("-" * 70)

    for class_name in CLASS_NAMES:
        print(
            f"{class_name:<10}"
            f"{train_counts[class_name]:>12}"
            f"{val_counts[class_name]:>14}"
            f"{test_counts[class_name]:>12}"
        )

    print("-" * 70)

    print(f"{'TOTAL':<10}{len(y_train):>12}{len(y_val):>14}{len(y_test):>12}")


# ============================================================
# CLASS PERCENTAGES
# ============================================================


def print_class_percentages(y_train, y_val, y_test):

    print("\n")
    print("=" * 70)
    print("CLASS PERCENTAGES")
    print("=" * 70)

    train_counts = get_class_counts(y_train)
    val_counts = get_class_counts(y_val)
    test_counts = get_class_counts(y_test)

    totals = {"train": len(y_train), "val": len(y_val), "test": len(y_test)}

    print(f"{'Class':<10}{'Training':>12}{'Validation':>14}{'Testing':>12}")

    print("-" * 70)

    for class_name in CLASS_NAMES:
        train_percentage = train_counts[class_name] / totals["train"] * 100

        val_percentage = val_counts[class_name] / totals["val"] * 100

        test_percentage = test_counts[class_name] / totals["test"] * 100

        print(
            f"{class_name:<10}"
            f"{train_percentage:>11.3f}%"
            f"{val_percentage:>13.3f}%"
            f"{test_percentage:>11.3f}%"
        )


# ============================================================
# PLOT CLASS DISTRIBUTION
# ============================================================


def plot_class_distribution(y_train, y_val, y_test):

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_counts = get_class_counts(y_train)
    val_counts = get_class_counts(y_val)
    test_counts = get_class_counts(y_test)

    x = np.arange(len(CLASS_NAMES))

    width = 0.25

    plt.figure(figsize=(10, 6))

    plt.bar(x - width, [train_counts[c] for c in CLASS_NAMES], width, label="Training")

    plt.bar(x, [val_counts[c] for c in CLASS_NAMES], width, label="Validation")

    plt.bar(x + width, [test_counts[c] for c in CLASS_NAMES], width, label="Testing")

    plt.xlabel("Class")
    plt.ylabel("Number of Heartbeats")

    plt.title("MIT-BIH ECG Class Distribution")

    plt.xticks(x, CLASS_NAMES)

    plt.legend()

    plt.tight_layout()

    output_file = OUTPUT_DIR / "class_distribution.png"

    plt.savefig(output_file, dpi=300)

    plt.close()

    print(f"\n✓ Class distribution plot saved:\n  {output_file}")


# ============================================================
# FIND FIRST SAMPLE OF EACH CLASS
# ============================================================


def find_class_examples(X, y):

    examples = {}

    for class_name in CLASS_NAMES:
        indices = np.where(y == class_name)[0]

        if len(indices) > 0:
            examples[class_name] = indices[0]

    return examples


# ============================================================
# PLOT EXAMPLE HEARTBEATS
# ============================================================


def plot_example_heartbeats(X_train, y_train):

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    examples = find_class_examples(X_train, y_train)

    for class_name in CLASS_NAMES:
        if class_name not in examples:
            print(f"⚠ No training example found for class {class_name}")

            continue

        index = examples[class_name]

        signal = X_train[index]

        plt.figure(figsize=(10, 4))

        plt.plot(signal)

        plt.xlabel("Sample")

        plt.ylabel("Amplitude")

        plt.title(f"Example ECG Heartbeat - Class {class_name}")

        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        output_file = OUTPUT_DIR / f"example_class_{class_name}.png"

        plt.savefig(output_file, dpi=300)

        plt.close()

        print(f"✓ Class {class_name} waveform saved:\n  {output_file}")


# ============================================================
# PLOT ALL CLASSES TOGETHER
# ============================================================


def plot_all_class_examples(X_train, y_train):

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    examples = find_class_examples(X_train, y_train)

    plt.figure(figsize=(12, 8))

    for class_name in CLASS_NAMES:
        if class_name not in examples:
            continue

        index = examples[class_name]

        signal = X_train[index]

        plt.plot(signal, label=f"Class {class_name}")

    plt.xlabel("Sample")

    plt.ylabel("Amplitude")

    plt.title("Example ECG Heartbeats by Class")

    plt.legend()

    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = OUTPUT_DIR / "all_class_examples.png"

    plt.savefig(output_file, dpi=300)

    plt.close()

    print(f"\n✓ Combined waveform plot saved:\n  {output_file}")


# ============================================================
# CHECK LABELS
# ============================================================


def check_labels(y_train, y_val, y_test):

    print("\n")
    print("=" * 70)
    print("LABEL CHECK")
    print("=" * 70)

    valid_classes = set(CLASS_NAMES)

    datasets = [
        ("Training", y_train),
        ("Validation", y_val),
        ("Testing", y_test),
    ]

    for name, y in datasets:
        actual_classes = set(np.unique(y))

        invalid_classes = actual_classes - valid_classes

        if invalid_classes:
            print(f"✗ {name}: Invalid labels found: {invalid_classes}")

        else:
            print(f"✓ {name}: All labels are valid")


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 70)
    print("MIT-BIH ECG DATASET ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    (X_train, y_train, X_val, y_val, X_test, y_test) = load_dataset()

    # --------------------------------------------------------
    # Basic information
    # --------------------------------------------------------

    print_dataset_information(X_train, y_train, X_val, y_val, X_test, y_test)

    # --------------------------------------------------------
    # Checks
    # --------------------------------------------------------

    check_lengths(X_train, y_train, X_val, y_val, X_test, y_test)

    check_shape(X_train, X_val, X_test)

    check_invalid_values(X_train, X_val, X_test)

    check_labels(y_train, y_val, y_test)

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print_signal_statistics(X_train, X_val, X_test)

    # --------------------------------------------------------
    # Class analysis
    # --------------------------------------------------------

    print_class_distribution(y_train, y_val, y_test)

    print_class_percentages(y_train, y_val, y_test)

    # --------------------------------------------------------
    # Create plots
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("CREATING ANALYSIS PLOTS")
    print("=" * 70)

    plot_class_distribution(y_train, y_val, y_test)

    plot_example_heartbeats(X_train, y_train)

    plot_all_class_examples(X_train, y_train)

    # --------------------------------------------------------
    # Final message
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("DATASET ANALYSIS COMPLETE")
    print("=" * 70)

    print(f"\nAnalysis files saved in:\n{OUTPUT_DIR}")

    print("\n✓ Dataset structure checked")
    print("✓ X/y lengths checked")
    print("✓ Segment dimensions checked")
    print("✓ NaN/Inf values checked")
    print("✓ Labels checked")
    print("✓ Signal statistics calculated")
    print("✓ Class distribution calculated")
    print("✓ ECG waveform plots created")


if __name__ == "__main__":
    main()
