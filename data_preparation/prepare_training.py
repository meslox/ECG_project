from pathlib import Path
import json

import numpy as np
from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset"
PREPARED_DIR = DATASET_DIR / "prepared"


# ============================================================
# CONFIGURATION
# ============================================================

CLASS_NAMES = ["N", "S", "V", "F", "Q"]

LABEL_TO_INT = {
    "N": 0,
    "S": 1,
    "V": 2,
    "F": 3,
    "Q": 4,
}

INT_TO_LABEL = {
    0: "N",
    1: "S",
    2: "V",
    3: "F",
    4: "Q",
}

SEGMENT_LENGTH = 200


# ============================================================
# LOAD DATA
# ============================================================


def load_dataset():
    print("=" * 70)
    print("LOADING DATASET")
    print("=" * 70)

    X_train = np.load(DATASET_DIR / "X_train.npy")
    y_train = np.load(DATASET_DIR / "y_train.npy")

    X_val = np.load(DATASET_DIR / "X_val.npy")
    y_val = np.load(DATASET_DIR / "y_val.npy")

    X_test = np.load(DATASET_DIR / "X_test.npy")
    y_test = np.load(DATASET_DIR / "y_test.npy")

    print("\nDataset loaded successfully.")

    print(f"\nTraining:")
    print(f"  X: {X_train.shape}")
    print(f"  y: {y_train.shape}")

    print(f"\nValidation:")
    print(f"  X: {X_val.shape}")
    print(f"  y: {y_val.shape}")

    print(f"\nTesting:")
    print(f"  X: {X_test.shape}")
    print(f"  y: {y_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test


# ============================================================
# CONVERT LABELS
# ============================================================


def convert_labels(labels):
    """
    Convert string labels:

        N -> 0
        S -> 1
        V -> 2
        F -> 3
        Q -> 4
    """

    converted = np.empty(len(labels), dtype=np.int64)

    for i, label in enumerate(labels):
        if label not in LABEL_TO_INT:
            raise ValueError(f"Unknown label found: {label}")

        converted[i] = LABEL_TO_INT[label]

    return converted


# ============================================================
# NORMALIZATION
# ============================================================


def calculate_normalization_parameters(X_train):
    """
    Calculate normalization parameters using TRAINING data only.

    This prevents information from validation/test data leaking
    into the training process.
    """

    mean = np.mean(X_train)
    std = np.std(X_train)

    if std == 0:
        raise ValueError("Training data standard deviation is zero.")

    return mean, std


def normalize_data(X, mean, std):
    """
    Standard score normalization:

        X_normalized = (X - mean) / std
    """

    return ((X - mean) / std).astype(np.float32)


# ============================================================
# RESHAPE FOR CNN + LSTM
# ============================================================


def reshape_for_model(X):
    """
    CNN + LSTM expects:

        (samples, timesteps, features)

    Current shape:

        (samples, 200)

    New shape:

        (samples, 200, 1)
    """

    if X.ndim != 2:
        raise ValueError(f"Expected 2D input, but received shape {X.shape}")

    if X.shape[1] != SEGMENT_LENGTH:
        raise ValueError(
            f"Expected {SEGMENT_LENGTH} samples per segment, but received {X.shape[1]}"
        )

    return X[..., np.newaxis]


# ============================================================
# CLASS DISTRIBUTION
# ============================================================


def print_class_distribution(y_train, y_val, y_test):

    print("\n")
    print("=" * 70)
    print("CLASS DISTRIBUTION")
    print("=" * 70)

    print(f"{'Class':<10}{'Training':>12}{'Validation':>14}{'Testing':>12}")

    print("-" * 70)

    for class_id, class_name in enumerate(CLASS_NAMES):
        train_count = np.sum(y_train == class_id)
        val_count = np.sum(y_val == class_id)
        test_count = np.sum(y_test == class_id)

        print(f"{class_name:<10}{train_count:>12}{val_count:>14}{test_count:>12}")

    print("-" * 70)

    print(f"{'TOTAL':<10}{len(y_train):>12}{len(y_val):>14}{len(y_test):>12}")


# ============================================================
# CLASS WEIGHTS
# ============================================================


def calculate_class_weights(y_train):

    print("\n")
    print("=" * 70)
    print("CALCULATING CLASS WEIGHTS")
    print("=" * 70)

    classes = np.unique(y_train)

    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)

    class_weights = {
        int(class_id): float(weight) for class_id, weight in zip(classes, weights)
    }

    print("\nClass weights:")

    for class_id, class_name in enumerate(CLASS_NAMES):
        if class_id in class_weights:
            print(f"  {class_name} ({class_id}) : {class_weights[class_id]:.6f}")
        else:
            print(f"  {class_name} ({class_id}) : NOT PRESENT")

    return class_weights


# ============================================================
# SAVE DATA
# ============================================================


def save_prepared_data(X_train, y_train, X_val, y_val, X_test, y_test):

    PREPARED_DIR.mkdir(parents=True, exist_ok=True)

    print("\n")
    print("=" * 70)
    print("SAVING PREPARED DATA")
    print("=" * 70)

    np.save(PREPARED_DIR / "X_train.npy", X_train)
    np.save(PREPARED_DIR / "y_train.npy", y_train)

    np.save(PREPARED_DIR / "X_val.npy", X_val)
    np.save(PREPARED_DIR / "y_val.npy", y_val)

    np.save(PREPARED_DIR / "X_test.npy", X_test)
    np.save(PREPARED_DIR / "y_test.npy", y_test)

    print("\nPrepared files saved to:")
    print(PREPARED_DIR)

    print("\nFiles:")

    for filename in [
        "X_train.npy",
        "y_train.npy",
        "X_val.npy",
        "y_val.npy",
        "X_test.npy",
        "y_test.npy",
    ]:
        print(f"  ✓ {filename}")


# ============================================================
# SAVE CONFIGURATION
# ============================================================


def save_configuration(mean, std, class_weights):

    configuration = {
        "classes": CLASS_NAMES,
        "label_to_int": LABEL_TO_INT,
        "int_to_label": {str(key): value for key, value in INT_TO_LABEL.items()},
        "segment_length": SEGMENT_LENGTH,
        "input_shape": [SEGMENT_LENGTH, 1],
        "normalization": {
            "method": "standard_score",
            "formula": "(X - mean) / std",
            "mean": float(mean),
            "std": float(std),
            "calculated_from": "training_data_only",
        },
        "class_weights": {str(key): value for key, value in class_weights.items()},
    }

    config_path = PREPARED_DIR / "training_config.json"

    with open(config_path, "w") as file:
        json.dump(configuration, file, indent=4)

    print("\nTraining configuration saved:")
    print(config_path)


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 70)
    print("MIT-BIH ECG TRAINING DATA PREPARATION")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load original dataset
    # --------------------------------------------------------

    (X_train, y_train, X_val, y_val, X_test, y_test) = load_dataset()

    # --------------------------------------------------------
    # 2. Convert labels
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("CONVERTING LABELS")
    print("=" * 70)

    y_train = convert_labels(y_train)
    y_val = convert_labels(y_val)
    y_test = convert_labels(y_test)

    print("\nLabel mapping:")

    for label, number in LABEL_TO_INT.items():
        print(f"  {label} -> {number}")

    print("\n✓ Labels converted successfully.")

    # --------------------------------------------------------
    # 3. Show class distribution
    # --------------------------------------------------------

    print_class_distribution(y_train, y_val, y_test)

    # --------------------------------------------------------
    # 4. Calculate class weights
    # --------------------------------------------------------

    class_weights = calculate_class_weights(y_train)

    # --------------------------------------------------------
    # 5. Calculate normalization parameters
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("CALCULATING NORMALIZATION PARAMETERS")
    print("=" * 70)

    mean, std = calculate_normalization_parameters(X_train)

    print(f"\nTraining mean : {mean:.8f}")
    print(f"Training std  : {std:.8f}")

    # --------------------------------------------------------
    # 6. Normalize
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("NORMALIZING DATA")
    print("=" * 70)

    X_train = normalize_data(X_train, mean, std)
    X_val = normalize_data(X_val, mean, std)
    X_test = normalize_data(X_test, mean, std)

    print("\n✓ Training data normalized")
    print("✓ Validation data normalized")
    print("✓ Testing data normalized")

    print("\nNormalization parameters came from TRAINING data only.")

    # --------------------------------------------------------
    # 7. Reshape
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("RESHAPING DATA FOR CNN + LSTM")
    print("=" * 70)

    X_train = reshape_for_model(X_train)
    X_val = reshape_for_model(X_val)
    X_test = reshape_for_model(X_test)

    print(f"\nTraining shape   : {X_train.shape}")
    print(f"Validation shape : {X_val.shape}")
    print(f"Testing shape    : {X_test.shape}")

    print("\nExpected format:")
    print("(samples, 200, 1)")

    # --------------------------------------------------------
    # 8. Save prepared data
    # --------------------------------------------------------

    save_prepared_data(X_train, y_train, X_val, y_val, X_test, y_test)

    # --------------------------------------------------------
    # 9. Save configuration
    # --------------------------------------------------------

    save_configuration(mean, std, class_weights)

    # --------------------------------------------------------
    # 10. Final checks
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("FINAL CHECKS")
    print("=" * 70)

    print(f"\nTraining:")
    print(f"  X shape : {X_train.shape}")
    print(f"  y shape : {y_train.shape}")
    print(f"  X dtype : {X_train.dtype}")
    print(f"  y dtype : {y_train.dtype}")

    print(f"\nValidation:")
    print(f"  X shape : {X_val.shape}")
    print(f"  y shape : {y_val.shape}")

    print(f"\nTesting:")
    print(f"  X shape : {X_test.shape}")
    print(f"  y shape : {y_test.shape}")

    print("\n")
    print("=" * 70)
    print("TRAINING DATA PREPARATION COMPLETE")
    print("=" * 70)

    print("\n✓ Labels converted")
    print("✓ Class weights calculated")
    print("✓ Normalization completed")
    print("✓ Data reshaped for CNN + LSTM")
    print("✓ Prepared data saved")
    print("✓ Training configuration saved")


if __name__ == "__main__":
    main()
