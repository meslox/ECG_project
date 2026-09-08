# model/evaluate_validation.py

from pathlib import Path
import json

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    ConfusionMatrixDisplay,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "dataset" / "prepared"

MODEL_PATH = PROJECT_ROOT / "model" / "saved_models" / "best_cnn_lstm.keras"

RESULTS_DIR = PROJECT_ROOT / "model" / "validation_results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

CLASS_NAMES = ["N", "S", "V", "F", "Q"]

NUM_CLASSES = len(CLASS_NAMES)


# ============================================================
# GPU CHECK
# ============================================================

print("=" * 60)
print("TensorFlow GPU Check")
print("=" * 60)

print("TensorFlow version:", tf.__version__)

gpus = tf.config.list_physical_devices("GPU")

if gpus:
    print("GPU detected:")
    for gpu in gpus:
        print(" ", gpu)
else:
    print("No GPU detected.")

print()


# ============================================================
# CHECK MODEL
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Best model not found:\n{MODEL_PATH}\n\nRun model/train.py first."
    )


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

print("=" * 60)
print("Loading Validation Dataset")
print("=" * 60)

X_val = np.load(DATA_DIR / "X_val.npy")
y_val = np.load(DATA_DIR / "y_val.npy")

X_val = X_val.astype(np.float32)
y_val = y_val.astype(np.int64)

print("X_val:", X_val.shape)
print("y_val:", y_val.shape)
print()


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("=" * 60)
print("Loading Best Model")
print("=" * 60)

print("Model:", MODEL_PATH)

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")
print()


# ============================================================
# MODEL PREDICTION
# ============================================================

print("=" * 60)
print("Making Validation Predictions")
print("=" * 60)

probabilities = model.predict(X_val, batch_size=64, verbose=1)

# Select class with highest probability
y_pred = np.argmax(probabilities, axis=1)

print("Predictions generated.")
print()


# ============================================================
# BASIC ACCURACY
# ============================================================

accuracy = accuracy_score(y_val, y_pred)

print("=" * 60)
print("VALIDATION ACCURACY")
print("=" * 60)

print(f"Accuracy: {accuracy:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")

print()


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_val, y_pred, labels=range(NUM_CLASSES))

print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print()

print("              Predicted")
print("          N      S      V      F      Q")

for i, class_name in enumerate(CLASS_NAMES):
    print(
        f"Actual {class_name} "
        f"{cm[i, 0]:6d} "
        f"{cm[i, 1]:6d} "
        f"{cm[i, 2]:6d} "
        f"{cm[i, 3]:6d} "
        f"{cm[i, 4]:6d}"
    )

print()


# ============================================================
# PER-CLASS METRICS
# ============================================================

precision, recall, f1, support = precision_recall_fscore_support(
    y_val, y_pred, labels=range(NUM_CLASSES), zero_division=0
)

print("=" * 60)
print("PER-CLASS PERFORMANCE")
print("=" * 60)

print(f"{'Class':<10}{'Precision':<15}{'Recall':<15}{'F1-score':<15}{'Support':<10}")

for i, class_name in enumerate(CLASS_NAMES):
    print(
        f"{class_name:<10}"
        f"{precision[i]:<15.4f}"
        f"{recall[i]:<15.4f}"
        f"{f1[i]:<15.4f}"
        f"{support[i]:<10d}"
    )

print()


# ============================================================
# MACRO / WEIGHTED AVERAGES
# ============================================================

macro_precision = np.mean(precision)
macro_recall = np.mean(recall)
macro_f1 = np.mean(f1)

weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
    y_val, y_pred, labels=range(NUM_CLASSES), average="weighted", zero_division=0
)

print("=" * 60)
print("OVERALL METRICS")
print("=" * 60)

print(f"Accuracy          : {accuracy:.4f}")
print(f"Macro Precision   : {macro_precision:.4f}")
print(f"Macro Recall      : {macro_recall:.4f}")
print(f"Macro F1          : {macro_f1:.4f}")

print()

print(f"Weighted Precision: {weighted_precision:.4f}")
print(f"Weighted Recall   : {weighted_recall:.4f}")
print(f"Weighted F1       : {weighted_f1:.4f}")

print()


# ============================================================
# PREDICTED CLASS DISTRIBUTION
# ============================================================

print("=" * 60)
print("PREDICTED CLASS DISTRIBUTION")
print("=" * 60)

predicted_counts = np.bincount(y_pred, minlength=NUM_CLASSES)

actual_counts = np.bincount(y_val, minlength=NUM_CLASSES)

print(f"{'Class':<10}{'Actual':<15}{'Predicted':<15}")

for i, class_name in enumerate(CLASS_NAMES):
    print(f"{class_name:<10}{actual_counts[i]:<15}{predicted_counts[i]:<15}")

print()


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_val, y_pred, labels=range(NUM_CLASSES), target_names=CLASS_NAMES, zero_division=0
)

print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(report)


# ============================================================
# SAVE CONFUSION MATRIX IMAGE
# ============================================================

plt.figure(figsize=(8, 7))

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)

disp.plot(values_format="d")

plt.title("Validation Confusion Matrix")
plt.tight_layout()

cm_path = RESULTS_DIR / "confusion_matrix.png"

plt.savefig(cm_path, dpi=150, bbox_inches="tight")

plt.close()

print("Confusion matrix saved to:")
print(cm_path)


# ============================================================
# SAVE METRICS TO JSON
# ============================================================

metrics = {
    "dataset": "Validation",
    "model": str(MODEL_PATH),
    "num_samples": int(len(y_val)),
    "accuracy": float(accuracy),
    "macro_precision": float(macro_precision),
    "macro_recall": float(macro_recall),
    "macro_f1": float(macro_f1),
    "weighted_precision": float(weighted_precision),
    "weighted_recall": float(weighted_recall),
    "weighted_f1": float(weighted_f1),
    "classes": {},
}


for i, class_name in enumerate(CLASS_NAMES):
    metrics["classes"][class_name] = {
        "precision": float(precision[i]),
        "recall": float(recall[i]),
        "f1_score": float(f1[i]),
        "support": int(support[i]),
    }


metrics["confusion_matrix"] = cm.tolist()

metrics["actual_class_counts"] = {
    CLASS_NAMES[i]: int(actual_counts[i]) for i in range(NUM_CLASSES)
}

metrics["predicted_class_counts"] = {
    CLASS_NAMES[i]: int(predicted_counts[i]) for i in range(NUM_CLASSES)
}


metrics_path = RESULTS_DIR / "validation_metrics.json"

with open(metrics_path, "w") as file:
    json.dump(metrics, file, indent=4)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

predictions_path = RESULTS_DIR / "validation_predictions.npy"

np.save(predictions_path, y_pred)


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 60)
print("VALIDATION EVALUATION COMPLETE")
print("=" * 60)

print()
print("Results saved in:")
print(RESULTS_DIR)

print()
print("Files:")
print(f"  {cm_path.name}")
print(f"  {metrics_path.name}")
print(f"  {predictions_path.name}")

print()
print("IMPORTANT:")
print("The test dataset (DS2) was NOT loaded or evaluated.")
