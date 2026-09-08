# model/train.py

from pathlib import Path
import json

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from cnn_lstm_model import create_cnn_lstm_model


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "dataset" / "prepared"
MODEL_DIR = PROJECT_ROOT / "model" / "saved_models"
RESULTS_DIR = PROJECT_ROOT / "model" / "training_results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

BATCH_SIZE = 64
EPOCHS = 50

LEARNING_RATE = 0.001

# Q has only 6 training samples, so its raw calculated
# class weight (~1368) is far too large for a first experiment.
MAX_CLASS_WEIGHT = 20.0

RANDOM_SEED = 42

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


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
    print("WARNING: No GPU detected. Training will use CPU.")

print()


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("Loading Prepared Dataset")
print("=" * 60)

X_train = np.load(DATA_DIR / "X_train.npy")
y_train = np.load(DATA_DIR / "y_train.npy")

X_val = np.load(DATA_DIR / "X_val.npy")
y_val = np.load(DATA_DIR / "y_val.npy")

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_val  :", X_val.shape)
print("y_val  :", y_val.shape)

print()


# ============================================================
# DATA TYPE
# ============================================================

X_train = X_train.astype(np.float32)
X_val = X_val.astype(np.float32)

y_train = y_train.astype(np.int64)
y_val = y_val.astype(np.int64)


# ============================================================
# CLASS INFORMATION
# ============================================================

CLASS_NAMES = ["N", "S", "V", "F", "Q"]
NUM_CLASSES = len(CLASS_NAMES)

print("=" * 60)
print("Training Class Distribution")
print("=" * 60)

class_counts = np.bincount(y_train, minlength=NUM_CLASSES)

for class_id, class_name in enumerate(CLASS_NAMES):
    print(f"{class_name}: {class_counts[class_id]} samples")

print()


# ============================================================
# CALCULATE CLASS WEIGHTS
# ============================================================

print("=" * 60)
print("Class Weights")
print("=" * 60)

total_samples = len(y_train)

class_weights = {}

for class_id in range(NUM_CLASSES):
    count = class_counts[class_id]

    if count == 0:
        class_weights[class_id] = 1.0
        continue

    # Balanced-class formula
    weight = total_samples / (NUM_CLASSES * count)

    # Prevent extreme weight values
    weight = min(weight, MAX_CLASS_WEIGHT)

    class_weights[class_id] = float(weight)

    print(f"{CLASS_NAMES[class_id]}: {class_weights[class_id]:.4f}")

print()
print(f"Maximum class weight capped at: {MAX_CLASS_WEIGHT}")
print()


# ============================================================
# BUILD MODEL
# ============================================================

print("=" * 60)
print("Building CNN + LSTM Model")
print("=" * 60)

model = create_cnn_lstm_model(input_shape=(200, 1), num_classes=NUM_CLASSES)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

print()


# ============================================================
# CALLBACKS
# ============================================================

BEST_MODEL_PATH = MODEL_DIR / "best_cnn_lstm.keras"

callbacks = [
    # Save the model whenever validation loss improves
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(BEST_MODEL_PATH),
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1,
    ),
    # Reduce learning rate when validation loss stops improving
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
    ),
    # Stop training if validation loss stops improving
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=7, mode="min", restore_best_weights=True, verbose=1
    ),
]


# ============================================================
# TRAIN MODEL
# ============================================================

print("=" * 60)
print("Starting Training")
print("=" * 60)

print("Batch size :", BATCH_SIZE)
print("Epochs     :", EPOCHS)
print("Learning rate:", LEARNING_RATE)
print()

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    class_weight=None,
    callbacks=callbacks,
    shuffle=True,
    verbose=1,
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

FINAL_MODEL_PATH = MODEL_DIR / "final_cnn_lstm.keras"

model.save(FINAL_MODEL_PATH)

print()
print("Final model saved to:")
print(FINAL_MODEL_PATH)


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_path = RESULTS_DIR / "training_history.json"

history_data = {
    key: [float(value) for value in values] for key, values in history.history.items()
}

with open(history_path, "w") as file:
    json.dump(history_data, file, indent=4)

print()
print("Training history saved to:")
print(history_path)


# ============================================================
# SAVE TRAINING CONFIGURATION
# ============================================================

config = {
    "model": "ECG_CNN_LSTM",
    "input_shape": [200, 1],
    "num_classes": NUM_CLASSES,
    "class_names": CLASS_NAMES,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "learning_rate": LEARNING_RATE,
    "max_class_weight": MAX_CLASS_WEIGHT,
    "class_weights": {
        CLASS_NAMES[class_id]: weight for class_id, weight in class_weights.items()
    },
    "optimizer": "Adam",
    "loss": "sparse_categorical_crossentropy",
    "seed": RANDOM_SEED,
}

config_path = RESULTS_DIR / "training_config.json"

with open(config_path, "w") as file:
    json.dump(config, file, indent=4)

print("Training configuration saved to:")
print(config_path)


# ============================================================
# PLOT TRAINING ACCURACY
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(history.history["accuracy"], label="Training Accuracy")

plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("CNN + LSTM Training Accuracy")

plt.legend()
plt.grid(True)

accuracy_plot = RESULTS_DIR / "accuracy.png"

plt.savefig(accuracy_plot, dpi=150, bbox_inches="tight")

plt.close()

print()
print("Accuracy plot saved to:")
print(accuracy_plot)


# ============================================================
# PLOT TRAINING LOSS
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(history.history["loss"], label="Training Loss")

plt.plot(history.history["val_loss"], label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CNN + LSTM Training Loss")

plt.legend()
plt.grid(True)

loss_plot = RESULTS_DIR / "loss.png"

plt.savefig(loss_plot, dpi=150, bbox_inches="tight")

plt.close()

print("Loss plot saved to:")
print(loss_plot)


# ============================================================
# TRAINING COMPLETE
# ============================================================

print()
print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print("Best model :")
print(BEST_MODEL_PATH)

print()
print("Final model:")
print(FINAL_MODEL_PATH)

print()
print("Training results:")
print(RESULTS_DIR)
