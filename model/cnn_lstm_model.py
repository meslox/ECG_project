import tensorflow as tf
from tensorflow.keras import layers, models


# ============================================================
# MODEL CONFIGURATION
# ============================================================

INPUT_SHAPE = (200, 1)
NUM_CLASSES = 5


# ============================================================
# CNN + LSTM MODEL
# ============================================================


def create_cnn_lstm_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
    """
    Create a 1D CNN + LSTM model for ECG heartbeat
    classification.

    Input:
        (200, 1)

    Output:
        5 classes:
            0 -> N
            1 -> S
            2 -> V
            3 -> F
            4 -> Q
    """

    model = models.Sequential(
        [
            # ------------------------------------------------
            # INPUT
            # ------------------------------------------------
            layers.Input(shape=input_shape),
            # ------------------------------------------------
            # CNN BLOCK 1
            # ------------------------------------------------
            layers.Conv1D(filters=32, kernel_size=7, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            # ------------------------------------------------
            # CNN BLOCK 2
            # ------------------------------------------------
            layers.Conv1D(filters=64, kernel_size=5, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            # ------------------------------------------------
            # CNN BLOCK 3
            # ------------------------------------------------
            layers.Conv1D(
                filters=128, kernel_size=3, padding="same", activation="relu"
            ),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            # ------------------------------------------------
            # LSTM
            # ------------------------------------------------
            layers.LSTM(64, return_sequences=False),
            # ------------------------------------------------
            # FULLY CONNECTED LAYER
            # ------------------------------------------------
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.5),
            # ------------------------------------------------
            # OUTPUT
            # ------------------------------------------------
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="ECG_CNN_LSTM",
    )

    return model


# ============================================================
# MODEL SUMMARY
# ============================================================

if __name__ == "__main__":
    model = create_cnn_lstm_model()

    model.summary()
