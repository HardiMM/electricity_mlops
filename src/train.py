import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

EXPERIMENT_NAME = sys.argv[1] if len(sys.argv) > 1 else "experiment_1"

EXPERIMENTS = {
    "experiment_1": {
        "lstm_1": 64,
        "lstm_2": 32,
        "dropout": 0.2
    },

    "experiment_2": {
        "lstm_1": 32,
        "lstm_2": 16,
        "dropout": 0.2
    },

    "experiment_3": {
        "lstm_1": 64,
        "lstm_2": 32,
        "dropout": 0.3
    }
}

config = EXPERIMENTS[EXPERIMENT_NAME]

print("=" * 60)
print(f"RUNNING: {EXPERIMENT_NAME}")
print("=" * 60)

print("Configuration:")
print(config)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "electricity_cleaned.csv"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

print("\nDataset shape:", df.shape)


# ============================================================
# 2. DATE PROCESSING
# ============================================================

df["Dates"] = pd.to_datetime(df["Dates"])

df = df.sort_values("Dates").reset_index(drop=True)


# ============================================================
# 3. FEATURES
# ============================================================

features = [
    "Total Consumption",
    "DayOfWeek_sin",
    "DayOfWeek_cos",
    "DayOfYear_sin",
    "DayOfYear_cos",
    "Month_sin",
    "Month_cos",
    "lag_1",
    "lag_7",
    "lag_30",
    "lag_365",
    "rolling_mean_7",
    "rolling_std_7",
    "rolling_mean_30",
    "rolling_std_30"
]

target = "Total Consumption"

model_df = df[["Dates"] + features].copy()


# ============================================================
# 4. REMOVE MISSING VALUES
# ============================================================

model_df = model_df.dropna().reset_index(drop=True)

print("Data after removing missing values:", model_df.shape)


# ============================================================
# 5. CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT
# ============================================================

n = len(model_df)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

train_df = model_df.iloc[:train_end].copy()
val_df = model_df.iloc[train_end:val_end].copy()
test_df = model_df.iloc[val_end:].copy()

print("\nTrain:", train_df.shape)
print("Validation:", val_df.shape)
print("Test:", test_df.shape)


# ============================================================
# 6. SCALING
# ============================================================

scaler = MinMaxScaler()

train_scaled = scaler.fit_transform(
    train_df[features]
)

val_scaled = scaler.transform(
    val_df[features]
)

test_scaled = scaler.transform(
    test_df[features]
)


# ============================================================
# 7. CREATE SEQUENCES
# ============================================================

LOOKBACK = 30

target_index = features.index(target)


def create_sequences(data, lookback, target_index):

    X = []
    y = []

    for i in range(lookback, len(data)):

        X.append(
            data[i - lookback:i]
        )

        y.append(
            data[i, target_index]
        )

    return np.array(X), np.array(y)


X_train, y_train = create_sequences(
    train_scaled,
    LOOKBACK,
    target_index
)

X_val, y_val = create_sequences(
    val_scaled,
    LOOKBACK,
    target_index
)

X_test, y_test = create_sequences(
    test_scaled,
    LOOKBACK,
    target_index
)

print("\nSequence shapes:")
print("X_train:", X_train.shape)
print("X_val:", X_val.shape)
print("X_test:", X_test.shape)


# ============================================================
# 8. BUILD LSTM MODEL
# ============================================================

model = Sequential([

    LSTM(
        config["lstm_1"],
        return_sequences=True,
        input_shape=(
            X_train.shape[1],
            X_train.shape[2]
        )
    ),

    Dropout(config["dropout"]),

    LSTM(
        config["lstm_2"],
        return_sequences=False
    ),

    Dropout(config["dropout"]),

    Dense(
        16,
        activation="relu"
    ),

    Dense(1)
])


model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)


print("\nModel:")
model.summary()


# ============================================================
# 9. CALLBACKS
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    min_lr=0.000001
)


# ============================================================
# 10. TRAIN MODEL
# ============================================================

history = model.fit(

    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=100,
    batch_size=32,

    callbacks=[
        early_stopping,
        reduce_lr
    ],

    verbose=1
)


# ============================================================
# 11. PREDICTION
# ============================================================

y_pred_scaled = model.predict(
    X_test,
    verbose=0
)


# ============================================================
# 12. CONVERT PREDICTIONS BACK TO ORIGINAL SCALE
# ============================================================

def inverse_target(values):

    temp = np.zeros(
        (len(values), len(features))
    )

    temp[:, target_index] = values.flatten()

    return scaler.inverse_transform(
        temp
    )[:, target_index]


y_test_actual = inverse_target(y_test)

y_pred_actual = inverse_target(
    y_pred_scaled
)


# ============================================================
# 13. EVALUATION
# ============================================================

mae = mean_absolute_error(
    y_test_actual,
    y_pred_actual
)

rmse = np.sqrt(
    mean_squared_error(
        y_test_actual,
        y_pred_actual
    )
)

mape = np.mean(
    np.abs(
        (y_test_actual - y_pred_actual)
        / y_test_actual
    )
) * 100

r2 = r2_score(
    y_test_actual,
    y_pred_actual
)


# ============================================================
# 14. SAVE METRICS
# ============================================================

metrics = pd.DataFrame({

    "Experiment": [
        EXPERIMENT_NAME
    ],

    "LSTM_1": [
        config["lstm_1"]
    ],

    "LSTM_2": [
        config["lstm_2"]
    ],

    "Dropout": [
        config["dropout"]
    ],

    "Lookback": [
        LOOKBACK
    ],

    "MAE": [
        mae
    ],

    "RMSE": [
        rmse
    ],

    "MAPE": [
        mape
    ],

    "R2": [
        r2
    ]
})


metrics_path = os.path.join(
    RESULTS_DIR,
    f"{EXPERIMENT_NAME}_metrics.csv"
)

metrics.to_csv(
    metrics_path,
    index=False
)


# ============================================================
# 15. SAVE MODEL
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    f"{EXPERIMENT_NAME}.keras"
)

model.save(model_path)


# ============================================================
# 16. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

print(f"Experiment : {EXPERIMENT_NAME}")
print(f"MAE        : {mae:.2f}")
print(f"RMSE       : {rmse:.2f}")
print(f"MAPE       : {mape:.2f}%")
print(f"R2         : {r2:.4f}")

print("\nModel saved to:")
print(model_path)

print("\nMetrics saved to:")
print(metrics_path)

print("\nExperiment completed successfully.")