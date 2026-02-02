# =========================================================
# ADVANCED TIME SERIES FORECASTING WITH EXPLAINABILITY
# LSTM + SARIMAX + SHAP (COMPLETE ONE-SET CODE)
# =========================================================

# -----------------------------
# 1. IMPORT LIBRARIES
# -----------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from statsmodels.tsa.statespace.sarimax import SARIMAX
import shap


# -----------------------------
# 2. LOAD DATASET (Keras Weather)
# -----------------------------
zip_path = tf.keras.utils.get_file(
    origin="https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip",
    fname="jena_climate_2009_2016.csv.zip",
    extract=True
)

csv_path = zip_path.replace(".zip", "")
df = pd.read_csv(csv_path)


# -----------------------------
# 3. DATA CLEANING
# -----------------------------
df = df.drop(columns=["Date Time"])
df.fillna(method="ffill", inplace=True)


# -----------------------------
# 4. FEATURE SCALING
# -----------------------------
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)


# -----------------------------
# 5. SEQUENCE CREATION
# -----------------------------
def create_sequences(data, target_col, window_size=48):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i : i + window_size])
        y.append(data[i + window_size, target_col])
    return np.array(X), np.array(y)


TARGET_COL  = df.columns.get_loc("T (degC)")
WINDOW_SIZE = 48

X, y = create_sequences(scaled_data, TARGET_COL, WINDOW_SIZE)


# -----------------------------
# 6. WALK-FORWARD SPLIT
# -----------------------------
train_size = int(0.7 * len(X))
val_size   = int(0.15 * len(X))

X_train = X[:train_size]
y_train = y[:train_size]

X_val = X[train_size : train_size + val_size]
y_val = y[train_size : train_size + val_size]

X_test = X[train_size + val_size :]
y_test = y[train_size + val_size :]


# -----------------------------
# 7. BUILD LSTM MODEL
# -----------------------------
model = Sequential([
    LSTM(
        64,
        return_sequences=True,
        input_shape=(X.shape[1], X.shape[2])
    ),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="mse"
)

model.summary()


# -----------------------------
# 8. TRAIN MODEL
# -----------------------------
callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True),
    ReduceLROnPlateau(patience=5, factor=0.5)
]

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=64,
    callbacks=callbacks,
    verbose=1
)


# -----------------------------
# 9. LOSS CURVE
# -----------------------------
plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.title("LSTM Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.show()


# -----------------------------
# 10. LSTM EVALUATION
# -----------------------------
lstm_preds = model.predict(X_test).flatten()

mae_lstm  = mean_absolute_error(y_test, lstm_preds)
rmse_lstm = np.sqrt(mean_squared_error(y_test, lstm_preds))
mape_lstm = np.mean(
    np.abs((y_test - lstm_preds) / y_test)
) * 100

print("LSTM PERFORMANCE")
print("MAE :", mae_lstm)
print("RMSE:", rmse_lstm)
print("MAPE:", mape_lstm)


# -----------------------------
# 11. ACTUAL vs PREDICTED PLOT
# -----------------------------
plt.figure(figsize=(10, 5))
plt.plot(y_test[:200], label="Actual")
plt.plot(lstm_preds[:200], label="Predicted")
plt.xlabel("Time Steps")
plt.ylabel("Scaled Temperature")
plt.title("Actual vs Predicted Temperature (LSTM)")
plt.legend()
plt.grid(True)
plt.show()


# -----------------------------
# 12. STATISTICAL BASELINE (SARIMAX)
# -----------------------------
target_series = df["T (degC)"]

train_series = target_series[: int(0.85 * len(target_series))]
test_series  = target_series[int(0.85 * len(target_series)) :]

sarimax_model = SARIMAX(
    train_series,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 24)
)

sarimax_fit  = sarimax_model.fit(disp=False)
sarimax_preds = sarimax_fit.forecast(len(test_series))


# -----------------------------
# 13. SARIMAX EVALUATION
# -----------------------------
mae_sarimax  = mean_absolute_error(test_series, sarimax_preds)
rmse_sarimax = np.sqrt(mean_squared_error(test_series, sarimax_preds))
mape_sarimax = np.mean(
    np.abs((test_series - sarimax_preds) / test_series)
) * 100

print("\nSARIMAX PERFORMANCE")
print("MAE :", mae_sarimax)
print("RMSE:", rmse_sarimax)
print("MAPE:", mape_sarimax)


# -----------------------------
# 14. COMPARISON TABLE
# -----------------------------
comparison_df = pd.DataFrame({
    "Model": ["LSTM", "SARIMAX"],
    "MAE":   [mae_lstm, mae_sarimax],
    "RMSE":  [rmse_lstm, rmse_sarimax],
    "MAPE":  [mape_lstm, mape_sarimax]
})

comparison_df


# -----------------------------
# 15. SHAP EXPLAINABILITY
# -----------------------------
explainer = shap.GradientExplainer(
    model,
    X_train[:100]
)

shap_values = explainer.shap_values(
    X_test[:50]
)


# -----------------------------
# 16. SHAP SUMMARY PLOT
# -----------------------------
shap.summary_plot(
    shap_values[0],
    X_test[:50],
    feature_names=df.columns
)


# -----------------------------
# 17. SHAP FEATURE IMPORTANCE (BAR)
# -----------------------------
shap.summary_plot(
    shap_values[0],
    X_test[:50],
    feature_names=df.columns,
    plot_type="bar"
)
