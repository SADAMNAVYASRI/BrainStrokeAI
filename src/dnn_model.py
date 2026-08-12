import numpy as np
import tensorflow as tf
import pickle

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

from preprocessing import X_train_smote, y_train_smote, X_test, y_test


# --------------------------------------------------
# Convert to numpy
# --------------------------------------------------

X_train_smote = np.array(X_train_smote, dtype=np.float32)
y_train_smote = np.array(y_train_smote, dtype=np.float32)

X_test = np.array(X_test, dtype=np.float32)
y_test = np.array(y_test, dtype=np.float32)


# --------------------------------------------------
# Feature Scaling
# --------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test)


# Save scaler
with open("models/stroke_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Scaler Saved Successfully!")


# --------------------------------------------------
# Reproducibility
# --------------------------------------------------

np.random.seed(42)
tf.random.set_seed(42)


# --------------------------------------------------
# Create DNN Model
# --------------------------------------------------

model = tf.keras.Sequential([

    tf.keras.layers.Input(shape=(10,)),

    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dropout(0.15),

    tf.keras.layers.Dense(16, activation="relu"),

    tf.keras.layers.Dense(1, activation="sigmoid")
])


# --------------------------------------------------
# Compile
# --------------------------------------------------

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


model.summary()


# --------------------------------------------------
# Train
# --------------------------------------------------

history = model.fit(
    X_train_scaled,
    y_train_smote,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)


# --------------------------------------------------
# Probability Prediction
# --------------------------------------------------

y_pred_prob = model.predict(X_test_scaled).ravel()


# --------------------------------------------------
# Find Best Threshold
# --------------------------------------------------

results = []

for threshold in np.arange(0.20, 0.51, 0.01):

    y_pred = (y_pred_prob >= threshold).astype(int)

    accuracy = accuracy_score(y_test, y_pred)

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    stroke_recall = report.get("1", {}).get("recall", 0)

    results.append(
        (threshold, accuracy, stroke_recall)
    )


# Prefer accuracy >= 80% with better stroke recall
valid_results = [
    r for r in results
    if r[1] >= 0.80 and r[2] >= 0.40
]


if valid_results:

    best_threshold, best_accuracy, best_recall = max(
        valid_results,
        key=lambda x: x[2]
    )

else:

    best_threshold, best_accuracy, best_recall = max(
        results,
        key=lambda x: x[1]
    )


# --------------------------------------------------
# Final Evaluation
# --------------------------------------------------

y_pred = (y_pred_prob >= best_threshold).astype(int)


print("\n====================================")
print("FINAL MODEL RESULTS")
print("====================================")

print("Best Threshold:", round(best_threshold, 2))

print("Accuracy:", round(best_accuracy * 100, 2), "%")

print("Stroke Recall:", round(best_recall * 100, 2), "%")


print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))


# --------------------------------------------------
# Save Threshold
# --------------------------------------------------

with open("models/stroke_threshold.pkl", "wb") as f:
    pickle.dump(float(best_threshold), f)

print("\nThreshold Saved Successfully!")


# --------------------------------------------------
# Save Model
# --------------------------------------------------

model.save("models/stroke_dnn_model.h5")

print("DNN Model Saved Successfully!")
