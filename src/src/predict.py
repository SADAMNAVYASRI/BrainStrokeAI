import numpy as np
import tensorflow as tf

from sklearn.preprocessing import StandardScaler


# Load trained model
model = tf.keras.models.load_model("../models/stroke_dnn_model.h5")


# User input
print("Enter Patient Details")

gender = int(input("Gender (Male=1 Female=0): "))
age = float(input("Age: "))
hypertension = int(input("Hypertension (0/1): "))
heart_disease = int(input("Heart Disease (0/1): "))
ever_married = int(input("Ever Married (Yes=1 No=0): "))
work_type = int(input("Work Type (Encoded value): "))
residence_type = int(input("Residence Type (Urban=1 Rural=0): "))
avg_glucose_level = float(input("Average Glucose Level: "))
bmi = float(input("BMI: "))
smoking_status = int(input("Smoking Status (Encoded value): "))


# Create input array
patient_data = np.array([[
    gender,
    age,
    hypertension,
    heart_disease,
    ever_married,
    work_type,
    residence_type,
    avg_glucose_level,
    bmi,
    smoking_status
]])


# Prediction
prediction = model.predict(patient_data)


if prediction[0][0] > 0.5:
    print("⚠️ Stroke Risk Detected")
else:
    print("✅ No Stroke Risk")
