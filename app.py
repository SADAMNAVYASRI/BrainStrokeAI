from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import pickle

app = Flask(__name__)

model = tf.keras.models.load_model(
    "models/stroke_dnn_model.h5"
)

with open("models/stroke_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("models/stroke_threshold.pkl", "rb") as f:
    threshold = pickle.load(f)

print("Model Loaded Successfully!")
print("Threshold:", threshold)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    gender_value = request.form["gender"]
    if gender_value == "Female":
        gender = 0
    elif gender_value == "Male":
        gender = 1
    elif gender_value == "Other":
        gender = 2

     


    age = float(request.form["age"])

    hypertension = float(request.form["hypertension"])

    heart_disease = float(request.form["heart_disease"])


    married_value = request.form["ever_married"]

    if married_value == "No":
        ever_married = 0
    else:
        ever_married = 1

    work_value = request.form["work_type"]
    if work_value == "0":
        work_type = 0
    elif work_value == "1":
        work_type = 1
    elif work_value == "2":
        work_type = 2
    elif work_value == "3":
       work_type = 3
    else:
       work_type = 4


    residence_value = request.form["residence_type"]

    if residence_value == "Rural":
        residence_type = 0
    else:
        residence_type = 1


    avg_glucose_level = float(
        request.form["avg_glucose_level"]
    )

    bmi = float(
        request.form["bmi"]
    )


    smoking_value = request.form["smoking_status"]

    if smoking_value == "0":
        smoking_status = 0
    elif smoking_value == "1":
        smoking_status = 1
    elif smoking_value == "2":
        smoking_status = 2
    else:
        smoking_status = 3


    data = np.array([[
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
    ]], dtype=np.float32)


    print()
    print("INPUT DATA:")
    print(data)


    data_scaled = scaler.transform(data)


    prediction = model.predict(
        data_scaled,
        verbose=0
    )


    probability = float(prediction[0][0])


    print("STROKE PROBABILITY:", probability)
    print("THRESHOLD:", threshold)


    if probability >= threshold:
        result = "Stroke Risk Detected"
    else:
        result = "No Stroke Risk"
    return render_template(
        "index.html",
        prediction=result,
        probability=round(probability * 100, 2),
        form=request.form.to_dict()
    )


if __name__ == "__main__":
    app.run(debug=True)