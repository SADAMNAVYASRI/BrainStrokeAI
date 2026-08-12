# 🧠 AI Agent for Brain Stroke Prediction Using Electronic Health Records and Clinical Decision Support

An AI-based brain stroke risk prediction system that uses **Electronic Health Record (EHR) data**, **Deep Neural Networks (DNN)**, and a **Flask web application** to estimate stroke risk from user-provided health information.

The system processes clinical and demographic information, predicts the probability of stroke, and provides a simple risk result:

* 🔴 **Stroke Risk Detected**
* 🟢 **No Stroke Risk**

> **Note:** This project is developed for educational and research purposes.

---

## 📌 Project Overview

Brain stroke is a serious medical condition where early identification of risk factors can support better awareness and decision-making.

This project demonstrates how machine learning and deep learning can be applied to healthcare-related electronic health record data to estimate the likelihood of stroke.

The system takes several patient attributes as input, preprocesses the data, passes it through a trained Deep Neural Network, and converts the predicted probability into an understandable risk result.

---

## ✨ Key Features

* 🧠 Deep Neural Network based stroke prediction
* 📊 Stroke probability estimation
* 🎯 Threshold-based risk classification
* ⚖️ SMOTE-based class balancing
* 🔢 Categorical feature encoding
* 📏 StandardScaler feature normalization
* 🌐 Flask-based web application
* 👤 Male, Female, and Other gender support
* 🩸 Average glucose level reference information
* ⚖️ BMI reference information
* 💾 Saved trained model and preprocessing objects
* 📱 Simple user-friendly prediction interface

---

## 🏗️ System Workflow

```text
                  User Health Details
                         │
                         ▼
                Data Preprocessing
                         │
                         ▼
              Categorical Encoding
                         │
                         ▼
              Missing Value Handling
                         │
                         ▼
                 SMOTE Balancing
                         │
                         ▼
                 StandardScaler
                         │
                         ▼
              Deep Neural Network
                         │
                         ▼
                Stroke Probability
                         │
                         ▼
                Threshold Decision
                    /          \
                   /            \
                  ▼              ▼
        🔴 Stroke Risk      🟢 No Stroke Risk
```

---

## 📊 Dataset

The project uses the **Healthcare Stroke Prediction Dataset**.

The dataset contains **5,110 records** with demographic, lifestyle, and clinical information.

### Dataset Features

| Feature             | Description                               |
| ------------------- | ----------------------------------------- |
| `id`                | Unique patient identifier                 |
| `gender`            | Gender of the patient                     |
| `age`               | Age of the patient                        |
| `hypertension`      | Hypertension status                       |
| `heart_disease`     | Heart disease status                      |
| `ever_married`      | Whether the patient has ever been married |
| `work_type`         | Type of occupation                        |
| `Residence_type`    | Rural or Urban residence                  |
| `avg_glucose_level` | Average glucose level                     |
| `bmi`               | Body Mass Index                           |
| `smoking_status`    | Smoking category                          |
| `stroke`            | Target variable                           |

### Target Variable

```text
0 → No Stroke
1 → Stroke
```

---

## 🧹 Data Preprocessing

Several preprocessing techniques were applied before training the model.

### 1. Missing Value Handling

The original dataset contains missing values in the `bmi` column.

A total of **201 BMI values** were missing.

These missing values were replaced using **median imputation**.

```text
Missing BMI values
        ↓
Median Imputation
        ↓
Complete BMI Feature
```

---

### 2. Categorical Encoding

Categorical features were converted into numerical values so that they could be processed by the machine learning model.

### Gender Mapping

```text
Female = 0
Male   = 1
Other  = 2
```

### Ever Married Mapping

```text
No  = 0
Yes = 1
```

### Work Type Mapping

```text
Govt_job      = 0
Never_worked  = 1
Private       = 2
Self-employed = 3
children      = 4
```

### Residence Type Mapping

```text
Rural = 0
Urban = 1
```

### Smoking Status Mapping

```text
Unknown          = 0
formerly smoked  = 1
never smoked     = 2
smokes           = 3
```

---

## 🗑️ Feature Removal

The `id` column was removed because it is only an identifier and does not provide meaningful information for predicting stroke.

The final input contains **10 features**.

```text
gender
age
hypertension
heart_disease
ever_married
work_type
Residence_type
avg_glucose_level
bmi
smoking_status
```

---

## 🔀 Train-Test Split

The dataset was divided into training and testing sets.

```text
80% → Training Data
20% → Testing Data
```

A **stratified split** was used so that the distribution of the target classes was maintained between the training and testing datasets.

---

## ⚖️ Handling Class Imbalance with SMOTE

The original dataset contains significantly fewer stroke cases compared with non-stroke cases.

To address this class imbalance, **SMOTE (Synthetic Minority Over-sampling Technique)** was applied to the training data.

```text
Original Training Data
          ↓
        SMOTE
          ↓
Balanced Training Data
```

SMOTE was applied **only to the training data** to avoid data leakage into the test set.

---

## 📏 Feature Scaling

After balancing the training data, **StandardScaler** was used to normalize the feature values.

The scaler was fitted on the training data and then used to transform both training and testing data.

```text
Training Features
       ↓
StandardScaler
       ↓
Scaled Features
```

The trained scaler is saved and reused during prediction so that user input is processed in the same way as the training data.

---

# 🧠 Deep Neural Network Model

The prediction model was developed using **TensorFlow/Keras**.

## Model Architecture

```text
Input Layer
10 Features
      │
      ▼
Dense Layer
64 Neurons
ReLU Activation
      │
      ▼
Dropout
0.25
      │
      ▼
Dense Layer
32 Neurons
ReLU Activation
      │
      ▼
Dropout
0.15
      │
      ▼
Dense Layer
16 Neurons
ReLU Activation
      │
      ▼
Output Layer
1 Neuron
Sigmoid Activation
      │
      ▼
Stroke Probability
```

---

## ⚙️ Model Configuration

| Parameter          | Value                |
| ------------------ | -------------------- |
| Model              | Deep Neural Network  |
| Framework          | TensorFlow / Keras   |
| Optimizer          | Adam                 |
| Learning Rate      | 0.001                |
| Loss Function      | Binary Cross Entropy |
| Batch Size         | 32                   |
| Epochs             | 50                   |
| Hidden Activations | ReLU                 |
| Output Activation  | Sigmoid              |
| Input Features     | 10                   |

---

## 🎯 Prediction Logic

The DNN produces a probability between **0 and 1**.

The probability is compared with the saved prediction threshold.

```text
                 DNN Prediction
                       │
                       ▼
              Stroke Probability
                       │
                       ▼
              Compare with Threshold
                    /       \
                   /         \
                  ▼           ▼
          Probability ≥    Probability <
             Threshold       Threshold
                │               │
                ▼               ▼
       🔴 Stroke Risk       🟢 No Stroke
          Detected             Risk
```

The current trained model uses a threshold of approximately **0.50**.

---

# 📈 Model Performance

The final model was evaluated on the test dataset.

| Metric               |     Result |
| -------------------- | ---------: |
| Test Accuracy        | **84.64%** |
| Stroke Recall        |    **40%** |
| Prediction Threshold |   **0.50** |

### Confusion Matrix

```text
[[845 127]
 [ 30  20]]
```

The model achieves an overall test accuracy above **80%** while identifying a portion of the minority stroke cases.

---

## 🌐 Flask Web Application

A **Flask** web application was developed to provide a simple interface for users.

The user can enter:

* Gender
* Age
* Hypertension
* Heart Disease
* Ever Married
* Work Type
* Residence Type
* Average Glucose Level
* BMI
* Smoking Status

The application converts the entered values into the same numerical format used during model training.

---

## 🖥️ Application Prediction Flow

```text
User Input
    │
    ▼
Flask Form
    │
    ▼
Categorical Value Mapping
    │
    ▼
NumPy Input Array
    │
    ▼
Saved StandardScaler
    │
    ▼
Trained DNN Model
    │
    ▼
Stroke Probability
    │
    ▼
Saved Threshold
    │
    ├───────────────┐
    ▼               ▼
Stroke Risk     No Stroke Risk
Detected
```

---

## 🩸 Average Glucose Level Reference

The web application displays general reference information for glucose levels.

```text
< 100 mg/dL
→ Normal fasting range

100–125 mg/dL
→ Prediabetes range

≥ 126 mg/dL
→ Diabetes range
```

These values are provided only as general educational reference.

---

## ⚖️ BMI Reference

The web application also displays general BMI categories.

```text
< 18.5
→ Underweight

18.5–24.9
→ Normal

25–29.9
→ Overweight

≥ 30
→ Obesity
```

These categories are provided for general educational reference and should not be interpreted as a medical diagnosis.

---

# 📂 Project Structure

```text
BrainStrokeAI/
│
├── app.py
├── dnn_model.py
├── preprocessing.py
├── predict.py
├── train.py
├── requirements.txt
├── README.md
│
├── dataset/
│   └── healthcare-dataset-stroke-data.csv
│
├── models/
│   ├── stroke_dnn_model.h5
│   ├── stroke_scaler.pkl
│   └── stroke_threshold.pkl
│
├── templates/
│   └── index.html
│
└── static/
    └── style.css
```

### Files Description

| File / Folder      | Purpose                                     |
| ------------------ | ------------------------------------------- |
| `app.py`           | Flask web application and prediction logic  |
| `dnn_model.py`     | DNN model creation, training and evaluation |
| `preprocessing.py` | Data loading and preprocessing              |
| `predict.py`       | Prediction-related functionality            |
| `train.py`         | Training script                             |
| `dataset/`         | Dataset files                               |
| `models/`          | Trained model, scaler and threshold         |
| `templates/`       | HTML web interface                          |
| `static/`          | CSS styling                                 |
| `requirements.txt` | Python dependencies                         |
| `README.md`        | Project documentation                       |

> The virtual environment and Python cache files are not included in the repository.

---

# 🛠️ Technologies Used

## Programming Language

* Python

## Deep Learning

* TensorFlow
* Keras

## Machine Learning

* Scikit-learn
* imbalanced-learn
* SMOTE

## Data Processing

* Pandas
* NumPy

## Web Development

* Flask
* HTML
* CSS

## Model Storage

* HDF5 / Keras model
* Pickle

## Version Control

* Git
* GitHub

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/SADAMNAVYASRI/BrainStrokeAI.git
```

## 2. Navigate to the Project

```bash
cd BrainStrokeAI
```

## 3. Create a Virtual Environment

```bash
python -m venv venv
```

## 4. Activate the Virtual Environment

For Windows:

```bash
venv\Scripts\activate
```

## 5. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Application

Start the Flask application:

```bash
python app.py
```

The application will run on the local Flask server.

Open the address shown in the terminal, usually:

```text
http://127.0.0.1:5000/
```

Then enter the required health details and click:

**🔮 Predict Stroke Risk**

---

# 🧪 Example Prediction

Example input:

```text
Gender              : Male
Age                 : 80
Hypertension        : Yes
Heart Disease       : Yes
Ever Married        : Yes
Work Type           : Private
Residence Type      : Urban
Average Glucose     : 380
BMI                 : 45
Smoking Status      : Smokes
```

The model processes these details and returns a stroke probability followed by a risk classification.

Example:

```text
Stroke Probability: 72.91%

🔴 Stroke Risk Detected
```

> The prediction shown by the model is an example and should not be considered a medical diagnosis.

---

# 🔮 Future Enhancements

The project can be further improved by:

* Improving minority-class stroke recall
* Comparing multiple deep learning architectures
* Adding TabNet and FT-Transformer comparison
* Adding Explainable AI using SHAP
* Adding interactive charts
* Improving the user interface
* Adding prediction history
* Adding REST API support
* Deploying the Flask application to the cloud
* Adding secure user authentication
* Generating downloadable prediction reports

---

# ⚠️ Medical Disclaimer

This project is intended **only for educational and research purposes**.

The prediction generated by this machine learning system is based on patterns learned from the dataset and **must not be considered a medical diagnosis, medical advice, or a substitute for consultation with a qualified healthcare professional**.

---

# 👩‍💻 Developed By

**Sadam Navyasri**

**B.Tech – Data Science**

---

# 📜 License

This project is developed for **educational and research purposes**.

The project demonstrates the application of machine learning and deep learning techniques to healthcare-related data.
