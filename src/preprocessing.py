import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

print("🔥 NEW PREPROCESSING FILE IS RUNNING 🔥")

# Load dataset
df = pd.read_csv("dataset/healthcare-dataset-stroke-data.csv")

print("\nMissing Values Before:")
print(df.isnull().sum())

# Fill BMI missing values
median_bmi = df["bmi"].median()
df["bmi"] = df["bmi"].fillna(median_bmi)

print("\nMissing Values After:")
print(df.isnull().sum())


# ==========================================
# CATEGORY ENCODING
# ==========================================

categorical_columns = [
    "gender",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status"
]

print("\n==============================")
print("CATEGORY MAPPINGS")
print("==============================")

for col in categorical_columns:

    le = LabelEncoder()

    df[col] = le.fit_transform(df[col].astype(str))

    print("\n" + col + ":")

    for number, value in enumerate(le.classes_):
        print(value, "=", number)


# ==========================================
# FEATURES AND TARGET
# ==========================================

X = df.drop(["id", "stroke"], axis=1)
y = df["stroke"]


# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================
# SMOTE
# ==========================================

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)


print("\n==============================")
print("PREPROCESSING COMPLETED")
print("==============================")

print("Training Data Shape:", X_train_smote.shape)
print("Testing Data Shape:", X_test.shape)
