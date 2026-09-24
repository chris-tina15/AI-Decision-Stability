import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# 1. Load dataset
df = pd.read_csv("Data/bank-full.csv", sep=";")


# 2. Separate features and target
X = df.drop(columns=["y", "duration"])
y = df["y"]


# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# 4. Define feature groups
numerical_features = [
    "age",
    "balance",
    "day",
    "campaign",
    "pdays",
    "previous"
]

categorical_features = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome"
]


# 5. Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)


# 6. Fit only on training data
X_train_processed = preprocessor.fit_transform(X_train)


# 7. Transform test data
X_test_processed = preprocessor.transform(X_test)


# 8. Display results
print("Preprocessing completed successfully.")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])
print("Processed training shape:", X_train_processed.shape)
print("Processed testing shape:", X_test_processed.shape)