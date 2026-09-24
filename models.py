import pandas as pd
from scipy.stats import binomtest
from scipy.stats import beta
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression


# Load dataset
df = pd.read_csv("Data/bank-full.csv", sep=";")


# Separate features and target
X = df.drop(columns=["y", "duration"])
y = df["y"]


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Feature groups
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


# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)


# Fit preprocessing on training data
X_train_processed = preprocessor.fit_transform(X_train)

# Transform testing data
X_test_processed = preprocessor.transform(X_test)


# Create Logistic Regression model
logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)


# Train the model
logistic_model.fit(X_train_processed, y_train)


print("Logistic Regression training completed successfully.")
print("Training samples:", X_train_processed.shape[0])
print("Training features:", X_train_processed.shape[1])


# Generate predictions on test data
y_pred = logistic_model.predict(X_test_processed)

print("Test predictions generated successfully.")
print("Number of predictions:", len(y_pred))

# Generate predictions on test data
y_pred = logistic_model.predict(X_test_processed)

print("Test predictions generated successfully.")
print("Number of predictions:", len(y_pred))

# Evaluate Logistic Regression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label="yes")
recall = recall_score(y_test, y_pred, pos_label="yes")
f1 = f1_score(y_test, y_pred, pos_label="yes")

print()
print("Logistic Regression Baseline Performance")
print("-----------------------------------------")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

from sklearn.ensemble import RandomForestClassifier


# Random Forest model
random_forest_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

# Train Random Forest
random_forest_model.fit(X_train_processed, y_train)

print()
print("Random Forest training completed successfully.")

# Generate Random Forest predictions
rf_pred = random_forest_model.predict(X_test_processed)

print("Random Forest predictions generated successfully.")
print("Number of predictions:", len(rf_pred))

# Evaluate Random Forest
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred, pos_label="yes")
rf_recall = recall_score(y_test, rf_pred, pos_label="yes")
rf_f1 = f1_score(y_test, rf_pred, pos_label="yes")

print()
print("Random Forest Baseline Performance")
print("-----------------------------------")
print(f"Accuracy : {rf_accuracy:.4f}")
print(f"Precision: {rf_precision:.4f}")
print(f"Recall   : {rf_recall:.4f}")
print(f"F1-score : {rf_f1:.4f}")

from xgboost import XGBClassifier


# XGBoost model
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss"
)


# Train XGBoost
xgb_model.fit(X_train_processed, y_train.map({"no": 0, "yes": 1}))

print()
print("XGBoost training completed successfully.")

# Generate XGBoost predictions
xgb_pred = xgb_model.predict(X_test_processed)

print("XGBoost predictions generated successfully.")
print("Number of predictions:", len(xgb_pred))

# Convert test labels to 0/1 for XGBoost evaluation
y_test_xgb = y_test.map({"no": 0, "yes": 1})


# Evaluate XGBoost
xgb_accuracy = accuracy_score(y_test_xgb, xgb_pred)
xgb_precision = precision_score(y_test_xgb, xgb_pred, pos_label=1)
xgb_recall = recall_score(y_test_xgb, xgb_pred, pos_label=1)
xgb_f1 = f1_score(y_test_xgb, xgb_pred, pos_label=1)

print()
print("XGBoost Baseline Performance")
print("-----------------------------")
print(f"Accuracy : {xgb_accuracy:.4f}")
print(f"Precision: {xgb_precision:.4f}")
print(f"Recall   : {xgb_recall:.4f}")
print(f"F1-score : {xgb_f1:.4f}")

from sklearn.metrics import confusion_matrix

print()
print("Baseline Confusion Matrices")
print("----------------------------")

print("Logistic Regression:")
print(confusion_matrix(y_test, y_pred))

print()
print("Random Forest:")
print(confusion_matrix(y_test, rf_pred))

print()
print("XGBoost:")
print(confusion_matrix(y_test_xgb, xgb_pred))

perturbation_features = [
    "age",
    "balance",
    "campaign",
    "previous"
]

print()
print("Selected Perturbation Features")
print("-------------------------------")
print(perturbation_features)

def create_campaign_perturbation(data, change):
    perturbed_data = data.copy()

    perturbed_data["campaign"] = (
        perturbed_data["campaign"] + change
    ).clip(lower=1)

    return perturbed_data

test_campaign_data = create_campaign_perturbation(X_test, 1)

print()
print("Campaign Perturbation Test")
print("--------------------------")
print("Original campaign values:")
print(X_test["campaign"].head().tolist())

print("Perturbed campaign values:")
print(test_campaign_data["campaign"].head().tolist())

def create_previous_perturbation(data, change):
    perturbed_data = data.copy()

    perturbed_data["previous"] = (
        perturbed_data["previous"] + change
    ).clip(lower=0)

    return perturbed_data

test_previous_data = create_previous_perturbation(X_test, 1)

print()
print("Previous Perturbation Test")
print("--------------------------")
print("Original previous values:")
print(X_test["previous"].head().tolist())

print("Perturbed previous values:")
print(test_previous_data["previous"].head().tolist())


def create_age_perturbation(data, change):
    perturbed_data = data.copy()

    perturbed_data["age"] = (
        perturbed_data["age"] + change
    ).clip(18, 95)

    return perturbed_data


test_age_data = create_age_perturbation(X_test, 1)

# Logistic Regression - Age Perturbation Test

age_perturbed_processed = preprocessor.transform(test_age_data)

logistic_age_pred = logistic_model.predict(age_perturbed_processed)

age_prediction_changes = (y_pred != logistic_age_pred).sum()

print()
print("Logistic Regression - Age Stability Test")
print("----------------------------------------")
print("Changed predictions:", age_prediction_changes)
print("Total predictions:", len(y_pred))

def create_balance_perturbation(data, percentage):
    perturbed_data = data.copy()

    perturbed_data["balance"] = (
        perturbed_data["balance"] * (1 + percentage / 100)
    )

    return perturbed_data

test_balance_data = create_balance_perturbation(X_test, 1)
# Logistic Regression - Balance Perturbation Test

balance_perturbed_processed = preprocessor.transform(test_balance_data)

logistic_balance_pred = logistic_model.predict(balance_perturbed_processed)

balance_prediction_changes = (
    y_pred != logistic_balance_pred
).sum()

print()
print("Logistic Regression - Balance Stability Test")
print("---------------------------------------------")
print("Changed predictions:", balance_prediction_changes)
print("Total predictions:", len(y_pred))

# Logistic Regression - Campaign Perturbation Test

campaign_perturbed_processed = preprocessor.transform(test_campaign_data)

logistic_campaign_pred = logistic_model.predict(
    campaign_perturbed_processed
)

campaign_prediction_changes = (
    y_pred != logistic_campaign_pred
).sum()

print()
print("Logistic Regression - Campaign Stability Test")
print("----------------------------------------------")
print("Changed predictions:", campaign_prediction_changes)
print("Total predictions:", len(y_pred))

# Logistic Regression - Previous Perturbation Test

previous_perturbed_processed = preprocessor.transform(test_previous_data)

logistic_previous_pred = logistic_model.predict(
    previous_perturbed_processed
)

previous_prediction_changes = (
    y_pred != logistic_previous_pred
).sum()

print()
print("Logistic Regression - Previous Stability Test")
print("----------------------------------------------")
print("Changed predictions:", previous_prediction_changes)
print("Total predictions:", len(y_pred))

def calculate_stability(original_predictions, perturbed_predictions):
    changed = (original_predictions != perturbed_predictions).sum()
    total = len(original_predictions)

    flip_rate = changed / total
    stability = 1 - flip_rate

    return changed, flip_rate, stability
changed, flip_rate, stability = calculate_stability(
    y_pred,
    logistic_campaign_pred
)

print()
print("Stability Calculation Test")
print("--------------------------")
print("Changed predictions:", changed)
print(f"Flip rate: {flip_rate:.6f}")
print(f"Stability: {stability:.6f}")

# Random Forest - Campaign Stability Test

rf_campaign_pred = random_forest_model.predict(
    campaign_perturbed_processed
)

rf_changed, rf_flip_rate, rf_stability = calculate_stability(
    rf_pred,
    rf_campaign_pred
)

print()
print("Random Forest - Campaign Stability Test")
print("----------------------------------------")
print("Changed predictions:", rf_changed)
print(f"Flip rate: {rf_flip_rate:.6f}")
print(f"Stability: {rf_stability:.6f}")

# XGBoost - Campaign Stability Test

xgb_campaign_pred = xgb_model.predict(
    campaign_perturbed_processed
)

xgb_changed, xgb_flip_rate, xgb_stability = calculate_stability(
    xgb_pred,
    xgb_campaign_pred
)

print()
print("XGBoost - Campaign Stability Test")
print("----------------------------------")
print("Changed predictions:", xgb_changed)
print(f"Flip rate: {xgb_flip_rate:.6f}")
print(f"Stability: {xgb_stability:.6f}")

# ============================================================
# SYSTEMATIC PERTURBATION EXPERIMENT
# ============================================================

def run_perturbation_experiment(
    feature,
    perturbation_type,
    perturbation_levels
):
    results = []

    for level in perturbation_levels:

        # Create perturbed copy from ORIGINAL test data
        perturbed_data = X_test.copy()

        # Apply perturbation
        if perturbation_type == "percentage":
            perturbed_data[feature] = (
                perturbed_data[feature] * (1 + level / 100)
            )

        elif perturbation_type == "integer":
            perturbed_data[feature] = (
                perturbed_data[feature] + level
            )

        # Apply required boundaries
        if feature == "age":
            perturbed_data[feature] = perturbed_data[feature].clip(18, 95)

        elif feature == "campaign":
            perturbed_data[feature] = perturbed_data[feature].clip(lower=1)

        elif feature == "previous":
            perturbed_data[feature] = perturbed_data[feature].clip(lower=0)

        # Preprocess using the SAME fitted preprocessor
        perturbed_processed = preprocessor.transform(
            perturbed_data
        )

        # Predictions from all three models
        logistic_perturbed_pred = logistic_model.predict(
            perturbed_processed
        )

        rf_perturbed_pred = random_forest_model.predict(
            perturbed_processed
        )

        xgb_perturbed_pred = xgb_model.predict(
            perturbed_processed
        )

        # Calculate stability for each model
        log_changed, log_flip_rate, log_stability = calculate_stability(
            y_pred,
            logistic_perturbed_pred
        )

        rf_changed, rf_flip_rate, rf_stability = calculate_stability(
            rf_pred,
            rf_perturbed_pred
        )

        xgb_changed, xgb_flip_rate, xgb_stability = calculate_stability(
            xgb_pred,
            xgb_perturbed_pred
        )
        # Convert Logistic Regression and Random Forest predictions to numeric
        logistic_original_numeric = (y_pred == "yes").astype(int)
        logistic_perturbed_numeric = (logistic_perturbed_pred == "yes").astype(int)

        rf_original_numeric = (rf_pred == "yes").astype(int)
        rf_perturbed_numeric = (rf_perturbed_pred == "yes").astype(int)

        # Direction of prediction changes
        log_0_to_1 = (
        (logistic_original_numeric == 0) &
        (logistic_perturbed_numeric == 1)
        ).sum()

        log_1_to_0 = (
        (logistic_original_numeric == 1) &
        (logistic_perturbed_numeric == 0)
        ).sum()

        rf_0_to_1 = (
        (rf_original_numeric == 0) &
        (rf_perturbed_numeric == 1)
        ).sum()

        rf_1_to_0 = (
        (rf_original_numeric == 1) &
        (rf_perturbed_numeric == 0)
        ).sum()

        xgb_0_to_1 = (
        (xgb_pred == 0) &
        (xgb_perturbed_pred == 1)
        ).sum()

        xgb_1_to_0 = (
        (xgb_pred == 1) &
        (xgb_perturbed_pred == 0)
        ).sum()

        # Store results
        results.append({
            "Feature": feature,
            "Perturbation": level,

            "Logistic_Changed": log_changed,
            "Logistic_Flip_Rate": log_flip_rate,
            "Logistic_Stability": log_stability,

            "RF_Changed": rf_changed,
            "RF_Flip_Rate": rf_flip_rate,
            "RF_Stability": rf_stability,

            "XGB_Changed": xgb_changed,
            "XGB_Flip_Rate": xgb_flip_rate,
            "XGB_Stability": xgb_stability,

            "Logistic_0_to_1": log_0_to_1,
            "Logistic_1_to_0": log_1_to_0,

            "RF_0_to_1": rf_0_to_1,
            "RF_1_to_0": rf_1_to_0,

            "XGB_0_to_1": xgb_0_to_1,
            "XGB_1_to_0": xgb_1_to_0
        })

    return pd.DataFrame(results)

# ============================================================
# FIRST SYSTEMATIC EXPERIMENT
# Campaign: +1
# ============================================================

campaign_results = run_perturbation_experiment(
    feature="campaign",
    perturbation_type="integer",
    perturbation_levels=[-2, -1, 1, 2, 5]
)

print()
print("Systematic Campaign Perturbation Results")
print("----------------------------------------")
print(campaign_results)
# ============================================================
# SYSTEMATIC BALANCE PERTURBATION EXPERIMENT
# ============================================================

balance_results = run_perturbation_experiment(
    feature="balance",
    perturbation_type="percentage",
    perturbation_levels=[-5, -2, -1, 1, 2, 5]
)

print()
print("Systematic Balance Perturbation Results")
print("---------------------------------------")
print(balance_results)
# ============================================================
# SYSTEMATIC AGE PERTURBATION EXPERIMENT
# ============================================================

age_results = run_perturbation_experiment(
    feature="age",
    perturbation_type="integer",
    perturbation_levels=[-5, -2, -1, 1, 2, 5]
)

print()
print("Systematic Age Perturbation Results")
print("-----------------------------------")
print(age_results)

# ============================================================
# SYSTEMATIC PREVIOUS PERTURBATION EXPERIMENT
# ============================================================

previous_results = run_perturbation_experiment(
    feature="previous",
    perturbation_type="integer",
    perturbation_levels=[1, 2, 5]
)

print()
print("Systematic Previous Perturbation Results")
print("----------------------------------------")
print(previous_results)
# ============================================================
# COMBINE ALL PERTURBATION RESULTS
# ============================================================

all_results = pd.concat(
    [
        campaign_results,
        balance_results,
        age_results,
        previous_results
    ],
    ignore_index=True
)

print()
print("Combined Perturbation Experiment Results")
print("-----------------------------------------")
print(all_results)

# Save results
all_results.to_csv(
    "perturbation_results.csv",
    index=False
)

print()
print("Results saved successfully.")
print("File: perturbation_results.csv")
# ============================================================
# FEATURE-LEVEL SENSITIVITY SUMMARY
# ============================================================

sensitivity_summary = all_results.groupby("Feature").agg(
    Logistic_Avg_Flip_Rate=("Logistic_Flip_Rate", "mean"),
    RF_Avg_Flip_Rate=("RF_Flip_Rate", "mean"),
    XGB_Avg_Flip_Rate=("XGB_Flip_Rate", "mean")
).reset_index()

print()
print("Feature-Level Sensitivity Summary")
print("---------------------------------")
print(sensitivity_summary)

sensitivity_summary.to_csv(
    "feature_sensitivity_summary.csv",
    index=False
)

print()
print("Sensitivity summary saved successfully.")
print("File: feature_sensitivity_summary.csv")
# ============================================================
# FEATURE SENSITIVITY VISUALIZATION
# ============================================================

import matplotlib.pyplot as plt

sensitivity_plot = sensitivity_summary.set_index("Feature")

sensitivity_plot[
    [
        "Logistic_Avg_Flip_Rate",
        "RF_Avg_Flip_Rate",
        "XGB_Avg_Flip_Rate"
    ]
].mul(100).plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Average Prediction Flip Rate by Feature")
plt.xlabel("Feature")
plt.ylabel("Average Flip Rate (%)")
plt.xticks(rotation=0)
plt.legend(
    ["Logistic Regression", "Random Forest", "XGBoost"]
)
plt.tight_layout()

plt.savefig(
    "feature_sensitivity_comparison.png",
    dpi=300
)

plt.show()

print()
print("Feature sensitivity chart saved successfully.")
print("File: feature_sensitivity_comparison.png")

print()
print("Result columns:")
print(all_results.columns.tolist())
# ============================================================
# M7 - STATISTICAL VALIDATION
# McNemar's Exact Test
# ============================================================

def mcnemar_exact_test(changed_0_to_1, changed_1_to_0):
    """
    Exact McNemar test using the discordant prediction pairs.

    0 -> 1 : prediction changed from negative to positive
    1 -> 0 : prediction changed from positive to negative
    """

    b = int(changed_0_to_1)
    c = int(changed_1_to_0)

    discordant = b + c

    if discordant == 0:
        return 1.0

    result = binomtest(
        k=min(b, c),
        n=discordant,
        p=0.5,
        alternative="two-sided"
    )

    return result.pvalue


mcnemar_results = []

for _, row in all_results.iterrows():

    logistic_p = mcnemar_exact_test(
        row["Logistic_0_to_1"],
        row["Logistic_1_to_0"]
    )

    rf_p = mcnemar_exact_test(
        row["RF_0_to_1"],
        row["RF_1_to_0"]
    )

    xgb_p = mcnemar_exact_test(
        row["XGB_0_to_1"],
        row["XGB_1_to_0"]
    )

    mcnemar_results.append({
        "Feature": row["Feature"],
        "Perturbation": row["Perturbation"],

        "Logistic_McNemar_p": logistic_p,
        "RF_McNemar_p": rf_p,
        "XGB_McNemar_p": xgb_p
    })


mcnemar_results = pd.DataFrame(mcnemar_results)

print()
print("McNemar's Exact Test Results")
print("----------------------------")
print(mcnemar_results)

mcnemar_results.to_csv(
    "mcnemar_results.csv",
    index=False
)

print()
print("McNemar results saved successfully.")
print("File: mcnemar_results.csv")
# ============================================================
# 95% CONFIDENCE INTERVALS FOR FLIP RATE
# ============================================================

def calculate_flip_rate_ci(changed, total, confidence=0.95):

    if changed == 0:
        lower = 0.0
    else:
        lower = beta.ppf(
            (1 - confidence) / 2,
            changed,
            total - changed + 1
        )

    if changed == total:
        upper = 1.0
    else:
        upper = beta.ppf(
            1 - (1 - confidence) / 2,
            changed + 1,
            total - changed
        )

    return lower, upper


confidence_results = []

total_predictions = len(y_pred)

for _, row in all_results.iterrows():

    log_lower, log_upper = calculate_flip_rate_ci(
        int(row["Logistic_Changed"]),
        total_predictions
    )

    rf_lower, rf_upper = calculate_flip_rate_ci(
        int(row["RF_Changed"]),
        total_predictions
    )

    xgb_lower, xgb_upper = calculate_flip_rate_ci(
        int(row["XGB_Changed"]),
        total_predictions
    )

    confidence_results.append({
        "Feature": row["Feature"],
        "Perturbation": row["Perturbation"],

        "Logistic_Flip_Rate": row["Logistic_Flip_Rate"],
        "Logistic_CI_Lower": log_lower,
        "Logistic_CI_Upper": log_upper,

        "RF_Flip_Rate": row["RF_Flip_Rate"],
        "RF_CI_Lower": rf_lower,
        "RF_CI_Upper": rf_upper,

        "XGB_Flip_Rate": row["XGB_Flip_Rate"],
        "XGB_CI_Lower": xgb_lower,
        "XGB_CI_Upper": xgb_upper
    })


confidence_results = pd.DataFrame(confidence_results)

print()
print("95% Confidence Intervals for Flip Rate")
print("---------------------------------------")
print(confidence_results)

confidence_results.to_csv(
    "flip_rate_confidence_intervals.csv",
    index=False
)

print()
print("Confidence interval results saved successfully.")
print("File: flip_rate_confidence_intervals.csv")
# ============================================================
# OVERALL MODEL STABILITY SUMMARY
# ============================================================

stability_summary = {
    "Logistic Regression": all_results["Logistic_Stability"].mean(),
    "Random Forest": all_results["RF_Stability"].mean(),
    "XGBoost": all_results["XGB_Stability"].mean()
}

print()
print("Overall Model Stability Summary")
print("--------------------------------")

for model, stability in stability_summary.items():
    print(f"{model}: {stability:.6f} ({stability * 100:.2f}%)")
    # ============================================================
# FINAL MODEL COMPARISON SUMMARY
# ============================================================

final_summary = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest",
        "XGBoost"
    ],

    "Accuracy": [
        accuracy_score(y_test, y_pred),
        accuracy_score(y_test, rf_pred),
        accuracy_score(y_test_xgb, xgb_pred)
    ],

    "Precision": [
        precision_score(y_test, y_pred, pos_label="yes"),
        precision_score(y_test, rf_pred, pos_label="yes"),
        precision_score(y_test_xgb, xgb_pred)
    ],

    "Recall": [
        recall_score(y_test, y_pred, pos_label="yes"),
        recall_score(y_test, rf_pred, pos_label="yes"),
        recall_score(y_test_xgb, xgb_pred)
    ],

    "F1_Score": [
        f1_score(y_test, y_pred, pos_label="yes"),
        f1_score(y_test, rf_pred, pos_label="yes"),
        f1_score(y_test_xgb, xgb_pred)
    ],

    "Average_Stability": [
        all_results["Logistic_Stability"].mean(),
        all_results["RF_Stability"].mean(),
        all_results["XGB_Stability"].mean()
    ]
})

print()
print("Final Model Comparison Summary")
print("--------------------------------")
print(final_summary)

final_summary.to_csv(
    "final_model_comparison.csv",
    index=False
)

print()
print("Final comparison saved successfully.")
print("File: final_model_comparison.csv")
# ============================================================
# ACCURACY VS STABILITY VISUALIZATION
# ============================================================

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))

plt.scatter(
    final_summary["Accuracy"] * 100,
    final_summary["Average_Stability"] * 100,
    s=120
)

for _, row in final_summary.iterrows():
    plt.annotate(
        row["Model"],
        (
            row["Accuracy"] * 100,
            row["Average_Stability"] * 100
        ),
        xytext=(8, 5),
        textcoords="offset points"
    )

plt.xlabel("Baseline Accuracy (%)")
plt.ylabel("Average Decision Stability (%)")
plt.title("Baseline Accuracy vs Decision Stability")
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    "accuracy_vs_stability.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print()
print("Accuracy vs Stability chart saved successfully.")
print("File: accuracy_vs_stability.png")
# ============================================================
# CAMPAIGN PERTURBATION VS FLIP RATE
# ============================================================

plt.figure(figsize=(8, 6))

for model, column in [
    ("Logistic Regression", "Logistic_Flip_Rate"),
    ("Random Forest", "RF_Flip_Rate"),
    ("XGBoost", "XGB_Flip_Rate")
]:
    plt.plot(
        campaign_results["Perturbation"],
        campaign_results[column] * 100,
        marker="o",
        label=model
    )

plt.xlabel("Campaign Perturbation")
plt.ylabel("Prediction Flip Rate (%)")
plt.title("Effect of Campaign Perturbation on Prediction Stability")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    "campaign_perturbation_flip_rate.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print()
print("Campaign perturbation chart saved successfully.")
print("File: campaign_perturbation_flip_rate.png")

# ============================================================
# FEATURE SENSITIVITY COMPARISON
# ============================================================

plt.figure(figsize=(9, 6))

x = range(len(sensitivity_summary["Feature"]))
width = 0.25

plt.bar(
    [i - width for i in x],
    sensitivity_summary["Logistic_Avg_Flip_Rate"] * 100,
    width=width,
    label="Logistic Regression"
)

plt.bar(
    x,
    sensitivity_summary["RF_Avg_Flip_Rate"] * 100,
    width=width,
    label="Random Forest"
)

plt.bar(
    [i + width for i in x],
    sensitivity_summary["XGB_Avg_Flip_Rate"] * 100,
    width=width,
    label="XGBoost"
)

plt.xticks(
    list(x),
    sensitivity_summary["Feature"]
)

plt.xlabel("Feature")
plt.ylabel("Average Prediction Flip Rate (%)")
plt.title("Feature-Level Prediction Sensitivity")
plt.legend()
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    "final_feature_sensitivity.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print()
print("Final feature sensitivity chart saved successfully.")
print("File: final_feature_sensitivity.png")