
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# AI DECISION STABILITY ANALYSIS
# ============================================================

st.set_page_config(
    page_title="AI Decision Stability Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"

FINAL_COMPARISON = BASE_DIR / "final_model_comparison.csv"
PERTURBATION_RESULTS = BASE_DIR / "perturbation_results.csv"
SENSITIVITY_RESULTS = BASE_DIR / "feature_sensitivity_summary.csv"
MCNEMAR_RESULTS = BASE_DIR / "mcnemar_results.csv"
CI_RESULTS = BASE_DIR / "flip_rate_confidence_intervals.csv"
DATASET_FILE = DATA_DIR / "bank-full.csv"


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_csv(path):
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data
def load_dataset(path):
    if path.exists():
        return pd.read_csv(path, sep=";")
    return None


final_df = load_csv(FINAL_COMPARISON)
perturbation_df = load_csv(PERTURBATION_RESULTS)
sensitivity_df = load_csv(SENSITIVITY_RESULTS)
mcnemar_df = load_csv(MCNEMAR_RESULTS)
ci_df = load_csv(CI_RESULTS)
dataset_df = load_dataset(DATASET_FILE)


def check_file(data, filename):
    if data is None:
        st.error(
            f"Required file not found: `{filename}`\n\n"
            f"Expected location:\n`{BASE_DIR}`"
        )
        return False
    return True


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🔬 AI Decision Stability")

st.sidebar.caption(
    "Experimental analysis of machine-learning decision stability"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Project Sections",
    [
        "Overview",
        "Dataset & Preprocessing",
        "Baseline Performance",
        "Perturbation Analysis",
        "Decision Stability",
        "Feature Sensitivity",
        "Statistical Validation",
        "Results & Conclusion"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "Project: Does High Accuracy Guarantee Stable AI Decisions?"
)


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.title("🔬 AI Decision Stability Analysis")

    st.markdown(
        """
        ## Does High Accuracy Guarantee Stable AI Decisions?

        This project experimentally investigates whether machine-learning
        models that achieve high predictive accuracy also maintain stable
        predictions when their input features undergo small, controlled
        perturbations.
        """
    )

    st.divider()

    st.header("Research Question")

    st.info(
        """
        Does a highly accurate machine-learning model also maintain stable
        predictions when its input features are changed by small,
        controlled amounts?
        """
    )

    st.divider()

    st.header("Project Objective")

    st.write(
        """
        The objective is to compare conventional predictive performance
        with decision stability. Three machine-learning models are trained
        on the same dataset, evaluated on the same test set, and subjected
        to controlled perturbations of selected numerical features.
        """

    )

    st.divider()

    st.header("Experimental Pipeline")

    pipeline = [
        ("01", "Dataset", "Bank Marketing Dataset"),
        ("02", "Preprocessing", "Scaling and One-Hot Encoding"),
        ("03", "Model Training", "Three ML Models"),
        ("04", "Baseline Evaluation", "Accuracy, Precision, Recall, F1"),
        ("05", "Perturbation", "Controlled Feature Changes"),
        ("06", "Stability Analysis", "Flip Rate and Stability"),
        ("07", "Statistical Analysis", "McNemar Test and 95% CI"),
        ("08", "Interpretation", "Accuracy vs Stability")
    ]

    cols = st.columns(4)

    for i, (number, title, description) in enumerate(pipeline):
        with cols[i % 4]:
            st.markdown(f"### {number}")
            st.markdown(f"**{title}**")
            st.caption(description)

    st.divider()

    st.header("Machine-Learning Models")

    model_cols = st.columns(3)

    model_information = [
        (
            "Logistic Regression",
            "Linear classification model",
            "Baseline model"
        ),
        (
            "Random Forest",
            "Bagging tree ensemble",
            "Non-linear ensemble"
        ),
        (
            "XGBoost",
            "Gradient boosting ensemble",
            "Boosting model"
        )
    ]

    for col, (name, model_type, role) in zip(
        model_cols, model_information
    ):
        with col:
            st.subheader(name)
            st.write(f"**Type:** {model_type}")
            st.write(f"**Role:** {role}")

    st.divider()

    st.header("Core Evaluation Measures")

    metric_cols = st.columns(4)

    with metric_cols[0]:
        st.metric("Predictive Performance", "Accuracy / F1")

    with metric_cols[1]:
        st.metric("Decision Change", "Flip Rate")

    with metric_cols[2]:
        st.metric("Decision Consistency", "Stability")

    with metric_cols[3]:
        st.metric("Statistical Analysis", "McNemar + 95% CI")


# ============================================================
# DATASET & PREPROCESSING
# ============================================================

elif page == "Dataset & Preprocessing":

    st.title("📊 Dataset & Preprocessing")

    if dataset_df is not None:

        st.header("Dataset Overview")

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric("Instances", f"{len(dataset_df):,}")

        with c2:
            st.metric("Original Features", len(dataset_df.columns) - 1)

        with c3:
            st.metric("Target", "y")

        with c4:
            st.metric("Classes", dataset_df["y"].nunique())

        st.divider()

        st.header("Target Distribution")

        target_counts = dataset_df["y"].value_counts()

        fig, ax = plt.subplots(figsize=(7, 4))
        target_counts.plot(kind="bar", ax=ax)
        ax.set_title("Target Class Distribution")
        ax.set_xlabel("Target Class")
        ax.set_ylabel("Number of Samples")
        ax.tick_params(axis="x", rotation=0)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.dataframe(
            target_counts.rename("Count").to_frame(),
            use_container_width=True
        )

    else:
        st.warning(
            "`Data/bank-full.csv` was not found. Dataset visualization "
            "is unavailable, but the result pages can still be viewed."
        )

    st.divider()

    st.header("Features Used in the Experiment")

    feature_table = pd.DataFrame({
        "Feature Type": [
            "Numerical",
            "Categorical",
            "Target"
        ],
        "Variables": [
            "age, balance, day, campaign, pdays, previous",
            "job, marital, education, default, housing, loan, contact, month, poutcome",
            "y"
        ]
    })

    st.dataframe(
        feature_table,
        hide_index=True,
        use_container_width=True
    )

    st.divider()

    st.header("Preprocessing Pipeline")

    st.code(
        """
Bank Marketing Dataset
          |
          v
Remove target y
          |
          v
Exclude duration
          |
          v
80/20 Stratified Train-Test Split
          |
          +--------------------+
          |                    |
          v                    v
Numerical Features       Categorical Features
          |                    |
          v                    v
 StandardScaler          OneHotEncoder
          |                    |
          +---------+----------+
                    |
                    v
             ColumnTransformer
                    |
                    v
          50-Dimensional Matrix
        """,
        language="text"
    )

    st.info(
        """
        The `duration` feature is excluded from the primary experiment
        because it is associated with the contact interaction and can
        create an unrealistic predictive setting for a pre-contact
        decision.
        """
    )


# ============================================================
# BASELINE PERFORMANCE
# ============================================================

elif page == "Baseline Performance":

    st.title("📈 Baseline Model Performance")

    if not check_file(
        final_df,
        "final_model_comparison.csv"
    ):
        st.stop()

    st.header("Performance Summary")

    display_df = final_df.copy()

    st.dataframe(
        display_df.style.format({
            "Accuracy": "{:.2%}",
            "Precision": "{:.2%}",
            "Recall": "{:.2%}",
            "F1_Score": "{:.2%}",
            "Average_Stability": "{:.2%}"
        }),
        hide_index=True,
        use_container_width=True
    )

    st.divider()

    st.header("Baseline Performance Comparison")

    plot_df = final_df.set_index("Model")[
        ["Accuracy", "Precision", "Recall", "F1_Score"]
    ] * 100

    fig, ax = plt.subplots(figsize=(10, 5))

    plot_df.plot(kind="bar", ax=ax)

    ax.set_title("Baseline Model Performance")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Metric")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    st.header("Model-wise Metrics")

    cols = st.columns(3)

    for col, (_, row) in zip(cols, final_df.iterrows()):
        with col:
            st.subheader(row["Model"])
            st.metric(
                "Accuracy",
                f"{row['Accuracy'] * 100:.2f}%"
            )
            st.metric(
                "F1-Score",
                f"{row['F1_Score'] * 100:.2f}%"
            )
            st.metric(
                "Recall",
                f"{row['Recall'] * 100:.2f}%"
            )


# ============================================================
# PERTURBATION ANALYSIS
# ============================================================

elif page == "Perturbation Analysis":

    st.title("🧪 Controlled Perturbation Analysis")

    if not check_file(
        perturbation_df,
        "perturbation_results.csv"
    ):
        st.stop()

    st.write(
        """
        Each experiment starts from the original test set. One selected
        feature is modified while the remaining features remain unchanged.
        The original and perturbed predictions are then compared.
        """
    )

    st.header("Perturbation Features")

    feature_info = pd.DataFrame({
        "Feature": [
            "age",
            "balance",
            "campaign",
            "previous"
        ],
        "Perturbation Method": [
            "Integer change",
            "Percentage change",
            "Integer change",
            "Integer change"
        ]
    })

    st.dataframe(
        feature_info,
        hide_index=True,
        use_container_width=True
    )

    st.divider()

    selected_feature = st.selectbox(
        "Select a feature",
        sorted(perturbation_df["Feature"].unique())
    )

    selected_data = perturbation_df[
        perturbation_df["Feature"] == selected_feature
    ].copy()

    st.header(
        f"{selected_feature}: Perturbation vs Flip Rate"
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        selected_data["Perturbation"],
        selected_data["Logistic_Flip_Rate"] * 100,
        marker="o",
        label="Logistic Regression"
    )

    ax.plot(
        selected_data["Perturbation"],
        selected_data["RF_Flip_Rate"] * 100,
        marker="o",
        label="Random Forest"
    )

    ax.plot(
        selected_data["Perturbation"],
        selected_data["XGB_Flip_Rate"] * 100,
        marker="o",
        label="XGBoost"
    )

    ax.set_xlabel("Perturbation Level")
    ax.set_ylabel("Prediction Flip Rate (%)")
    ax.set_title(
        f"Effect of {selected_feature} Perturbation"
    )
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    st.header("Perturbation Results")

    result_columns = [
        "Feature",
        "Perturbation",
        "Logistic_Changed",
        "Logistic_Flip_Rate",
        "Logistic_Stability",
        "RF_Changed",
        "RF_Flip_Rate",
        "RF_Stability",
        "XGB_Changed",
        "XGB_Flip_Rate",
        "XGB_Stability"
    ]

    result_table = selected_data[result_columns].copy()

    rate_columns = [
        "Logistic_Flip_Rate",
        "Logistic_Stability",
        "RF_Flip_Rate",
        "RF_Stability",
        "XGB_Flip_Rate",
        "XGB_Stability"
    ]

    st.dataframe(
        result_table.style.format({
            "Logistic_Flip_Rate": "{:.3%}",
            "Logistic_Stability": "{:.3%}",
            "RF_Flip_Rate": "{:.3%}",
            "RF_Stability": "{:.3%}",
            "XGB_Flip_Rate": "{:.3%}",
            "XGB_Stability": "{:.3%}"
        }),
        hide_index=True,
        use_container_width=True
    )


# ============================================================
# DECISION STABILITY
# ============================================================

elif page == "Decision Stability":

    st.title("📐 Decision Stability")

    if not check_file(
        final_df,
        "final_model_comparison.csv"
    ):
        st.stop()

    st.header("Metric Definitions")

    st.latex(
        r"""
        Flip\ Rate =
        \frac{\text{Changed Predictions}}
        {\text{Total Predictions}}
        """
    )

    st.latex(
        r"""
        Stability = 1 - Flip\ Rate
        """
    )

    st.divider()

    st.header("Average Stability Across Experiments")

    cols = st.columns(3)

    for col, (_, row) in zip(cols, final_df.iterrows()):
        with col:
            st.metric(
                row["Model"],
                f"{row['Average_Stability'] * 100:.2f}%"
            )
            st.caption(
                f"Average flip rate: "
                f"{(1 - row['Average_Stability']) * 100:.2f}%"
            )

    st.divider()

    st.header("Accuracy vs Decision Stability")

    fig, ax = plt.subplots(figsize=(9, 6))

    for _, row in final_df.iterrows():

        accuracy = row["Accuracy"] * 100
        stability = row["Average_Stability"] * 100

        ax.scatter(
            accuracy,
            stability,
            s=130
        )

        ax.annotate(
            row["Model"],
            (accuracy, stability),
            xytext=(8, 6),
            textcoords="offset points"
        )

    ax.set_xlabel("Baseline Accuracy (%)")
    ax.set_ylabel("Average Decision Stability (%)")
    ax.set_title(
        "Baseline Accuracy vs Average Decision Stability"
    )
    ax.grid(alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.info(
        """
        Accuracy measures predictive performance on the original test
        data, while stability measures prediction consistency after
        controlled input perturbations.
        """
    )


# ============================================================
# FEATURE SENSITIVITY
# ============================================================

elif page == "Feature Sensitivity":

    st.title("🎯 Feature-Level Sensitivity")

    if not check_file(
        sensitivity_df,
        "feature_sensitivity_summary.csv"
    ):
        st.stop()

    display_df = sensitivity_df.copy()

    st.header("Average Flip Rate by Feature")

    st.dataframe(
        display_df.style.format({
            "Logistic_Avg_Flip_Rate": "{:.3%}",
            "RF_Avg_Flip_Rate": "{:.3%}",
            "XGB_Avg_Flip_Rate": "{:.3%}"
        }),
        hide_index=True,
        use_container_width=True
    )

    st.divider()

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(display_df["Feature"]))
    width = 0.25

    ax.bar(
        x - width,
        display_df["Logistic_Avg_Flip_Rate"] * 100,
        width,
        label="Logistic Regression"
    )

    ax.bar(
        x,
        display_df["RF_Avg_Flip_Rate"] * 100,
        width,
        label="Random Forest"
    )

    ax.bar(
        x + width,
        display_df["XGB_Avg_Flip_Rate"] * 100,
        width,
        label="XGBoost"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(display_df["Feature"])
    ax.set_xlabel("Feature")
    ax.set_ylabel("Average Prediction Flip Rate (%)")
    ax.set_title("Feature-Level Prediction Sensitivity")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.warning(
        """
        These averages are interpreted within the perturbation scales
        used for each feature. They are not conventional feature-
        importance scores.
        """
    )


# ============================================================
# STATISTICAL VALIDATION
# ============================================================

elif page == "Statistical Validation":

    st.title("📊 Statistical Validation")

    st.header("McNemar's Exact Test")

    st.write(
        """
        McNemar's exact test is applied to paired original and perturbed
        predictions to examine whether prediction changes are directionally
        asymmetric between 0→1 and 1→0 transitions.
        """
    )

    st.latex(
        r"""
        H_0: P(0\rightarrow1)=P(1\rightarrow0)
        """
    )

    st.latex(
        r"""
        H_1: P(0\rightarrow1)\neq P(1\rightarrow0)
        """
    )

    st.write("Significance level: **α = 0.05**")

    if check_file(mcnemar_df, "mcnemar_results.csv"):

        model_key = st.selectbox(
            "Select model",
            [
                "Logistic_McNemar_p",
                "RF_McNemar_p",
                "XGB_McNemar_p"
            ]
        )

        model_name = {
            "Logistic_McNemar_p": "Logistic Regression",
            "RF_McNemar_p": "Random Forest",
            "XGB_McNemar_p": "XGBoost"
        }[model_key]

        selected_mcnemar = mcnemar_df[
            ["Feature", "Perturbation", model_key]
        ].copy()

        selected_mcnemar["Result"] = np.where(
            selected_mcnemar[model_key] < 0.05,
            "Significant",
            "Not significant"
        )

        st.subheader(
            f"{model_name}: McNemar Test Results"
        )

        st.dataframe(
            selected_mcnemar.style.format({
                model_key: "{:.6g}"
            }),
            hide_index=True,
            use_container_width=True
        )

        st.divider()

        fig, ax = plt.subplots(figsize=(10, 5))

        p_values = selected_mcnemar[model_key].values

        ax.plot(
            range(len(p_values)),
            p_values,
            marker="o"
        )

        ax.axhline(
            0.05,
            linestyle="--",
            label="α = 0.05"
        )

        ax.set_yscale("log")
        ax.set_xlabel("Perturbation Experiment")
        ax.set_ylabel("Exact p-value")
        ax.set_title(
            f"{model_name}: McNemar Exact Test p-values"
        )
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.divider()

    st.header("95% Confidence Intervals for Flip Rate")

    st.write(
        """
        The confidence intervals quantify uncertainty around the observed
        prediction flip rates.
        """
    )

    if check_file(
        ci_df,
        "flip_rate_confidence_intervals.csv"
    ):

        ci_choice = st.selectbox(
            "Select model for confidence interval analysis",
            ["Logistic", "RF", "XGB"]
        )

        rate_col = f"{ci_choice}_Flip_Rate"
        lower_col = f"{ci_choice}_CI_Lower"
        upper_col = f"{ci_choice}_CI_Upper"

        selected_ci = ci_df[
            [
                "Feature",
                "Perturbation",
                rate_col,
                lower_col,
                upper_col
            ]
        ].copy()

        st.dataframe(
            selected_ci.style.format({
                rate_col: "{:.3%}",
                lower_col: "{:.3%}",
                upper_col: "{:.3%}"
            }),
            hide_index=True,
            use_container_width=True
        )

        fig, ax = plt.subplots(figsize=(10, 5))

        x = np.arange(len(selected_ci))

        rate = selected_ci[rate_col].values * 100
        lower = selected_ci[lower_col].values * 100
        upper = selected_ci[upper_col].values * 100

        error = np.vstack([
            rate - lower,
            upper - rate
        ])

        ax.errorbar(
            x,
            rate,
            yerr=error,
            fmt="o",
            capsize=4
        )

        ax.set_xlabel("Perturbation Experiment")
        ax.set_ylabel("Flip Rate (%)")
        ax.set_title(
            f"{ci_choice}: Flip Rate with 95% Confidence Intervals"
        )
        ax.grid(alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.info(
        """
        McNemar's test evaluates directional asymmetry of prediction
        changes. The confidence interval describes uncertainty around
        the observed flip-rate estimate. Stability remains the primary
        measure of prediction consistency.
        """
    )


# ============================================================
# RESULTS & CONCLUSION
# ============================================================

elif page == "Results & Conclusion":

    st.title("🔎 Results & Conclusion")

    if not check_file(
        final_df,
        "final_model_comparison.csv"
    ):
        st.stop()

    st.header("Main Experimental Results")

    st.dataframe(
        final_df.style.format({
            "Accuracy": "{:.2%}",
            "Precision": "{:.2%}",
            "Recall": "{:.2%}",
            "F1_Score": "{:.2%}",
            "Average_Stability": "{:.2%}"
        }),
        hide_index=True,
        use_container_width=True
    )

    st.divider()

    st.header("Key Findings")

    findings = [
        "Baseline predictive accuracy and decision stability are distinct evaluation measures.",
        "The three models responded differently to the same controlled perturbations.",
        "Campaign produced the largest average flip-rate values among the tested features in this experimental setup.",
        "Most predictions remained unchanged across the tested perturbation conditions.",
        "Statistical analysis provides additional evidence about directional prediction changes and uncertainty in the observed flip rates."
    ]

    for i, finding in enumerate(findings, 1):
        st.markdown(f"**Finding {i}.** {finding}")

    st.divider()

    st.header("Overall Conclusion")

    st.success(
        """
        The experimental results indicate that high predictive accuracy
        alone does not completely characterize machine-learning decision
        behavior. Evaluating prediction stability under controlled input
        perturbations provides an additional perspective on model behavior.
        """
    )

    st.divider()

    st.header("Limitations")

    limitations = [
        "The study uses one primary dataset.",
        "Only selected numerical features were perturbed.",
        "Different features were tested using different perturbation scales.",
        "The experiment focuses on controlled perturbations rather than every possible real-world distribution shift.",
        "The conclusions are limited to the evaluated models, dataset, and perturbation design."
    ]

    for item in limitations:
        st.markdown(f"- {item}")

    st.header("Future Work")

    future_work = [
        "Evaluate additional tabular datasets.",
        "Investigate probability-level sensitivity in addition to class-label flips.",
        "Evaluate additional perturbation strategies.",
        "Study robustness under realistic noise and distribution shifts.",
        "Extend the analysis to additional machine-learning architectures."
    ]

    for item in future_work:
        st.markdown(f"- {item}")

    st.divider()

    st.caption(
        "AI Decision Stability Analysis | Experimental Research Dashboard"
    )
