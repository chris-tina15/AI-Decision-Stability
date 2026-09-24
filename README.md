# High Accuracy and Decision Stability in Machine Learning

### An Experimental Study of Sensitivity to Minimal Input Perturbations

---

## 📌 Overview

Machine learning models are commonly evaluated using metrics such as accuracy, precision, recall, and F1-score. However, high predictive performance does not necessarily describe how consistently a model behaves when its input values undergo small changes.

This project experimentally investigates the relationship between **predictive accuracy and decision stability** by applying small, controlled perturbations to selected input features and observing whether the model changes its predictions.

The study compares three conventional machine learning models:

- Logistic Regression
- Random Forest
- XGBoost

The experimental results are further examined using prediction flip rates, stability scores, feature-wise sensitivity analysis, McNemar's Exact Test, and 95% confidence intervals.

---

## 🎯 Research Question

> **Does high predictive accuracy guarantee stable machine learning decisions under minimal input perturbations?**

---

## 🔬 Research Objective

The main objective of this study is to experimentally examine whether models with high predictive accuracy also maintain stable predictions when selected input features are changed by small, controlled amounts.

### Specific Objectives

- Evaluate the baseline predictive performance of different machine learning models.
- Apply controlled perturbations to selected numerical features.
- Measure changes between original and perturbed predictions.
- Quantify prediction stability and flip rates.
- Identify features associated with greater prediction sensitivity.
- Statistically validate directional changes in predictions.
- Examine the relationship between predictive performance and decision stability.

---

## 📊 Dataset

The experiment uses the **UCI Bank Marketing Dataset**.

### Dataset characteristics

- **Instances:** 45,211
- **Target variable:** `y`
- **Target classes:** `yes`, `no`
- **Original input features:** 16
- **Primary experimental features:** 15
- **Task:** Binary classification

The `duration` feature is excluded from the primary experiment because it represents the duration of the contact interaction and may introduce an unrealistic predictive setting for a pre-contact decision.

---

## ⚙️ Methodology

The experimental workflow consists of the following stages:

```text
Bank Marketing Dataset
        ↓
Data Preprocessing
        ↓
80/20 Stratified Train-Test Split
        ↓
Feature Transformation
        ↓
Baseline Model Training
        ↓
Baseline Performance Evaluation
        ↓
Controlled Input Perturbation
        ↓
Original vs Perturbed Predictions
        ↓
Decision Stability Analysis
        ↓
Feature Sensitivity Analysis
        ↓
Statistical Validation
        ↓
Results and Interpretation
