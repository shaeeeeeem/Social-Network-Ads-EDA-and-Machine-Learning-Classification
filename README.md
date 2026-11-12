# Social-Network-Ads-EDA-and-Machine-Learning-Classification

## Overview
This project predicts whether a user will purchase a product advertised on a social network based on their demographic and behavioral profile. Using a 5,000-record dataset (`Social_Network_Ads_5000_rows.csv`), this project covers end-to-end data cleaning, exploratory data analysis, feature scaling, model training, comparison, and evaluation across multiple classification algorithms.

---

## Project Objectives
- Perform Exploratory Data Analysis (EDA) to evaluate customer distributions, demographic trends, and purchase patterns.
- Preprocess data including encoding categorical features, feature scaling, and splitting into train and test sets.
- Train and benchmark multiple supervised machine learning models to identify the optimal classifier for conversion prediction.
- Evaluate performance using robust classification metrics beyond basic accuracy.

---

## Dataset Description
The dataset contains customer records with demographic attributes and purchase outcomes:

| Feature | Type | Description |
| :--- | :--- | :--- |
| `User ID` | Numeric | Unique customer identifier (dropped during modeling) |
| `Gender` | Categorical | Gender of the user (`Male` / `Female`) |
| `Age` | Numeric | Age of the customer in years |
| `EstimatedSalary` | Numeric | Estimated annual salary |
| `Purchased` | Categorical / Binary | Target variable (`0` = Did not purchase, `1` = Purchased) |

---

## Tech Stack & Libraries
- **Language:** Python
- **Data Manipulation:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`
- **Machine Learning & Modeling:** `scikit-learn`, `xgboost`

---

## Workflow & Methodology

1. **Data Inspection & Cleaning:**
   - Load `Social_Network_Ads_5000_rows.csv`.
   - Verify missing values, data types, and drop irrelevant identifiers (`User ID`).

2. **Exploratory Data Analysis (EDA):**
   - Check target variable distribution (`Purchased`) to evaluate class balance.
   - Plot univariate distributions for `Age` and `EstimatedSalary`.
   - Analyze cross-feature relationships via scatter plots and pair plots segmented by purchase status.
   - Generate correlation heatmaps to observe relationships among numeric variables.

3. **Data Preprocessing & Scaling:**
   - Encode binary categorical variables (e.g., `Gender` using label encoding or one-hot encoding).
   - Partition data into training and testing sets (e.g., 80/20 split with stratification).
   - Apply feature normalization/standardization (`StandardScaler`) to ensure distance-based models perform reliably.

4. **Model Training & Comparison:**
   - Train a suite of baseline and advanced classification algorithms:
     - **Logistic Regression**
     - **K-Nearest Neighbors (KNN)**
     - **Support Vector Machine (SVM)** (Linear & RBF kernel)
     - **Decision Tree Classifier**
     - **Random Forest Classifier**
     - **XGBoost Classifier**

5. **Evaluation Metrics:**
   - **Confusion Matrix:** True Positives, False Positives, True Negatives, False Negatives.
   - **Classification Report:** Precision, Recall, and F1-Score per class.
   - **ROC-AUC Score & Curve:** Discrimination capability across different probability thresholds.

---

## Getting Started

### Prerequisites
Install all necessary packages via `pip`:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
