# 📊 Customer Churn Prediction using Machine Learning

This project focuses on predicting customer churn for a bank using structured customer data. It leverages two powerful machine learning models — **XGBoost** and **Random Forest Classifier** — to classify customers as churners or non-churners with high accuracy.

---

## 📌 Dataset Source

The dataset used in this project is adapted from:  
🔗 [TechVidvan – Customer Churn Prediction](https://techvidvan.com/tutorials/wp-content/uploads/sites/2/2022/11/customer-churn-plot.png)

---

## 🎯 Objective

- Predict whether a customer is likely to churn based on historical data
- Compare model performance between XGBoost and Random Forest
- Understand churn drivers through visual analytics and feature importance
- Enhance business decision-making using data insights

---

## 🧠 Models & Evaluation

We trained and evaluated two models on the processed dataset:

| Model             | Accuracy | Churn Precision | Churn Recall | Churn F1-Score |
|------------------|----------|------------------|----------------|----------------|
| **XGBoost**       | 0.8652   | 0.75             | 0.50           | 0.60           |
| **Random Forest** | 0.8632   | 0.79             | 0.45           | 0.57           |

- Both models achieved strong accuracy.
- XGBoost performed slightly better in identifying churners (higher recall).

---

## 📉 Classification Reports

### ✅ XGBoost Classification Report & Confusion Matrix
![XGBoost Report](./assets/xgboost_report.png)

### ✅ Random Forest Classification Report & Confusion Matrix
![Random Forest Report](./assets/random_forest_report.png)

---

## 📊 Data Visualization Insights

### 🧓 Age Distribution Across Geography
Age distribution of customers across France, Spain, and Germany.

![Age vs Geography](./assets/age_by_geography.png)

---

### 🌍 Churn Analysis by Geography

#### 🔁 Churn Distribution
Overall churner distribution by country.

![Overall Geography Distribution](./assets/overall_location_donut.png)

#### ❌ Churners vs ✅ Non-Churners by Location
Churn comparison by geography to help identify high-risk regions.

![Churners & Non-Churners by Location](./assets/churners_vs_nonchurners.png)

---

## 📦 Technologies Used

- Python
- Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn
- XGBoost

---

## 📈 Future Improvements

- Hyperparameter tuning with `GridSearchCV`
- Model explainability with SHAP
- Deployment as a web app using Streamlit or Flask

---



