# Bitcoin-Price-Prediction-using-Python-
This project focuses on analyzing historical Bitcoin price data and developing machine learning models to predict future trends. Using libraries like pandas, numpy, matplotlib, and advanced ML models such as Logistic Regression, SVM, and XGBoost, we explore various financial features and perform data visualization, feature engineering.


# 📈 Bitcoin Price Prediction using Machine Learning

This project aims to predict the future movement of Bitcoin prices using historical data and machine learning algorithms. The approach involves exploratory data analysis, feature engineering, and classification models.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [Features Engineered](#features-engineered)
- [Modeling](#modeling)
- [Results](#results)
- [Usage](#usage)
- [License](#license)

---

## 🔍 Overview

We analyze the historical price data of Bitcoin from 2014 onwards to forecast whether the closing price of the next day will be higher than the current day (binary classification). This involves:

- Cleaning and visualizing price trends
- Feature extraction like `open-close`, `low-high`, `is_quarter_end`
- Building models using Logistic Regression, Support Vector Machine, and XGBoost
- Evaluating models with ROC-AUC metrics

---

## 🛠 Tech Stack

- Python
- Jupyter Notebook
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- XGBoost

---

## 📊 Dataset

The dataset includes daily Bitcoin trading data:

- Date
- Open, High, Low, Close Prices
- Adjusted Close (dropped due to redundancy)
- Volume

CSV Source: `crypto.csv`

---

## 🧠 Features Engineered

- `open-close`: Difference between open and close prices
- `low-high`: Difference between low and high prices
- `is_quarter_end`: Flag indicating if the day is end of a quarter
- `target`: Binary label indicating if next day’s close price is higher

---

## 🤖 Modeling

Three machine learning models are compared:

- Logistic Regression
- Support Vector Machine (SVM)
- XGBoost Classifier

Evaluation Metric: **ROC-AUC Curve**

---

## 📈 Results

- Feature correlation checks ensured no high multicollinearity.
- Distribution and box plots provided insights on outliers and data spread.
- XGBoost generally outperforms others in prediction accuracy.


