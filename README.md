# 🎓 Student Math Score Prediction – End-to-End ML Project

## 📌 Problem Statement
Predict a student’s **math score** based on demographic information and academic performance
in reading and writing, using a robust machine learning pipeline.

This project demonstrates a **complete ML lifecycle**:
EDA → preprocessing → model training → evaluation → inference → deployment.

---

## 📊 Dataset
- Source: Student Performance Dataset
- Rows: ~1000 students
- Features include:
  - Gender
  - Race/Ethnicity
  - Parental level of education
  - Lunch type
  - Test preparation course
  - Reading score
  - Writing score

---

## 🎯 Target Variable
- **Math Score** (Regression problem)

### ⚠️ Data Leakage Handling
- Excluded derived columns such as `total_score` and `average_score`
- Ensured strict separation between features and target

---

## 🧠 Feature Engineering
- Categorical features → **OneHotEncoding**
- Numerical features → **Standard Scaling**
- Implemented using `ColumnTransformer`
- Reused the same preprocessing pipeline for training and inference

---

## 🤖 Models Evaluated
- Linear Regression (baseline)
- Random Forest Regressor
- ElasticNet
- Gradient Boosting Regressor

### 📈 Evaluation Metrics
- MAE
- RMSE
- R² Score
- 5-Fold Cross-Validation for stability

---

## 🏆 Final Model Selection
**Gradient Boosting Regressor**

**Reason:**
- Slightly lower mean R² than Linear Regression
- **Significantly lower variance in cross-validation**
- Better handling of non-linear relationships and feature interactions
- More robust for real-world deployment

---

## 🔍 Inference
Run prediction locally:

```bash
python src/predict.py

## 📷 Screenshots

![App UI] & [Prediction output]

## 🚀 Live Demo

🔗 Streamlit App:  
https://student-performance-ml-ufsmu9zzkjzyp2tdqqkekm.streamlit.app/

