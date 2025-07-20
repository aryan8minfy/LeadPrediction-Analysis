
# 🧠 Sales Conversion Prediction – Intelligent Lead Scoring with ML & AWS

## 📌 Project Overview

This project presents an end-to-end machine learning pipeline that predicts the likelihood of a lead converting to a customer. Designed for a real-world ed-tech scenario (X Education), this solution helps sales teams focus on high-conversion prospects using predictive modeling, AWS cloud infrastructure, and ML workflow orchestration.

---

## 🎯 Business Objective

- **Problem**: Large volume of incoming leads (~30% conversion rate), but no mechanism to identify high-value leads.
- **Goal**: Build a data-driven lead scoring system to increase conversion efficiency to ~80% by prioritizing “Hot Leads.”
- **Solution**: Develop and deploy an ML pipeline that processes raw data, scores leads, and automates retraining via drift detection.

---

## 🛠️ Tools & Services Used

| Layer                | Tools & Services                                                                 |
|---------------------|----------------------------------------------------------------------------------|
| Data Storage         | Amazon S3                                                                       |
| ETL & Transformation | AWS Glue, Glue Crawler                                                           |
| Data Warehouse       | Amazon Redshift                                                                 |
| Model Development    | Amazon SageMaker Studio, Scikit-learn, XGBoost                                  |
| Experiment Tracking  | MLflow (hosted on SageMaker Studio)                                             |
| Deployment           | Flask API                                                                       |
| Orchestration        | Apache Airflow (MWAA)                                                            |
| Monitoring           | Drift Detection using SciPy, Alerts in Airflow DAG                              |

---

## 📊 Dataset Details

- Source: Provided CSV containing 9240 leads and 37 columns.
- Target: `Converted` (1 = converted, 0 = not converted)
- Format: Mixed numeric, binary, and categorical fields

---

## 🔍 EDA & Insights

- ✅ Missing values handled with imputation or column drops.
- ✅ Skewed features like `TotalVisits` were log-transformed.
- ✅ Rare categorical values grouped as `'Other'`.
- ✅ Outliers removed using IQR method.

📸 *[Add a screenshot of your EDA notebook here]*

---

## 🧼 Preprocessing Pipeline

- Binary mapping via custom transformer
- Log transformation + scaling on numeric columns
- One-hot encoding on categorical features
- Final feature matrix prepared using `ColumnTransformer`

📸 *[Add schema pipeline diagram or transformation logs]*

---

## 🧪 Modeling

Models tested:
- Logistic Regression
- Random Forest ✅ (Best model with F1 ~0.80)
- XGBoost

All models tracked via MLflow, and the best one was saved and registered.

📸 *[Add MLflow UI screenshot showing experiment comparison]*

---

## 🧰 ML Workflow on AWS

### 🔸 1. **Amazon S3**
- Uploaded raw and processed data
- Used by Glue jobs and Flask app

### 🔸 2. **AWS Glue + Crawler**
- Glue Crawler detects schema from S3 and catalogs it
- Glue Job transforms raw data and loads to Redshift

📸 *[Insert screenshot of Glue Job and Crawler success]*

### 🔸 3. **Amazon Redshift**
- Structured storage of clean tabular data
- Used by SageMaker for training input

### 🔸 4. **Amazon SageMaker Studio**
- Hosted MLflow tracking server
- Used for training, evaluation, and registry

### 🔸 5. **MLflow Registry**
- Logged all models
- Registered best model and transitioned to Staging

📸 *[Insert screenshot of MLflow model registry]*

### 🔸 6. **Flask Deployment**
- Flask app serves prediction via REST endpoint
- Uses latest `.pkl` model artifact from S3

📸 *[Screenshot of prediction form or response sample]*

### 🔸 7. **Apache Airflow (MWAA)**
- Orchestrates full pipeline:
    - New data detection
    - Preprocessing → Training → Drift Check
    - Branching: alert or redeploy
- Dynamically runs Python scripts from S3

📸 *[Airflow DAG screenshot + logs from drift check or model training]*

---

## 🧠 Drift Detection

- Uses `ks_2samp` from SciPy on numeric fields
- If drift detected, model retraining is triggered
- Drift reports (JSON + HTML via Evidently) are saved in S3

📸 *[Attach drift report sample or HTML dashboard]*

---

## 🔁 Folder Structure

```bash
sales-conversion-prediction/
├── data/
│   └── sample_cleaned.csv
├── notebooks/
│   └── eda_analysis.ipynb
├── scripts/
│   ├── preprocess.py
│   ├── train_model.py
│   ├── predict.py
│   ├── drift_check.py
│   └── deploy_flask.py
├── dags/
│   └── lead_scoring_pipeline.py
├── aws/
│   ├── aws_architecture_diagram.png
│   └── glue_etl_flow.png
├── presentation/
│   └── Sales_Conversion_Client_Presentation.pptx
├── requirements/
│   └── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 Results & Business Impact

- Random Forest model achieved over **80% conversion accuracy** on top-scored leads.
- Delivered a **production-grade scoring system** using AWS-native services.
- Automated workflows save manual effort and **enable real-time retraining** on drift.

📌 *Potential CRM integration + live dashboard = next phase*

---

## 📧 Contact / Deployment Readiness

This solution is client-ready and easily deployable in any AWS cloud setup using S3, SageMaker, and MWAA. For further inquiries or support, please open an issue or contact [YourName/Team].
