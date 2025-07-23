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

<img width="579" height="464" alt="image" src="https://github.com/user-attachments/assets/5a97809d-437c-41cc-aaef-29397f4701ac" />

---

## 🧼 Preprocessing Pipeline

- Binary mapping via custom transformer
- Log transformation + scaling on numeric columns
- One-hot encoding on categorical features
- Final feature matrix prepared using `ColumnTransformer`

<img width="5400" height="900" alt="fixed_arrow_preprocessing_pipeline" src="https://github.com/user-attachments/assets/8ef00743-5efc-4240-b630-3e0679edb8e3" />


---

## 🧪 Modeling

Models tested:
- Logistic Regression
- Random Forest ✅ (Best model with F1 ~0.80)
- XGBoost

All models tracked via MLflow, and the best one was saved and registered.

<img width="959" height="539" alt="image" src="https://github.com/user-attachments/assets/470f9a87-976e-42c5-8a34-42498c5e9d96" />

---

## 🧰 ML Workflow on AWS

### 🔸 1. **Amazon S3**
- Uploaded raw and processed data
- Used by Glue jobs and Flask app

---

### 🔸 2. **AWS Glue + Crawler**
- Glue Crawler detects schema from S3 and catalogs it
- Glue Job transforms raw data and loads to Redshift

<img width="959" height="511" alt="Visual_ETL" src="https://github.com/user-attachments/assets/14610942-3c6b-4a6a-adff-5f02f1336060" />

---

### 🔸 3. **Amazon Redshift**
- Structured storage of clean tabular data
- Used by SageMaker for training input

<img width="956" height="508" alt="DataStoredInRedshift" src="https://github.com/user-attachments/assets/6158caca-3c0d-41f0-a588-8687a2a5bb83" />

---

### 🔸 4. **Amazon SageMaker Studio**
- Hosted MLflow tracking server
- Used for training, evaluation, and registry

<img width="959" height="518" alt="MLFlowInStudio 225946" src="https://github.com/user-attachments/assets/131bd64a-e034-4d3e-afa7-c5d7df39fa00" />

<img width="959" height="527" alt="MLFlowAWS" src="https://github.com/user-attachments/assets/781647b5-de77-437f-94c3-994839890075" />

---

### 🔸 5. **MLflow Registry**
- Logged all models
- Registered best model and transitioned to Staging

<img width="959" height="504" alt="BestModelRegisteredMLFlow" src="https://github.com/user-attachments/assets/99ce6fb4-4e60-4a3d-b5f8-a758eb35ef9c" />


### 🔸 6. **Flask Deployment**
- Flask app serves prediction via REST endpoint
- Uses latest `.pkl` model artifact from S3

<img width="959" height="544" alt="FlaskDeploymentAWS" src="https://github.com/user-attachments/assets/3d9fc474-68dd-4a97-97e1-b882ed72ac1a" />

<img width="959" height="539" alt="FlaskAppAWS" src="https://github.com/user-attachments/assets/46f4fdfa-3bb0-46c8-b6cc-592351edcc26" />

---

### 🔸 7. **Apache Airflow (MWAA)**
- Orchestrates full pipeline:
    - New data detection
    - Preprocessing → Training → Drift Check
    - Branching: alert or redeploy
- Dynamically runs Python scripts from S3

<img width="959" height="541" alt="image" src="https://github.com/user-attachments/assets/b3943411-f119-4891-81f0-fe657315b854" />


---

## 🧠 Drift Detection

- Uses `ks_2samp` from SciPy on numeric fields
- If drift detected, model retraining is triggered
- Drift reports (JSON + HTML via Evidently) are saved in S3

<img width="470" height="480" alt="image" src="https://github.com/user-attachments/assets/67eb4c73-29d3-43e1-bf09-2835d356ee81" />

---

### AWS Architecture

<img width="1024" height="1024" alt="ChatGPT Image Jul 20, 2025, 11_59_48 PM" src="https://github.com/user-attachments/assets/157a6e15-8039-4670-a8c7-000a3c9164bf" />

<img width="1655" height="3844" alt="Untitled diagram _ Mermaid Chart-2025-07-21-054418" src="https://github.com/user-attachments/assets/54cade26-0cd7-4081-9866-b3923710289a" />


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

This solution is client-ready and easily deployable in any AWS cloud setup using S3, SageMaker, and MWAA. For further inquiries or support, please open an issue or contact.
