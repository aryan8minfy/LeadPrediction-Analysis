from airflow import DAG
from airflow.providers.amazon.aws.sensors.s3 import S3KeySensor
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.utils.dates import days_ago
from airflow.utils.trigger_rule import TriggerRule
from datetime import timedelta
import boto3
import subprocess
import sys
import os
import pandas as pd
import numpy as np

default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

S3_BUCKET = 'july17storagebucket'
NEW_DATA_PREFIX = 'new_data/'
SCRIPTS_PREFIX = 'scripts/'

# 🔍 Get latest CSV file from new_data/
def get_latest_file_key(**kwargs):
    s3 = boto3.client('s3')
    try:
        response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=NEW_DATA_PREFIX)
        new_files = [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith('.csv')]
        if not new_files:
            raise ValueError("❌ No new CSV files found in new_data/")
        # Sort by modification time to get the latest file
        objects_with_time = []
        for obj in response.get('Contents', []):
            if obj['Key'].endswith('.csv'):
                objects_with_time.append((obj['Key'], obj['LastModified']))
        latest_file = sorted(objects_with_time, key=lambda x: x[1], reverse=True)[0][0]
        print(f"✅ Latest file found: {latest_file}")
        return latest_file
    except Exception as e:
        print(f"❌ Error finding latest file: {str(e)}")
        raise

# 📥 Enhanced dependencies download
def download_dependencies():
    """Download utils.py and other required files to /tmp with validation"""
    s3 = boto3.client('s3')
    required_files = ['utils.py']
    
    for file_name in required_files:
        local_path = f"/tmp/{file_name}"
        try:
            print(f"📥 Downloading {file_name} from S3 to {local_path}")
            s3.download_file(S3_BUCKET, f"{SCRIPTS_PREFIX}{file_name}", local_path)
            
            # Validate the file was downloaded correctly
            if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
                print(f"✅ Successfully downloaded {file_name} ({os.path.getsize(local_path)} bytes)")
            else:
                raise Exception(f"Downloaded file {file_name} is empty or missing")
                
        except Exception as e:
            print(f"❌ Failed to download {file_name}: {str(e)}")
            raise

# 🧼 Enhanced script runner with better error handling
def run_script_from_s3(script_name, args=None):
    s3 = boto3.client('s3')
    local_path = f"/tmp/{script_name}"

    try:
        print(f"📥 Downloading {script_name} from S3 to {local_path}")
        s3.download_file(S3_BUCKET, f"{SCRIPTS_PREFIX}{script_name}", local_path)
        print(f"✅ Successfully downloaded {script_name}")
    except Exception as e:
        print(f"❌ Failed to download {script_name}: {str(e)}")
        raise

    # Ensure the script is executable and set PYTHONPATH to include /tmp
    env = os.environ.copy()
    env['PYTHONPATH'] = '/tmp:' + env.get('PYTHONPATH', '')
    
    cmd = ["python3", local_path]
    if args:
        cmd.extend(args)

    print(f"🚀 Running: {' '.join(cmd)}")
    print(f"📍 Working directory: {os.getcwd()}")
    print(f"🐍 Python path: {env.get('PYTHONPATH', 'Not set')}")
    
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd='/tmp')
    
    print("📄 STDOUT:")
    print(result.stdout)
    
    if result.stderr:
        print("⚠️ STDERR:")
        print(result.stderr)
    
    if result.returncode != 0:
        print(f"🔥 {script_name} script failed with return code {result.returncode}!")
        raise Exception(f"Script {script_name} failed")
    else:
        print(f"✅ {script_name} completed successfully")

# 📦 Setup task to download dependencies
def setup_dependencies():
    """Download all required dependencies before running other tasks"""
    download_dependencies()

# 🧼 Preprocessing with dependency check
def preprocess(**kwargs):
    s3_key = kwargs['ti'].xcom_pull(task_ids='get_latest_csv_path')
    if not s3_key:
        raise ValueError("❌ No S3 key found from upstream task")
    print(f"🔍 Processing file: {s3_key}")
    run_script_from_s3("preprocess.py", ["--s3_key", s3_key])

# 🎯 Enhanced train model with data validation
def train_model(**kwargs):
    print("🎯 Starting model training...")
    
    # Pre-training data validation and cleaning
    try:
        print("🔍 Pre-training data validation...")
        
        # Check if preprocessed data exists
        x_path = '/usr/local/airflow/tmp/X.csv'
        y_path = '/usr/local/airflow/tmp/y.csv'
        
        if not os.path.exists(x_path) or not os.path.exists(y_path):
            raise ValueError("❌ Preprocessed data files not found!")
        
        # Load the preprocessed data
        X = pd.read_csv(x_path)
        y = pd.read_csv(y_path)
        
        print(f"📊 Data loaded - X shape: {X.shape}, y shape: {y.shape}")
        
        # Check for NaN values
        x_nan_count = X.isnull().sum().sum()
        y_nan_count = y.isnull().sum()
        
        print(f"📈 Data quality check:")
        print(f"   - X NaN count: {x_nan_count}")
        print(f"   - y NaN count: {y_nan_count}")
        
        # If NaN values exist, clean them
        if x_nan_count > 0 or y_nan_count > 0:
            print("⚠️ Found NaN values, applying emergency cleaning...")
            
            # Handle numeric columns
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                # Fill with median
                X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
                # Replace any remaining NaN with 0
                X[numeric_cols] = X[numeric_cols].fillna(0)
                # Replace infinite values
                X[numeric_cols] = X[numeric_cols].replace([np.inf, -np.inf], 0)
            
            # Handle categorical columns
            object_cols = X.select_dtypes(include='object').columns
            if len(object_cols) > 0:
                X[object_cols] = X[object_cols].fillna('Unknown')
            
            # Handle target variable
            if y.isnull().any().any():
                # For classification, use mode
                y = y.fillna(y.mode().iloc[0])
            
            # Save cleaned data back
            X.to_csv(x_path, index=False)
            y.to_csv(y_path, index=False)
            
            print("✅ Emergency cleaning completed")
            print(f"   - X NaN count after cleaning: {X.isnull().sum().sum()}")
            print(f"   - y NaN count after cleaning: {y.isnull().sum()}")
        
        # Additional validation
        print(f"📊 Final data validation:")
        print(f"   - X dtypes: {X.dtypes.value_counts().to_dict()}")
        print(f"   - y unique values: {y.nunique()}")
        print(f"   - y value counts: {y.value_counts().to_dict()}")
        
    except Exception as e:
        print(f"⚠️ Pre-training validation failed: {str(e)}")
        # Continue with training anyway
    
    # Run the training script
    run_script_from_s3("train.py")

# 📈 Enhanced drift check with better error handling
def drift_decision(**kwargs):
    s3_key = kwargs['ti'].xcom_pull(task_ids='get_latest_csv_path')
    if not s3_key:
        raise ValueError("❌ No S3 key found from upstream task")
    
    print(f"📈 Running drift check on: {s3_key}")
    
    # Download drift check script
    s3 = boto3.client('s3')
    local_path = "/tmp/drift_check.py"

    try:
        print("📥 Downloading drift_check.py...")
        s3.download_file(S3_BUCKET, f"{SCRIPTS_PREFIX}drift_check.py", local_path)
    except Exception as e:
        print(f"❌ Failed to download drift_check.py: {str(e)}")
        # If drift check script is not available, continue with deployment
        print("⚠️ Drift check unavailable, proceeding with deployment")
        return "deploy_flask"

    # Set up environment
    env = os.environ.copy()
    env['PYTHONPATH'] = '/tmp:' + env.get('PYTHONPATH', '')
    
    result = subprocess.run(
        ["python3", local_path, "--s3_key", s3_key],
        capture_output=True, text=True, env=env, cwd='/tmp'
    )
    
    print("🔍 Drift check output:")
    print(result.stdout)
    
    if result.stderr:
        print("⚠️ Drift check errors:")
        print(result.stderr)
    
    if result.returncode != 0:
        print("⚠️ Drift check script failed, proceeding with deployment anyway")
        return "deploy_flask"

    # Check for drift in output
    output_lower = result.stdout.lower()
    if "drift" in output_lower or "significant" in output_lower:
        print("🚨 Drift detected!")
        return "alert_drift"
    else:
        print("✅ No significant drift detected")
        return "deploy_flask"

# 🚀 Deploy Flask app
def deploy_flask(**kwargs):
    print("🚀 Deploying Flask application...")
    run_script_from_s3("deploy_flask.py")

# 🤖 Enhanced make predictions with data validation
def make_prediction(**kwargs):
    print("🤖 Running predictions...")
    
    # Pre-prediction validation
    try:
        print("🔍 Pre-prediction validation...")
        
        # Check if model exists
        model_files = [f for f in os.listdir("/usr/local/airflow/tmp") 
                      if f.startswith("best_model_") and f.endswith(".pkl")]
        
        if not model_files:
            print("⚠️ No model file found, training might have failed")
        else:
            print(f"✅ Found model files: {model_files}")
        
        # Check if preprocessor exists
        if not os.path.exists("/usr/local/airflow/tmp/preprocessor.pkl"):
            print("⚠️ Preprocessor not found")
        else:
            print("✅ Preprocessor found")
            
    except Exception as e:
        print(f"⚠️ Pre-prediction validation error: {str(e)}")
    
    run_script_from_s3("predict.py")

# 🚨 Enhanced drift alert
def alert_drift(**kwargs):
    print("🚨 DRIFT ALERT: Significant data drift detected!")
    print("📊 Recommended actions:")
    print("   1. Review the new data for quality issues")
    print("   2. Consider retraining the model")
    print("   3. Monitor prediction performance closely")
    print("   4. Check data collection processes")
    print("🔄 Continuing with previous model for predictions...")

# 📊 Enhanced summarize pipeline results
def summarize_pipeline_results(**kwargs):
    """Summarize the results of the entire pipeline"""
    try:
        # Check if training results exist
        results_path = "/usr/local/airflow/tmp/training_results.json"
        if os.path.exists(results_path):
            import json
            with open(results_path, 'r') as f:
                results = json.load(f)
            
            print("🎯 PIPELINE SUMMARY:")
            print("=" * 50)
            print("📈 Model Performance Results:")
            
            best_model = None
            best_f1 = 0
            
            for model_name, metrics in results.items():
                acc = metrics.get('accuracy', 0)
                f1 = metrics.get('f1_score', 0)
                print(f"   {model_name}:")
                print(f"     - Accuracy: {acc:.4f}")
                print(f"     - F1 Score: {f1:.4f}")
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_model = model_name
            
            print(f"\n🏆 Best Model: {best_model} (F1: {best_f1:.4f})")
            
            # Check what files were created
            tmp_files = [f for f in os.listdir("/usr/local/airflow/tmp") if f.endswith(('.pkl', '.csv', '.json'))]
            print(f"\n📁 Generated Files: {tmp_files}")
            
        else:
            print("⚠️ No training results found - pipeline may have failed")
            
        # Check for data quality issues log
        try:
            x_path = '/usr/local/airflow/tmp/X.csv'
            y_path = '/usr/local/airflow/tmp/y.csv'
            
            if os.path.exists(x_path) and os.path.exists(y_path):
                X = pd.read_csv(x_path)
                y = pd.read_csv(y_path)
                
                print(f"\n📊 Final Data Quality:")
                print(f"   - X shape: {X.shape}")
                print(f"   - X NaN values: {X.isnull().sum().sum()}")
                print(f"   - y shape: {y.shape}")
                print(f"   - y NaN values: {y.isnull().sum()}")
                
        except Exception as e:
            print(f"⚠️ Could not check final data quality: {str(e)}")
            
        # Check drift results if available
        drift_context = kwargs.get('ti')
        if drift_context:
            try:
                drift_result = drift_context.xcom_pull(task_ids='check_data_drift')
                if drift_result:
                    print(f"\n📊 Drift Check Result: {drift_result}")
            except:
                pass
                
        print("=" * 50)
        
    except Exception as e:
        print(f"⚠️ Error summarizing results: {e}")

# 🧹 Enhanced cleanup task
def cleanup_temp_files(**kwargs):
    """Clean up temporary files with better reporting"""
    temp_files = [
        '/tmp/utils.py', '/tmp/preprocess.py', '/tmp/train.py', 
        '/tmp/drift_check.py', '/tmp/deploy_flask.py', '/tmp/predict.py'
    ]
    
    # Also check for any additional temp files
    try:
        all_tmp_files = [f"/tmp/{f}" for f in os.listdir("/tmp") if f.endswith('.py')]
        temp_files.extend([f for f in all_tmp_files if f not in temp_files])
    except:
        pass
    
    cleaned = 0
    for file_path in temp_files:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"🗑️ Cleaned up {file_path}")
                cleaned += 1
        except Exception as e:
            print(f"⚠️ Could not clean up {file_path}: {str(e)}")
    
    print(f"✅ Cleanup completed - {cleaned} files removed")

# --------------------------------------------
# 📌 DAG Definition
# --------------------------------------------
with DAG(
    dag_id="lead_scoring_pipeline_nocopy",
    default_args=default_args,
    description="Enhanced ML pipeline with robust error handling and result summary",
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
    tags=["lead_scoring", "mlops", "drift", "enhanced"],
    max_active_runs=1,  # Prevent concurrent runs
) as dag:

    # 📡 Wait for new data
    wait_for_file = S3KeySensor(
        task_id="wait_for_new_csv",
        bucket_key="new_data/*.csv",
        bucket_name=S3_BUCKET,
        wildcard_match=True,
        aws_conn_id="aws_default",
        poke_interval=30,
        timeout=600,  # Increased timeout
        mode="poke",
        soft_fail=False,
    )

    # 🔍 Get latest file path
    get_file_path = PythonOperator(
        task_id="get_latest_csv_path",
        python_callable=get_latest_file_key,
        retries=2,
    )

    # 📦 Setup dependencies (NEW TASK)
    setup_deps = PythonOperator(
        task_id="setup_dependencies",
        python_callable=setup_dependencies,
        retries=2,
    )

    # 🧼 Data preprocessing
    preprocess_task = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess,
        retries=1,
    )

    # 🎯 Model training
    model_training = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
        retries=1,
    )

    # 📈 Drift detection with branching
    check_drift = BranchPythonOperator(
        task_id="check_data_drift",
        python_callable=drift_decision,
        retries=1,
    )

    # 🚀 Flask app deployment
    deploy_app = PythonOperator(
        task_id="deploy_flask",
        python_callable=deploy_flask,
        retries=1,
    )

    # 🤖 Prediction generation
    predict_task = PythonOperator(
        task_id="predict",
        python_callable=make_prediction,
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
        retries=1,
    )

    # 🚨 Drift alert handling
    drift_alert = PythonOperator(
        task_id="alert_drift",
        python_callable=alert_drift,
        trigger_rule=TriggerRule.NONE_FAILED,
        retries=0,
    )

    # 📊 Results summary task
    summarize_results = PythonOperator(
        task_id="summarize_results",
        python_callable=summarize_pipeline_results,
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
        retries=0,
    )

    # 🧹 Cleanup (optional final task)
    cleanup = PythonOperator(
        task_id="cleanup_temp_files",
        python_callable=cleanup_temp_files,
        trigger_rule=TriggerRule.ALL_DONE,  # Run regardless of success/failure
        retries=0,
    )

    # 📊 DAG Structure
    wait_for_file >> get_file_path >> setup_deps >> preprocess_task >> model_training >> check_drift
    check_drift >> [deploy_app, drift_alert]
    deploy_app >> predict_task >> summarize_results
    drift_alert >> predict_task
    summarize_results >> cleanup