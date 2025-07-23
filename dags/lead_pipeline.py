from airflow import DAG
from airflow.providers.amazon.aws.sensors.s3 import S3KeySensor
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.utils.dates import days_ago
from airflow.utils.trigger_rule import TriggerRule
from datetime import timedelta
import boto3
import subprocess
import os

# === DAG Defaults ===
default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

BUCKET = "storage21julybucket"
SCRIPT_PREFIX = "scripts/"
REFERENCE_DATA = "data/train.csv"  # used for drift detection

# === Download and Run Python Scripts from S3 ===
def download_and_run(script, args=None):
    s3 = boto3.client('s3')

    # Download utils.py (common dependency)
    try:
        s3.download_file(BUCKET, f"{SCRIPT_PREFIX}utils.py", "/tmp/utils.py")
        print("✅ utils.py downloaded")
    except Exception as e:
        print(f"❌ Failed to download utils.py: {e}")
        raise

    # Download the script
    script_path = f"/tmp/{script}"
    try:
        s3.download_file(BUCKET, f"{SCRIPT_PREFIX}{script}", script_path)
        print(f"✅ {script} downloaded")
    except Exception as e:
        print(f"❌ Failed to download {script}: {e}")
        raise

    env = os.environ.copy()
    env['PYTHONPATH'] = '/tmp:' + env.get('PYTHONPATH', '')

    cmd = ["python3", script_path]
    if args:
        cmd += args

    result = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd="/tmp")
    print("STDOUT:\n", result.stdout)
    print("STDERR:\n", result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"Script {script} failed.")

# === Task: Get Latest CSV from S3 ===
def get_latest_csv_key(**kwargs):
    s3 = boto3.client("s3")
    response = s3.list_objects_v2(Bucket=BUCKET, Prefix="new_data/")
    csvs = [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith(".csv")]
    if not csvs:
        raise ValueError("❌ No CSV files found in new_data/")
    latest = sorted(response['Contents'], key=lambda x: x['LastModified'], reverse=True)[0]['Key']
    print(f"✅ Latest file: {latest}")
    return latest

# === Task: Preprocess
def run_preprocess(**context):
    key = context['ti'].xcom_pull(task_ids='get_latest_csv_key')
    download_and_run("preprocess.py", ["--s3_key", key])

# === Task: Drift Check
def run_drift_check(**context):
    key = context['ti'].xcom_pull(task_ids='get_latest_csv_key')
    try:
        download_and_run("drift_check.py", ["--s3_key", key, "--ref_key", REFERENCE_DATA])
    except Exception as e:
        print(f"⚠️ Drift check failed: {e}")
        return "train_model"  # fallback

    try:
        with open("/tmp/drift_result.txt") as f:
            result = f.read().strip().lower()
            if result == "drift":
                print("🚨 Drift Detected")
                return "train_model"
            else:
                print("✅ No Drift Detected")
                return "no_drift_continue"
    except Exception as e:
        print(f"⚠️ Could not read drift_result.txt: {e}")
        return "train_model"

# === Task: Train Model
def train_model():
    download_and_run("train.py")

# === Task: Reuse Previous Model
def no_drift_continue():
    print("✅ No drift. Using previous model for prediction.")

# === Task: Predict
def run_predict(**context):
    key = context['ti'].xcom_pull(task_ids='get_latest_csv_key')
    download_and_run("predict.py", ["--s3_key", key])

# === Task: Cleanup
def cleanup_tmp():
    for f in os.listdir("/tmp"):
        path = f"/tmp/{f}"
        if os.path.isfile(path):
            os.remove(path)
            print(f"🧹 Removed: {path}")

# === DAG Definition ===
with DAG(
    dag_id="lead_scoring_pipeline",
    description="Lead Scoring MLOps Pipeline: Drift Detection → Train or Reuse → Predict",
    default_args=default_args,
    start_date=days_ago(1),
    schedule_interval=None,
    catchup=False,
    tags=["mlops", "lead-scoring"]
) as dag:

    wait_for_csv = S3KeySensor(
        task_id="wait_for_new_csv",
        bucket_key="new_data/*.csv",
        bucket_name=BUCKET,
        wildcard_match=True,
        aws_conn_id="aws_default",
        poke_interval=30,
        timeout=600
    )

    get_key = PythonOperator(
        task_id="get_latest_csv_key",
        python_callable=get_latest_csv_key
    )

    preprocess = PythonOperator(
        task_id="run_preprocessing",
        python_callable=run_preprocess
    )

    check_drift = BranchPythonOperator(
        task_id="run_drift_check",
        python_callable=run_drift_check
    )

    train = PythonOperator(
        task_id="train_model",
        python_callable=train_model
    )

    no_drift = PythonOperator(
        task_id="no_drift_continue",
        python_callable=no_drift_continue
    )

    predict = PythonOperator(
        task_id="predict",
        python_callable=run_predict,
        trigger_rule=TriggerRule.ONE_SUCCESS  # ✅ Make sure it runs even if only one upstream succeeded
    )

    cleanup = PythonOperator(
        task_id="cleanup_tmp",
        python_callable=cleanup_tmp,
        trigger_rule=TriggerRule.ALL_DONE
    )

    # === DAG Flow ===
    wait_for_csv >> get_key >> preprocess >> check_drift
    check_drift >> [train, no_drift]
    train >> predict
    no_drift >> predict
    predict >> cleanup
