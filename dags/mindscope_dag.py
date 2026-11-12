from datetime import datetime
from pathlib import Path
import json
import joblib
import pandas as pd 

from airflow import DAG
from airflow.operators.python import PythonOperator

from src.models.train import train_models
from src.evaluation.evaluate import evaluate_model


# Paths

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

# Preeepare Data

def prepare_data():

    train_data = pd.read_csv(DATA_DIR / "train_cleaned.csv")
    test_data = pd.read_csv(DATA_DIR / "test_cleaned.csv")

    print(train_data.shape)
    print(test_data.shape)


def train_model():

    train_data = pd.read_csv(DATA_DIR / "train_cleaned.csv")

    X_train = train_data.drop(
        columns=["Student_ID","Mental_Health_Risk"]
    )

    y_train = train_data["Mental_Health_Risk"]


    grid_results , best_name , best_pipeline = train_models(

        X_train,y_train

    )



    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        best_pipeline,
        MODEL_DIR / "mindscope_model.joblib"
    )

    with open(REPORT_DIR / "best_model.json", "w") as file:
        json.dump({
            "best_model": best_name,
            "best_cv_f1": float(grid_results[best_name].best_score_)
        }, file, indent=4)

    print("Best model:", best_name)


# Evaaaaluate model
def run_evaluation():

        test_data = pd.read_csv(DATA_DIR / "test_cleaned.csv")

        X_test = test_data.drop(
            columns=["Student_ID", "Mental_Health_Risk"]
        )
        y_test = test_data["Mental_Health_Risk"]

        model = joblib.load(
            MODEL_DIR / "mindscope_model.joblib"
        )

        _, metrics, report, confusion_df = evaluate_model(
            model, X_test, y_test
        )

        REPORT_DIR.mkdir(parents=True, exist_ok=True)

        with open(REPORT_DIR / "metrics.json", "w") as file:
            json.dump(metrics, file, indent=4)

        confusion_df.to_csv(
            REPORT_DIR / "confusion_matrix.csv"
        )

        print("Evaluation metrics:", metrics)


with DAG(
    dag_id = "mindscope_pipeline",
    start_date = datetime(2026,2,2),
    catchup=False,
    schedule=None,
    tags=["mindscope"]
) as dag:



    prepare_task = PythonOperator(
        task_id="prepare_data",
        python_callable = prepare_data
    )

    train_task = PythonOperator(
        task_id="train_model",
        python_callable=train_model
    )

    evaluate_task = PythonOperator(
        task_id = "evaluate_model",
        python_callable=run_evaluation
    )

    prepare_task>>train_task>>evaluate_task

