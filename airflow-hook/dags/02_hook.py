from datetime import datetime
import json
import logging
import os

from airflow.sdk import DAG
from airflow.providers.standard.operators.python import PythonOperator
from hooks import CarsHook  # ← импортируем из plugins/hooks/

# Пути к файлам внутри контейнера Airflow согласно заданию
RAW_DATA_PATH = "/data/raw/cars_raw.json"
CLEAN_DATA_PATH = "/data/cleaned/cars_cleaned.json"

def _fetch_cars(conn_id: str, templates_dict: dict, batch_size: int = 1000, **_):
    logger = logging.getLogger(__name__)
    output_path = templates_dict["output_path"]

    logger.info("Fetching all cars from the API...")
    hook = CarsHook(conn_id=conn_id)
    cars = list(hook.get_cars(batch_size=batch_size))
    logger.info(f"Fetched {len(cars)} car records")

    # Убедимся, что директория существует
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cars, f, ensure_ascii=False, indent=4)

    logger.info(f"Saved raw cars to {output_path}")

def _clean_cars_data(**_):
    import pandas as pd

    logger = logging.getLogger(__name__)

    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"Сырые данные не найдены по пути: {RAW_DATA_PATH}")

    logger.info(f"Loading raw data from {RAW_DATA_PATH}")
    df = pd.read_json(RAW_DATA_PATH)
    initial_shape = df.shape

    df = df.drop_duplicates()
    df = df.dropna()
    logger.info(f"Removed duplicates and gaps. Rows changed from {initial_shape[0]} to {df.shape[0]}")

    categorical_cols = ["Fuel_type", "Transmission"]
    for col in categorical_cols:
        if col in df.columns:
            logger.info(f"Converting categorical column: {col}")
            df[col] = df[col].astype("category").cat.codes
        else:
            logger.warning(f"Column {col} not found in dataset")

    os.makedirs(os.path.dirname(CLEAN_DATA_PATH), exist_ok=True)
    df.to_json(CLEAN_DATA_PATH, orient="records", force_ascii=False, indent=4)
    logger.info(f"Saved cleaned data to {CLEAN_DATA_PATH}")

with DAG(
    dag_id="02_hook",
    description="Fetches and processes car data from the custom API using a custom hook.",
    start_date=datetime(2026, 2, 3),
    schedule="@daily",
    catchup=False,
    max_active_runs=1,
) as dag:

    fetch_cars_task = PythonOperator(
        task_id="fetch_cars",
        python_callable=_fetch_cars,
        op_kwargs={"conn_id": "carsapi"},
        templates_dict={
            "output_path": RAW_DATA_PATH,
        },
    )

    clean_cars_task = PythonOperator(
        task_id="clean_cars_data",
        python_callable=_clean_cars_data,
    )

    fetch_cars_task >> clean_cars_task

