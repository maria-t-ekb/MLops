from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

from train_model import train


AIRFLOW_HOME = Path(os.environ.get("AIRFLOW_HOME", ".")).resolve()
DATA_DIR = AIRFLOW_HOME / "data" / "climate_energy"
DATA_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_CSV = DATA_DIR / "climate_energy.csv"
RAW_CSV = DATA_DIR / "raw_climate_energy.csv"
CLEAR_CSV = DATA_DIR / "df_clear.csv"


def normalize_columns(columns):
    return [
        c.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
        for c in columns
    ]


def download_data():
    df = pd.read_csv(SOURCE_CSV, delimiter=",")
    df.columns = normalize_columns(df.columns)

    df.to_csv(RAW_CSV, index=False)
    print("Saved raw dataset to:", RAW_CSV)
    print("df shape:", df.shape)
    print("columns:", list(df.columns))
    return True


def clear_data():
    df = pd.read_csv(RAW_CSV)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["day"] = df["date"].dt.day
        df = df.drop(columns=["date"])

    if "energy_consumption" not in df.columns:
        raise ValueError(
            f"Target column 'energy_consumption' not found. Available columns: {list(df.columns)}"
        )

    cat_columns = [col for col in ["country"] if col in df.columns]

    for col in df.columns:
        if col not in cat_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.drop_duplicates()
    df = df.dropna(subset=["energy_consumption"])

    df = df[df["energy_consumption"] >= 0]

    for col in df.columns:
        if col not in cat_columns:
            df[col] = df[col].fillna(df[col].median())

    for col in cat_columns:
        df[col] = df[col].fillna("unknown").astype(str)

    if cat_columns:
        ordinal = OrdinalEncoder()
        df[cat_columns] = ordinal.fit_transform(df[cat_columns])

    df = df.reset_index(drop=True)
    df.to_csv(CLEAR_CSV, index=False)

    print("Saved cleaned dataset to:", CLEAR_CSV)
    print("df_clear shape:", df.shape)
    return True


dag_climate = DAG(
    dag_id="train_climate_energy_pipe",
    start_date=datetime(2025, 2, 3),
    max_active_tasks=4,
    schedule=None,
    max_active_runs=1,
    catchup=False,
)

download_task = PythonOperator(
    python_callable=download_data,
    task_id="download_climate_energy",
    dag=dag_climate,
)

clear_task = PythonOperator(
    python_callable=clear_data,
    task_id="clear_climate_energy",
    dag=dag_climate,
)

train_task = PythonOperator(
    python_callable=train,
    task_id="train_climate_energy",
    dag=dag_climate,
)

download_task >> clear_task >> train_task