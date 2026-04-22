import json
import os
import uuid
from io import BytesIO
from datetime import datetime, timezone

import joblib
import pandas as pd
from minio import Minio
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")

GOLD_BUCKET = os.getenv("GOLD_BUCKET", "fraud-lake")
GOLD_PREFIX = os.getenv("GOLD_PREFIX", "gold/training_features/")
GOLD_VERSION = os.getenv("GOLD_VERSION", "v1")

ARTIFACTS_BUCKET = os.getenv("ARTIFACTS_BUCKET", "ml-artifacts")
MODEL_PREFIX = os.getenv("MODEL_PREFIX", "models/fraud_model/")
METRICS_PREFIX = os.getenv("METRICS_PREFIX", "metrics/fraud_model/")
REPORTS_PREFIX = os.getenv("REPORTS_PREFIX", "reports/fraud_model/")

if not MINIO_ACCESS_KEY or not MINIO_SECRET_KEY:
    raise ValueError("Missing MINIO_ACCESS_KEY or MINIO_SECRET_KEY")


FEATURE_COLUMNS = [
    "Time", "Amount",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
    "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28",
]
LABEL_COLUMN = "Class"

MODEL_TYPE = "logistic_regression"
USE_SCALER = True
LOGREG_SOLVER = "lbfgs"
LOGREG_MAX_ITER = 5000
LOGREG_CLASS_WEIGHT = "balanced"
LOGREG_RANDOM_STATE = 42


def get_minio_client() -> Minio:
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False,
    )


def ensure_bucket_exists(client: Minio, bucket_name: str) -> None:
    if not client.bucket_exists(bucket_name):
        client.make_bucket(bucket_name)


def list_gold_objects(client: Minio) -> list[str]:
    prefix = f"{GOLD_PREFIX}version={GOLD_VERSION}/"
    objects = client.list_objects(
        GOLD_BUCKET,
        prefix=prefix,
        recursive=True,
    )
    return sorted(
        obj.object_name
        for obj in objects
        if obj.object_name.endswith(".parquet")
    )


def read_parquet_object(client: Minio, bucket_name: str, object_name: str) -> pd.DataFrame:
    response = client.get_object(bucket_name, object_name)
    try:
        raw = response.read()
        return pd.read_parquet(BytesIO(raw), engine="pyarrow")
    finally:
        response.close()
        response.release_conn()


def load_gold_dataframe(client: Minio) -> pd.DataFrame:
    object_names = list_gold_objects(client)
    if not object_names:
        raise ValueError("No Gold parquet files found")

    frames = []
    for object_name in object_names:
        df = read_parquet_object(client, GOLD_BUCKET, object_name)
        if not df.empty:
            frames.append(df)

    if not frames:
        raise ValueError("Gold parquet files exist but produced no rows")

    return pd.concat(frames, ignore_index=True)


def validate_gold_dataframe(df: pd.DataFrame) -> None:
    required = FEATURE_COLUMNS + [LABEL_COLUMN, "split", "dataset_version"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in Gold data: {missing}")


def compute_metrics(y_true, y_pred, y_prob=None) -> dict:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if y_prob is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None

    return metrics


def upload_bytes(client: Minio, bucket_name: str, object_name: str, raw: bytes, content_type: str):
    client.put_object(
        bucket_name=bucket_name,
        object_name=object_name,
        data=BytesIO(raw),
        length=len(raw),
        content_type=content_type,
    )


def main():
    client = get_minio_client()
    ensure_bucket_exists(client, ARTIFACTS_BUCKET)

    df = load_gold_dataframe(client)
    validate_gold_dataframe(df)

    print(f"[TRAIN] Total Gold rows: {len(df)}")
    print(f"[TRAIN] Split counts:\n{df['split'].value_counts(dropna=False)}")

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("One of the dataset splits is empty")

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[LABEL_COLUMN].astype(int)

    X_val = val_df[FEATURE_COLUMNS]
    y_val = val_df[LABEL_COLUMN].astype(int)

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[LABEL_COLUMN].astype(int)


    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            solver=LOGREG_SOLVER,
            max_iter=LOGREG_MAX_ITER,
            class_weight=LOGREG_CLASS_WEIGHT,
            random_state=LOGREG_RANDOM_STATE,
        )),
    ])


    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    val_prob = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else None
    test_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    val_metrics = compute_metrics(y_val, val_pred, val_prob)
    test_metrics = compute_metrics(y_test, test_pred, test_prob)

    report = {
        "val_classification_report": classification_report(y_val, val_pred, output_dict=True, zero_division=0),
        "test_classification_report": classification_report(y_test, test_pred, output_dict=True, zero_division=0),
        "val_confusion_matrix": confusion_matrix(y_val, val_pred).tolist(),
        "test_confusion_matrix": confusion_matrix(y_test, test_pred).tolist(),
    }

    run_id = uuid.uuid4().hex[:8]
    now = datetime.now(timezone.utc)

    model_object = (
        f"{MODEL_PREFIX}version={GOLD_VERSION}/"
        f"run={run_id}/model.joblib"
    )
    metrics_object = (
        f"{METRICS_PREFIX}version={GOLD_VERSION}/"
        f"run={run_id}/metrics.json"
    )
    report_object = (
        f"{REPORTS_PREFIX}version={GOLD_VERSION}/"
        f"run={run_id}/report.json"
    )

    model_buffer = BytesIO()
    joblib.dump(model, model_buffer)
    model_raw = model_buffer.getvalue()

    metrics_payload = {
        "run_id": run_id,
        "trained_at": now.isoformat(),
        "gold_version": GOLD_VERSION,
        "model_metadata": {
            "model_type": MODEL_TYPE,
            "use_scaler": USE_SCALER,
            "solver": LOGREG_SOLVER,
            "max_iter": LOGREG_MAX_ITER,
            "class_weight": LOGREG_CLASS_WEIGHT,
            "random_state": LOGREG_RANDOM_STATE,
        },
        "data_metadata": {
            "feature_columns": FEATURE_COLUMNS,
            "label_column": LABEL_COLUMN,
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
        },
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    upload_bytes(
        client,
        ARTIFACTS_BUCKET,
        model_object,
        model_raw,
        "application/octet-stream",
    )
    upload_bytes(
        client,
        ARTIFACTS_BUCKET,
        metrics_object,
        json.dumps(metrics_payload, indent=2).encode("utf-8"),
        "application/json",
    )
    upload_bytes(
        client,
        ARTIFACTS_BUCKET,
        report_object,
        json.dumps(report, indent=2).encode("utf-8"),
        "application/json",
    )

    print(f"[TRAIN] Uploaded model: {ARTIFACTS_BUCKET}/{model_object}")
    print(f"[TRAIN] Uploaded metrics: {ARTIFACTS_BUCKET}/{metrics_object}")
    print(f"[TRAIN] Uploaded report: {ARTIFACTS_BUCKET}/{report_object}")
    print(f"[TRAIN] Validation metrics: {val_metrics}")
    print(f"[TRAIN] Test metrics: {test_metrics}")


if __name__ == "__main__":
    main()