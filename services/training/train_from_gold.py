import json
import os
import time
from io import BytesIO
from datetime import datetime, timezone

import joblib
import pandas as pd
from minio import Minio
from minio.error import S3Error
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

TRAIN_EVERY_N_NEW_ROWS = int(os.getenv("TRAIN_EVERY_N_NEW_ROWS", "1000"))
TRAIN_POLL_SECONDS = int(os.getenv("TRAIN_POLL_SECONDS", "30"))
TRAINING_STATE_OBJECT = os.getenv(
    "TRAINING_STATE_OBJECT",
    "_state/training/last_training_state.json",
)

if not MINIO_ACCESS_KEY or not MINIO_SECRET_KEY:
    raise ValueError("Missing MINIO_ACCESS_KEY or MINIO_SECRET_KEY")


FEATURE_COLUMNS = [
    "Time", "Amount",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
    "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28",
]
LABEL_COLUMN = "Class"
ID_COLUMN = "transaction_id"

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


def ensure_bucket_exists(client, bucket_name: str) -> None:
    try:
        if not client.bucket_exists(bucket_name):
            client.make_bucket(bucket_name)
    except S3Error as e:
        if e.code in {"BucketAlreadyOwnedByYou", "BucketAlreadyExists"}:
            return
        raise

def is_missing_object_error(error: S3Error) -> bool:
    return error.code in {"NoSuchKey", "NoSuchObject", "NoSuchBucket"}


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
    required = FEATURE_COLUMNS + [LABEL_COLUMN, ID_COLUMN, "split", "dataset_version"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in Gold data: {missing}")


def validate_trainable_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("One of the dataset splits is empty")

    train_classes = set(train_df[LABEL_COLUMN].astype(int).unique())
    val_classes = set(val_df[LABEL_COLUMN].astype(int).unique())
    test_classes = set(test_df[LABEL_COLUMN].astype(int).unique())

    if train_classes != {0, 1}:
        raise ValueError(
            f"Training split is not trainable yet. Classes found: {sorted(train_classes)}"
        )

    if val_classes != {0, 1}:
        raise ValueError(
            f"Validation split is not evaluable yet. Classes found: {sorted(val_classes)}"
        )

    if test_classes != {0, 1}:
        raise ValueError(
            f"Test split is not evaluable yet. Classes found: {sorted(test_classes)}"
        )


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
    ensure_bucket_exists(client, bucket_name)
    client.put_object(
        bucket_name=bucket_name,
        object_name=object_name,
        data=BytesIO(raw),
        length=len(raw),
        content_type=content_type,
    )


def object_exists(client: Minio, bucket_name: str, object_name: str) -> bool:
    try:
        client.stat_object(bucket_name, object_name)
        return True
    except S3Error as e:
        if is_missing_object_error(e):
            return False
        raise


def load_joblib_object(client: Minio, bucket_name: str, object_name: str):
    response = client.get_object(bucket_name, object_name)
    try:
        raw = response.read()
        return joblib.load(BytesIO(raw))
    finally:
        response.close()
        response.release_conn()


def evaluate_model(model, X, y) -> dict:
    pred = model.predict(X)
    prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None
    return compute_metrics(y, pred, prob)


def should_promote(candidate_metrics: dict, current_metrics: dict | None) -> tuple[bool, str]:
    promotion_min_delta = float(os.getenv("PROMOTION_MIN_DELTA", "0.005"))
    recall_tolerance = float(os.getenv("RECALL_TOLERANCE", "0.01"))

    if current_metrics is None:
        return True, "No existing latest model"

    cand_f1 = candidate_metrics["f1"]
    curr_f1 = current_metrics["f1"]
    cand_recall = candidate_metrics["recall"]
    curr_recall = current_metrics["recall"]

    if cand_f1 <= curr_f1 + promotion_min_delta:
        return (
            False,
            f"Candidate f1={cand_f1:.6f} not greater than current f1={curr_f1:.6f} "
            f"+ min_delta={promotion_min_delta:.6f}",
        )

    if cand_recall < curr_recall - recall_tolerance:
        return (
            False,
            f"Candidate recall={cand_recall:.6f} is worse than current recall={curr_recall:.6f} "
            f"beyond tolerance={recall_tolerance:.6f}",
        )

    return True, "Candidate outperformed current latest"


def load_training_state(client: Minio) -> dict:
    try:
        response = client.get_object(ARTIFACTS_BUCKET, TRAINING_STATE_OBJECT)
        try:
            return json.loads(response.read().decode("utf-8"))
        finally:
            response.close()
            response.release_conn()
    except S3Error as e:
        if is_missing_object_error(e):
            return {
                "last_trained_row_count": 0,
                "last_training_at": None,
                "last_run_id": None,
                "last_promoted": None,
            }
        raise


def save_training_state(client: Minio, state: dict) -> None:
    raw = json.dumps(state, indent=2).encode("utf-8")
    upload_bytes(
        client,
        ARTIFACTS_BUCKET,
        TRAINING_STATE_OBJECT,
        raw,
        "application/json",
    )


def should_train_based_on_new_rows(client: Minio, df: pd.DataFrame) -> tuple[bool, dict]:
    ensure_bucket_exists(client, ARTIFACTS_BUCKET)

    training_state = load_training_state(client)
    last_trained_row_count = int(training_state.get("last_trained_row_count", 0))

    current_row_count = int(df[ID_COLUMN].nunique())
    new_rows = current_row_count - last_trained_row_count

    metadata = {
        "current_row_count": current_row_count,
        "last_trained_row_count": last_trained_row_count,
        "new_rows": new_rows,
        "required_new_rows": TRAIN_EVERY_N_NEW_ROWS,
    }

    if last_trained_row_count == 0:
        return True, metadata

    if new_rows >= TRAIN_EVERY_N_NEW_ROWS:
        return True, metadata

    return False, metadata


def main():
    client = get_minio_client()
    ensure_bucket_exists(client, GOLD_BUCKET)
    ensure_bucket_exists(client, ARTIFACTS_BUCKET)

    df = load_gold_dataframe(client)
    validate_gold_dataframe(df)

    should_train, row_metadata = should_train_based_on_new_rows(client, df)
    if not should_train:
        raise ValueError(
            "Not enough new rows for retraining. "
            f"current_rows={row_metadata['current_row_count']}, "
            f"last_trained_row_count={row_metadata['last_trained_row_count']}, "
            f"new_rows={row_metadata['new_rows']}, "
            f"required={row_metadata['required_new_rows']}"
        )

    print(f"[TRAIN] Total Gold rows: {len(df)}")
    print(f"[TRAIN] Unique Gold transactions: {row_metadata['current_row_count']}")
    print(f"[TRAIN] New rows since last training: {row_metadata['new_rows']}")
    print(f"[TRAIN] Split counts:\n{df['split'].value_counts(dropna=False)}")
    print(f"[TRAIN] Class counts:\n{df[LABEL_COLUMN].value_counts(dropna=False)}")

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    validate_trainable_splits(train_df, val_df, test_df)

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

    latest_model_object = f"{MODEL_PREFIX}latest/model.joblib"

    current_metrics = None
    if object_exists(client, ARTIFACTS_BUCKET, latest_model_object):
        current_model = load_joblib_object(client, ARTIFACTS_BUCKET, latest_model_object)
        current_metrics = evaluate_model(current_model, X_val, y_val)
        print(f"[TRAIN] Current latest validation metrics: {current_metrics}")
    else:
        print("[TRAIN] No current latest model found. Candidate will be promoted automatically.")

    promote, promotion_reason = should_promote(val_metrics, current_metrics)
    print(f"[TRAIN] Promotion decision: {promote} | reason: {promotion_reason}")

    report = {
        "val_classification_report": classification_report(y_val, val_pred, output_dict=True, zero_division=0),
        "test_classification_report": classification_report(y_test, test_pred, output_dict=True, zero_division=0),
        "val_confusion_matrix": confusion_matrix(y_val, val_pred).tolist(),
        "test_confusion_matrix": confusion_matrix(y_test, test_pred).tolist(),
    }

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
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
            "row_metadata": row_metadata,
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "train_class_counts": {
                str(k): int(v) for k, v in train_df[LABEL_COLUMN].value_counts().to_dict().items()
            },
            "val_class_counts": {
                str(k): int(v) for k, v in val_df[LABEL_COLUMN].value_counts().to_dict().items()
            },
            "test_class_counts": {
                str(k): int(v) for k, v in test_df[LABEL_COLUMN].value_counts().to_dict().items()
            },
        },
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "promotion": {
            "promoted": promote,
            "reason": promotion_reason,
            "candidate_val_metrics": val_metrics,
            "current_val_metrics": current_metrics,
            "serving_alias": latest_model_object,
        },
    }

    upload_bytes(
        client,
        ARTIFACTS_BUCKET,
        model_object,
        model_raw,
        "application/octet-stream",
    )

    if promote:
        upload_bytes(
            client,
            ARTIFACTS_BUCKET,
            latest_model_object,
            model_raw,
            "application/octet-stream",
        )
        print(f"[TRAIN] Uploaded latest model: {ARTIFACTS_BUCKET}/{latest_model_object}")
    else:
        print("[TRAIN] Candidate model was NOT promoted to latest")

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

    save_training_state(
        client,
        {
            "last_trained_row_count": row_metadata["current_row_count"],
            "last_training_at": now.isoformat(),
            "last_run_id": run_id,
            "last_promoted": promote,
            "latest_model_object": latest_model_object if promote else None,
            "versioned_model_object": model_object,
            "metrics_object": metrics_object,
            "report_object": report_object,
        },
    )

    print(f"[TRAIN] Uploaded model: {ARTIFACTS_BUCKET}/{model_object}")
    print(f"[TRAIN] Uploaded metrics: {ARTIFACTS_BUCKET}/{metrics_object}")
    print(f"[TRAIN] Uploaded report: {ARTIFACTS_BUCKET}/{report_object}")
    print(f"[TRAIN] Updated training state: {ARTIFACTS_BUCKET}/{TRAINING_STATE_OBJECT}")
    print(f"[TRAIN] Validation metrics: {val_metrics}")
    print(f"[TRAIN] Test metrics: {test_metrics}")


def run_forever():
    retry_messages = [
        "No Gold parquet files found",
        "Gold parquet files exist but produced no rows",
        "One of the dataset splits is empty",
        "Training split is not trainable yet",
        "Validation split is not evaluable yet",
        "Test split is not evaluable yet",
        "Not enough new rows for retraining",
    ]

    while True:
        try:
            main()
        except ValueError as e:
            if any(msg in str(e) for msg in retry_messages):
                print(f"[TRAIN] Waiting: {e}. Retrying in {TRAIN_POLL_SECONDS} seconds...")
                time.sleep(TRAIN_POLL_SECONDS)
                continue
            raise
        except S3Error as e:
            if is_missing_object_error(e):
                print(f"[TRAIN] MinIO dependency not ready yet: {e}. Retrying in {TRAIN_POLL_SECONDS} seconds...")
                time.sleep(TRAIN_POLL_SECONDS)
                continue
            raise

        print(f"[TRAIN] Training cycle finished. Checking again in {TRAIN_POLL_SECONDS} seconds...")
        time.sleep(TRAIN_POLL_SECONDS)


if __name__ == "__main__":
    run_forever()
