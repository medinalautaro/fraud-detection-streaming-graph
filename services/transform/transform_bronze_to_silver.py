import json
import uuid
from io import BytesIO
from datetime import datetime, timezone
import os
import pandas as pd
from minio import Minio
from minio.error import S3Error
import time


MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "fraud-lake")

BRONZE_PREFIX = os.getenv("BRONZE_PREFIX", "bronze/transactions/")
SILVER_PREFIX = os.getenv("SILVER_PREFIX", "silver/transactions_clean/")
STATE_OBJECT = os.getenv(
    "STATE_OBJECT",
    "_state/bronze_to_silver/processed_files.json",
)


def get_minio_client() -> Minio:
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False,
    )


def load_state(client: Minio) -> dict:
    try:
        response = client.get_object(MINIO_BUCKET, STATE_OBJECT)
        try:
            return json.loads(response.read().decode("utf-8"))
        finally:
            response.close()
            response.release_conn()
    except S3Error as e:
        if e.code == "NoSuchKey":
            return {"processed_objects": [], "last_updated_at": None}
        raise


def save_state(client: Minio, state: dict) -> None:
    raw = json.dumps(state, ensure_ascii=False, indent=2).encode("utf-8")
    client.put_object(
        bucket_name=MINIO_BUCKET,
        object_name=STATE_OBJECT,
        data=BytesIO(raw),
        length=len(raw),
        content_type="application/json",
    )


def list_bronze_json_objects(client: Minio) -> list[str]:
    objects = client.list_objects(
        MINIO_BUCKET,
        prefix=BRONZE_PREFIX,
        recursive=True,
    )
    return sorted(
        obj.object_name
        for obj in objects
        if obj.object_name.endswith(".json")
    )


def read_json_object(client: Minio, object_name: str) -> dict:
    response = client.get_object(MINIO_BUCKET, object_name)
    try:
        return json.loads(response.read().decode("utf-8"))
    finally:
        response.close()
        response.release_conn()


def flatten_record(record: dict, source_object_name: str) -> dict:
    payload = record.get("payload", {})
    metadata = record.get("ingestion_metadata", {})

    row = dict(payload)
    row["ingested_at"] = metadata.get("ingested_at")
    row["ingestion_transaction_id"] = metadata.get("transaction_id")
    row["source_object_name"] = source_object_name
    return row


def build_incremental_dataframe(client: Minio, new_object_names: list[str]) -> pd.DataFrame:
    rows = []

    for object_name in new_object_names:
        record = read_json_object(client, object_name)
        rows.append(flatten_record(record, object_name))

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    df = df.rename(columns={
        "time": "Time",
        "amount": "Amount",
        "class": "Class",
    })

    if "transaction_id" in df.columns:
        df = df.drop_duplicates(subset=["transaction_id"], keep="last")

    return df


def upload_incremental_parquet(client: Minio, df: pd.DataFrame) -> str:
    now = datetime.now(timezone.utc)
    run_id = uuid.uuid4().hex[:8]

    object_name = (
        f"{SILVER_PREFIX}"
        f"year={now.year:04d}/"
        f"month={now.month:02d}/"
        f"day={now.day:02d}/"
        f"hour={now.hour:02d}/"
        f"part-{run_id}.parquet"
    )

    buffer = BytesIO()
    df.to_parquet(buffer, index=False, engine="pyarrow")
    buffer.seek(0)
    raw = buffer.getvalue()

    client.put_object(
        bucket_name=MINIO_BUCKET,
        object_name=object_name,
        data=BytesIO(raw),
        length=len(raw),
        content_type="application/octet-stream",
    )

    return object_name

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


def safe_load_state(client):
    try:
        return load_state(client)
    except S3Error as e:
        if is_missing_object_error(e):
            print("[TRANSFORM] No previous state found. Starting from empty state.")
            return {"processed_files": []}
        raise


def main():
    client = get_minio_client()

    ensure_bucket_exists(client, MINIO_BUCKET)
    state = safe_load_state(client)
    processed = set(state.get("processed_objects", []))

    all_bronze_objects = list_bronze_json_objects(client)
    new_objects = [obj for obj in all_bronze_objects if obj not in processed]

    print(f"[TRANSFORM] Bronze total: {len(all_bronze_objects)}")
    print(f"[TRANSFORM] Already processed: {len(processed)}")
    print(f"[TRANSFORM] New objects to process: {len(new_objects)}")

    if not new_objects:
        print("[TRANSFORM] No new bronze files found. Waiting...")
        time.sleep(5)
        return

    df = build_incremental_dataframe(client, new_objects)

    if df.empty:
        print("[TRANSFORM] No rows built from new objects")
        return

    print(f"[TRANSFORM] Incremental rows: {len(df)}")
    print(f"[TRANSFORM] Columns: {list(df.columns)}")

    silver_object = upload_incremental_parquet(client, df)
    print(f"[TRANSFORM] Uploaded silver parquet: {MINIO_BUCKET}/{silver_object}")

    state["processed_objects"] = sorted(processed.union(new_objects))
    state["last_updated_at"] = datetime.now(timezone.utc).isoformat()

    print("[TRANSFORM] Columns before upload:", df.columns.tolist())
    print(df.head())

    save_state(client, state)
    print(f"[TRANSFORM] Updated state: {MINIO_BUCKET}/{STATE_OBJECT}")


if __name__ == "__main__":
    while True:
        try:
            main()
        except S3Error as e:
            if is_missing_object_error(e):
                print(f"[TRANSFORM] Dependency not ready yet: {e}. Retrying...")
            else:
                raise

        print("[TRANSFORM] Sleeping before next iteration...")
        time.sleep(5)