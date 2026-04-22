import json
import os
import uuid
import hashlib
from io import BytesIO
from datetime import datetime, timezone

import pandas as pd
from minio import Minio
from minio.error import S3Error


MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "fraud-lake")

SILVER_PREFIX = os.getenv("SILVER_PREFIX", "silver/transactions_clean/")
GOLD_PREFIX = os.getenv("GOLD_PREFIX", "gold/training_features/")
GOLD_VERSION = os.getenv("GOLD_VERSION", "v1")
STATE_OBJECT = os.getenv(
    "GOLD_STATE_OBJECT",
    "_state/silver_to_gold/processed_files.json",
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


def list_silver_objects(client: Minio) -> list[str]:
    objects = client.list_objects(
        MINIO_BUCKET,
        prefix=SILVER_PREFIX,
        recursive=True,
    )
    return sorted(
        obj.object_name
        for obj in objects
        if obj.object_name.endswith(".parquet")
    )


def read_parquet_object(client: Minio, object_name: str) -> pd.DataFrame:
    response = client.get_object(MINIO_BUCKET, object_name)
    try:
        raw = response.read()
        return pd.read_parquet(BytesIO(raw), engine="pyarrow")
    finally:
        response.close()
        response.release_conn()


def deterministic_split(value: str) -> str:
    h = hashlib.md5(value.encode("utf-8")).hexdigest()
    bucket = int(h[:8], 16) % 100

    if bucket < 70:
        return "train"
    if bucket < 85:
        return "val"
    return "test"


def build_gold_dataframe(client: Minio, new_silver_objects: list[str]) -> pd.DataFrame:
    frames = []

    for object_name in new_silver_objects:
        df = read_parquet_object(client, object_name)
        if df.empty:
            continue

        df = df.copy()
        df["silver_source_object"] = object_name
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    required_columns = [ID_COLUMN, LABEL_COLUMN] + FEATURE_COLUMNS
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in silver data: {missing}")

    gold_cols = [ID_COLUMN] + FEATURE_COLUMNS + [LABEL_COLUMN]

    optional_cols = []
    for col in ["ingested_at", "source_object_name", "silver_source_object"]:
        if col in df.columns:
            optional_cols.append(col)

    df = df[gold_cols + optional_cols].copy()

    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)

    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=FEATURE_COLUMNS + [LABEL_COLUMN])

    df = df.drop_duplicates(subset=[ID_COLUMN], keep="last")

    df["split"] = df[ID_COLUMN].astype(str).apply(deterministic_split)
    df["dataset_version"] = GOLD_VERSION
    df["gold_created_at"] = datetime.now(timezone.utc).isoformat()

    return df


def upload_gold_parquet(client: Minio, df: pd.DataFrame) -> str:
    now = datetime.now(timezone.utc)
    run_id = uuid.uuid4().hex[:8]

    object_name = (
        f"{GOLD_PREFIX}"
        f"version={GOLD_VERSION}/"
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


def main():
    client = get_minio_client()

    state = load_state(client)
    processed = set(state.get("processed_objects", []))

    all_silver_objects = list_silver_objects(client)
    new_silver_objects = [obj for obj in all_silver_objects if obj not in processed]

    print(f"[GOLD] Silver total: {len(all_silver_objects)}")
    print(f"[GOLD] Already processed: {len(processed)}")
    print(f"[GOLD] New silver objects to process: {len(new_silver_objects)}")

    if not new_silver_objects:
        print("[GOLD] No new silver files found")
        return

    df = build_gold_dataframe(client, new_silver_objects)

    if df.empty:
        print("[GOLD] No rows built from new silver objects")
        return

    print(f"[GOLD] Rows: {len(df)}")
    print(f"[GOLD] Columns: {list(df.columns)}")
    print(f"[GOLD] Split counts:\n{df['split'].value_counts(dropna=False)}")

    gold_object = upload_gold_parquet(client, df)
    print(f"[GOLD] Uploaded gold parquet: {MINIO_BUCKET}/{gold_object}")

    state["processed_objects"] = sorted(processed.union(new_silver_objects))
    state["last_updated_at"] = datetime.now(timezone.utc).isoformat()

    save_state(client, state)
    print(f"[GOLD] Updated state: {MINIO_BUCKET}/{STATE_OBJECT}")


if __name__ == "__main__":
    main()