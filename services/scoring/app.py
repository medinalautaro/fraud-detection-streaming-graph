import json
import os
import socket
import time
from datetime import datetime, timezone
from io import BytesIO

import joblib
import pandas as pd
from confluent_kafka import Consumer, Producer
from minio import Minio

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
INPUT_TOPIC = os.getenv("INPUT_TOPIC", "raw_events")
APPROVED_TOPIC = os.getenv("APPROVED_TOPIC", "approved_transactions")
BLOCKED_TOPIC = os.getenv("BLOCKED_TOPIC", "blocked_transactions")
KAFKA_GROUP_ID = os.getenv("KAFKA_GROUP_ID", "fraud-scoring-group")

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin123")
ARTIFACTS_BUCKET = os.getenv("ARTIFACTS_BUCKET", "ml-artifacts")
MODEL_OBJECT = os.getenv("MODEL_OBJECT", "models/fraud_model/latest/model.joblib")

FRAUD_THRESHOLD = float(os.getenv("FRAUD_THRESHOLD", "0.80"))

FEATURE_COLUMNS = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]


def wait_for_tcp_service(host: str, port: int, service_name: str):
    while True:
        try:
            print(f"[SCORER] Waiting for {service_name} at {host}:{port}...")
            with socket.create_connection((host, port), timeout=5):
                print(f"[SCORER] {service_name} port is open")
                return
        except Exception as e:
            print(f"[SCORER] {service_name} not ready yet: {e}")
            time.sleep(2)


def wait_for_kafka():
    host, port = KAFKA_BOOTSTRAP_SERVERS.split(":")
    wait_for_tcp_service(host, int(port), "Kafka")


def load_model():
    client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False,
    )

    while True:
        try:
            print(f"[SCORER] Loading model from {ARTIFACTS_BUCKET}/{MODEL_OBJECT}...")
            response = client.get_object(ARTIFACTS_BUCKET, MODEL_OBJECT)
            try:
                model_bytes = response.read()
                model = joblib.load(BytesIO(model_bytes))
                print("[SCORER] Model loaded successfully")
                return model
            finally:
                response.close()
                response.release_conn()
        except Exception as e:
            print(f"[SCORER] Model not ready yet: {e}")
            time.sleep(5)


def event_to_features(event: dict) -> pd.DataFrame:
    row = {
        "Time": float(event["time"]),
        "Amount": float(event["amount"]),
    }
    for i in range(1, 29):
        row[f"V{i}"] = float(event[f"V{i}"])
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def compute_latency_ms(event_created_at: str, scored_at: str) -> float:
    start = datetime.fromisoformat(event_created_at)
    end = datetime.fromisoformat(scored_at)
    return (end - start).total_seconds() * 1000.0


def enrich_event(model, event: dict) -> dict:
    X = event_to_features(event)

    fraud_probability = float(model.predict_proba(X)[0][1])
    prediction = int(fraud_probability >= FRAUD_THRESHOLD)
    decision = "blocked" if prediction == 1 else "approved"

    scored_at = datetime.now(timezone.utc).isoformat()
    latency_ms = None

    if "event_created_at" in event:
        try:
            latency_ms = compute_latency_ms(event["event_created_at"], scored_at)
        except Exception:
            latency_ms = None

    return {
        **event,
        "fraud_probability": fraud_probability,
        "prediction": prediction,
        "decision": decision,
        "threshold": FRAUD_THRESHOLD,
        "model_version": "v1",
        "scored_at": scored_at,
        "latency_ms": latency_ms,
    }


def main():
    print("[SCORER] App started")
    wait_for_kafka()

    model = load_model()

    consumer = Consumer(
        {
            "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
            "group.id": KAFKA_GROUP_ID,
            "auto.offset.reset": "earliest",
        }
    )
    producer = Producer({"bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS})

    consumer.subscribe([INPUT_TOPIC])
    print(f"[SCORER] Subscribed to topic: {INPUT_TOPIC}")

    try:
        while True:
            msg = consumer.poll(1.0)

            if msg is None:
                continue

            if msg.error():
                print(f"[SCORER] Kafka error: {msg.error()}")
                continue

            event = json.loads(msg.value().decode("utf-8"))
            transaction_id = event["transaction_id"]

            enriched = enrich_event(model, event)
            output_topic = APPROVED_TOPIC if enriched["decision"] == "approved" else BLOCKED_TOPIC

            producer.produce(
                topic=output_topic,
                key=str(transaction_id).encode("utf-8"),
                value=json.dumps(enriched).encode("utf-8"),
            )
            producer.poll(0)

            print(
                f"[SCORER] transaction_id={transaction_id} "
                f"proba={enriched['fraud_probability']:.4f} "
                f"decision={enriched['decision'].upper()} "
                f"latency_ms={enriched['latency_ms']}"
            )

    except KeyboardInterrupt:
        print("[SCORER] Stopping scorer...")
    finally:
        consumer.close()
        producer.flush()


if __name__ == "__main__":
    main()