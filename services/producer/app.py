import json
import os
import socket
import time

import pandas as pd
from confluent_kafka import Producer


KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "raw_events")
PRODUCER_INTERVAL_SECONDS = float(os.getenv("PRODUCER_INTERVAL_SECONDS", "0.05"))
CSV_PATH = os.getenv("CSV_PATH", "/data/creditcard.csv")
MAX_ROWS = int(os.getenv("MAX_ROWS", "0"))  # 0 = todas


def wait_for_kafka():
    host, port = KAFKA_BOOTSTRAP_SERVERS.split(":")
    port = int(port)

    while True:
        try:
            print(f"[PRODUCER] Waiting for Kafka at {host}:{port}...")
            with socket.create_connection((host, port), timeout=5):
                print("[PRODUCER] Kafka port is open")
                return
        except Exception as e:
            print(f"[PRODUCER] Kafka not ready yet: {e}")
            time.sleep(2)


def delivery_report(err, msg):
    if err is not None:
        print(f"[PRODUCER] Delivery failed: {err}")
    else:
        print(
            f"[PRODUCER] Delivered transaction_id={msg.key().decode('utf-8')} "
            f"to topic={msg.topic()} partition={msg.partition()} offset={msg.offset()}"
        )


def load_dataset(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    required_columns = ["Time", "Amount", "Class"] + [f"V{i}" for i in range(1, 29)]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    return df


def row_to_event(row, transaction_id: int) -> dict:
    event = {
        "transaction_id": transaction_id,
        "time": float(row["Time"]),
        "amount": float(row["Amount"]),
        "class": int(row["Class"]),
    }

    for i in range(1, 29):
        event[f"V{i}"] = float(row[f"V{i}"])

    return event


def main():
    print("[PRODUCER] App started")
    print(f"[PRODUCER] CSV_PATH={CSV_PATH}")
    print(f"[PRODUCER] KAFKA_BOOTSTRAP_SERVERS={KAFKA_BOOTSTRAP_SERVERS}")
    print(f"[PRODUCER] KAFKA_TOPIC={KAFKA_TOPIC}")

    wait_for_kafka()

    while not os.path.exists(CSV_PATH):
        print(f"[PRODUCER] Waiting for dataset at {CSV_PATH}...")
        time.sleep(5)

    df = load_dataset(CSV_PATH)
    print(f"[PRODUCER] Loaded dataset with {len(df)} rows")

    if MAX_ROWS > 0:
        df = df.head(MAX_ROWS)
        print(f"[PRODUCER] Using only first {len(df)} rows")

    producer = Producer({"bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS})
    print("[PRODUCER] Producer created successfully")

    for idx, row in df.iterrows():
        transaction_id = idx + 1
        event = row_to_event(row, transaction_id)

        payload = json.dumps(event).encode("utf-8")
        key = str(transaction_id).encode("utf-8")

        producer.produce(
            topic=KAFKA_TOPIC,
            key=key,
            value=payload,
            callback=delivery_report,
        )
        producer.poll(0)

        print(f"[PRODUCER] Sent transaction_id={transaction_id}")

        if PRODUCER_INTERVAL_SECONDS > 0:
            time.sleep(PRODUCER_INTERVAL_SECONDS)

    producer.flush()
    print("[PRODUCER] Finished sending all rows")


if __name__ == "__main__":
    main()