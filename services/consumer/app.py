import json
import math
import os
import socket
import time
from io import BytesIO

from confluent_kafka import Consumer
from minio import Minio
from minio.error import S3Error
from neo4j import GraphDatabase


KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "raw_events")
KAFKA_GROUP_ID = os.getenv("KAFKA_GROUP_ID", "raw-events-consumer-group")

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin123")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "raw-events")

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")


CONSTRAINT_QUERIES = [
    "CREATE CONSTRAINT transaction_id_unique IF NOT EXISTS FOR (t:Transaction) REQUIRE t.transaction_id IS UNIQUE",
    "CREATE CONSTRAINT timebucket_id_unique IF NOT EXISTS FOR (tb:TimeBucket) REQUIRE tb.bucket_id IS UNIQUE",
    "CREATE CONSTRAINT amountbucket_id_unique IF NOT EXISTS FOR (ab:AmountBucket) REQUIRE ab.bucket_id IS UNIQUE",
    "CREATE CONSTRAINT group_a_tx_unique IF NOT EXISTS FOR (ga:LatentGroupA) REQUIRE ga.transaction_id IS UNIQUE",
    "CREATE CONSTRAINT group_b_tx_unique IF NOT EXISTS FOR (gb:LatentGroupB) REQUIRE gb.transaction_id IS UNIQUE",
    "CREATE CONSTRAINT group_c_tx_unique IF NOT EXISTS FOR (gc:LatentGroupC) REQUIRE gc.transaction_id IS UNIQUE",
    "CREATE CONSTRAINT label_value_unique IF NOT EXISTS FOR (l:Label) REQUIRE l.value IS UNIQUE",
]

UPSERT_QUERY = """
MERGE (t:Transaction {transaction_id: $transaction_id})
SET t.time = $time,
    t.amount = $amount,
    t.class = $class

MERGE (tb:TimeBucket {bucket_id: $time_bucket_id})
SET tb.start_time = $time_bucket_start,
    tb.end_time = $time_bucket_end

MERGE (ab:AmountBucket {bucket_id: $amount_bucket_id})
SET ab.min_amount = $amount_bucket_min,
    ab.max_amount = $amount_bucket_max

MERGE (ga:LatentGroupA {transaction_id: $transaction_id})
SET ga.v1 = $v1, ga.v2 = $v2, ga.v3 = $v3, ga.v4 = $v4, ga.v5 = $v5,
    ga.v6 = $v6, ga.v7 = $v7, ga.v8 = $v8, ga.v9 = $v9

MERGE (gb:LatentGroupB {transaction_id: $transaction_id})
SET gb.v10 = $v10, gb.v11 = $v11, gb.v12 = $v12, gb.v13 = $v13, gb.v14 = $v14,
    gb.v15 = $v15, gb.v16 = $v16, gb.v17 = $v17, gb.v18 = $v18

MERGE (gc:LatentGroupC {transaction_id: $transaction_id})
SET gc.v19 = $v19, gc.v20 = $v20, gc.v21 = $v21, gc.v22 = $v22, gc.v23 = $v23,
    gc.v24 = $v24, gc.v25 = $v25, gc.v26 = $v26, gc.v27 = $v27, gc.v28 = $v28

MERGE (l:Label {value: $class})

MERGE (t)-[:IN_TIME_BUCKET]->(tb)
MERGE (t)-[:IN_AMOUNT_BUCKET]->(ab)
MERGE (t)-[:HAS_GROUP_A]->(ga)
MERGE (t)-[:HAS_GROUP_B]->(gb)
MERGE (t)-[:HAS_GROUP_C]->(gc)
MERGE (t)-[:HAS_LABEL]->(l)
"""


def wait_for_tcp_service(host: str, port: int, service_name: str):
    while True:
        try:
            print(f"[CONSUMER] Waiting for {service_name} at {host}:{port}...")
            with socket.create_connection((host, port), timeout=5):
                print(f"[CONSUMER] {service_name} port is open")
                return
        except Exception as e:
            print(f"[CONSUMER] {service_name} not ready yet: {e}")
            time.sleep(2)


def wait_for_kafka():
    host, port = KAFKA_BOOTSTRAP_SERVERS.split(":")
    wait_for_tcp_service(host, int(port), "Kafka")


def wait_for_neo4j():
    uri = NEO4J_URI.replace("bolt://", "")
    host, port = uri.split(":")
    wait_for_tcp_service(host, int(port), "Neo4j")


def ensure_bucket(client: Minio, bucket_name: str):
    found = client.bucket_exists(bucket_name)
    if not found:
        client.make_bucket(bucket_name)
        print(f"[CONSUMER] Bucket created: {bucket_name}")
    else:
        print(f"[CONSUMER] Bucket already exists: {bucket_name}")


def upload_event(client: Minio, bucket_name: str, event: dict):
    event_id = event["transaction_id"]
    object_name = f"events/event_{event_id}.json"

    data = json.dumps(event, ensure_ascii=False).encode("utf-8")
    data_stream = BytesIO(data)

    client.put_object(
        bucket_name=bucket_name,
        object_name=object_name,
        data=data_stream,
        length=len(data),
        content_type="application/json",
    )
    print(f"[CONSUMER] Uploaded to MinIO: {bucket_name}/{object_name}")


def get_neo4j_driver():
    return GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD),
    )


def ensure_neo4j_ready():
    while True:
        try:
            driver = get_neo4j_driver()
            with driver.session() as session:
                session.run("RETURN 1")
            print("[CONSUMER] Connected to Neo4j")
            return driver
        except Exception as e:
            print(f"[CONSUMER] Waiting for Neo4j driver/session... {e}")
            time.sleep(2)


def create_constraints(driver):
    with driver.session() as session:
        for query in CONSTRAINT_QUERIES:
            session.run(query)
    print("[CONSUMER] Neo4j constraints are ready")


def time_bucket_info(time_value: float) -> dict:
    bucket_idx = int(time_value // 3600)
    start_time = bucket_idx * 3600
    end_time = start_time + 3599
    return {
        "time_bucket_id": f"time_{bucket_idx}",
        "time_bucket_start": start_time,
        "time_bucket_end": end_time,
    }


def amount_bucket_info(amount: float) -> dict:
    if amount < 10:
        bucket_id, mn, mx = "amount_0_10", 0.0, 10.0
    elif amount < 50:
        bucket_id, mn, mx = "amount_10_50", 10.0, 50.0
    elif amount < 100:
        bucket_id, mn, mx = "amount_50_100", 50.0, 100.0
    elif amount < 500:
        bucket_id, mn, mx = "amount_100_500", 100.0, 500.0
    else:
        bucket_id, mn, mx = "amount_500_plus", 500.0, None

    return {
        "amount_bucket_id": bucket_id,
        "amount_bucket_min": mn,
        "amount_bucket_max": mx,
    }


def transform_event_to_graph_params(event: dict) -> dict:
    params = {
        "transaction_id": int(event["transaction_id"]),
        "time": float(event["time"]),
        "amount": float(event["amount"]),
        "class": int(event["class"]),
    }

    params.update(time_bucket_info(params["time"]))
    params.update(amount_bucket_info(params["amount"]))

    for i in range(1, 29):
        params[f"v{i}"] = float(event[f"V{i}"])

    return params


def upsert_event_in_neo4j(driver, event: dict):
    params = transform_event_to_graph_params(event)

    with driver.session() as session:
        session.run(UPSERT_QUERY, params)

    print(f"[CONSUMER] Upserted into Neo4j: transaction_id={params['transaction_id']}")


def main():
    print("[CONSUMER] App started")

    wait_for_kafka()
    wait_for_neo4j()

    consumer = Consumer(
        {
            "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
            "group.id": KAFKA_GROUP_ID,
            "auto.offset.reset": "earliest",
        }
    )

    minio_client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False,
    )

    print(f"[CONSUMER] Connected to MinIO at {MINIO_ENDPOINT}")

    while True:
        try:
            ensure_bucket(minio_client, MINIO_BUCKET)
            break
        except S3Error as e:
            print(f"[CONSUMER] Waiting for MinIO... {e}")
            time.sleep(2)

    neo4j_driver = ensure_neo4j_ready()
    create_constraints(neo4j_driver)

    consumer.subscribe([KAFKA_TOPIC])
    print(f"[CONSUMER] Subscribed to topic: {KAFKA_TOPIC}")

    try:
        while True:
            msg = consumer.poll(1.0)

            if msg is None:
                continue

            if msg.error():
                print(f"[CONSUMER] Kafka error: {msg.error()}")
                continue

            raw_value = msg.value().decode("utf-8")
            event = json.loads(raw_value)

            print(f"[CONSUMER] Received transaction_id={event['transaction_id']}")

            upload_event(minio_client, MINIO_BUCKET, event)
            upsert_event_in_neo4j(neo4j_driver, event)

    except KeyboardInterrupt:
        print("[CONSUMER] Stopping consumer...")
    finally:
        consumer.close()
        neo4j_driver.close()


if __name__ == "__main__":
    main()