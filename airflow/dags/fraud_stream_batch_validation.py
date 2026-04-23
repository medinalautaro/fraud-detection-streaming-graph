from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from minio import Minio
from neo4j import GraphDatabase
import os

def check_minio_files():
    client = Minio(
        os.environ["MINIO_ENDPOINT"],
        access_key=os.environ["MINIO_ACCESS_KEY"],
        secret_key=os.environ["MINIO_SECRET_KEY"],
        secure=False,
    )

    bucket = os.environ["MINIO_BUCKET"]

    objects = list(client.list_objects(bucket, recursive=True))
    bronze_objects = [
        obj for obj in objects
        if obj.object_name.startswith("bronze/transactions/")
    ]

    if len(bronze_objects) == 0:
        raise ValueError("No bronze transaction files found in MinIO")

    print(f"Found {len(bronze_objects)} bronze files in MinIO")


def check_neo4j_transactions():
    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(
            os.environ["NEO4J_USER"],
            os.environ["NEO4J_PASSWORD"],
        ),
    )

    try:
        with driver.session() as session:
            result = session.run(
                "MATCH (t:Transaction) RETURN count(t) AS c"
            )
            count = result.single()["c"]

            if count == 0:
                raise ValueError("No Transaction nodes found in Neo4j")

            print(f"Found {count} Transaction nodes in Neo4j")
    finally:
        driver.close()


with DAG(
    dag_id="fraud_stream_batch_validation",
    start_date=datetime(2026, 4, 22),
    schedule=None,
    catchup=False,
    tags=["fraud", "streaming", "mlops"],
) as dag:

    run_producer_batch = BashOperator(
        task_id="run_producer_batch",
        bash_command="""
        docker exec -w /app mlops_producer \
        sh -c "MAX_ROWS=500 PRODUCER_INTERVAL_SECONDS=0 python -u app.py"
        """,
        cwd="/opt/airflow/project",
    )

    validate_minio = PythonOperator(
        task_id="validate_minio",
        python_callable=check_minio_files,
    )

    validate_neo4j = PythonOperator(
        task_id="validate_neo4j",
        python_callable=check_neo4j_transactions,
    )

    run_producer_batch >> validate_minio >> validate_neo4j