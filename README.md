# Fraud Detection Streaming Graph

End-to-end local MLOps project for fraud detection using simulated streaming data, Kafka, MinIO, Neo4j, Airflow, GraphQL, and a baseline machine learning model.

The project uses the Credit Card Fraud Detection dataset to simulate credit card transaction events. Events are streamed through Kafka, persisted in MinIO as a lakehouse-style data layer, represented in Neo4j as a graph, transformed through Bronze, Silver, and Gold layers, and used to train a baseline fraud detection model.

The main goal of this repository is not to build the most accurate fraud detection model. The main goal is to demonstrate how different MLOps, data engineering, graph, and streaming components can work together in a reproducible local environment.

After having the data file on the data folder and doing docker composed up you can use this to check the status of the different steps:

docker logs -f mlops_scoring
docker logs -f mlops_transform
docker logs -f mlops_gold_builder
docker logs -f mlops_training

Consider that scoring will only be able to trigger once a model has been trained. A model will be trained once enough fraudel transaction load to gold, that might take some time.

---

## Dataset

This project uses the Kaggle Credit Card Fraud Detection dataset:

```text
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
```
PLEASE DOWNLOAD THE DATA AND PLACE IT IN THE DATA FOLDER.

The producer expects the dataset at:

```text
data/creditcard.csv
```

The dataset contains anonymized transaction features:

- `Time`
- `Amount`
- `V1` to `V28`
- `Class`

The target variable is:

```text
Class
```

Where:

```text
0 = legitimate transaction
1 = fraudulent transaction
```


---

## Table of Contents

- [Project Goals](#project-goals)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [Main Services](#main-services)
- [Data Lake Layers](#data-lake-layers)
- [ML Artifacts](#ml-artifacts)
- [Graph Model](#graph-model)
- [Requirements](#requirements)
- [Environment Setup](#environment-setup)
- [Running the Project](#running-the-project)
- [Useful URLs](#useful-urls)
- [Default Credentials](#default-credentials)
- [Common Commands](#common-commands)
- [Validation](#validation)
- [Example GraphQL Queries](#example-graphql-queries)
- [Example Neo4j Queries](#example-neo4j-queries)
- [Troubleshooting](#troubleshooting)
- [Possible Improvements](#possible-improvements)
- [Author](#author)

---

## Project Goals

This repository focuses on MLOps architecture and system integration.

It demonstrates:

- Streaming transaction ingestion with Kafka
- Local object storage with MinIO
- Bronze, Silver, and Gold data lake layers
- Graph-based transaction representation with Neo4j
- GraphQL access to graph data
- Batch orchestration with Apache Airflow
- Baseline fraud detection model training
- Model artifact storage
- Real-time scoring and transaction routing
- Local reproducibility through Docker Compose

---

## Architecture

```text
Credit Card CSV
      |
      v
Data Ingestion Service
      |
      v
Producer Service
      |
      v
Kafka Topic: raw_events
      |
      |-------------------------------|
      |                               |
      v                               v
Main Consumer                   Scoring Service
      |                               |
      |                               v
      |                     Loads model from MinIO
      |                               |
      |               ----------------|----------------
      |               |                               |
      |               v                               v
      |     Kafka Topic: approved_transactions   Kafka Topic: blocked_transactions
      |               |                               |
      |               v                               v
      |        Approved Consumer              Blocked Consumer
      |               |                               |
      |               v                               v
      |          MinIO / Neo4j                  MinIO / Neo4j
      |
      |----------------------------|
      v                            v
MinIO Bronze Layer              Neo4j Graph
      |                            |
      v                            v
Bronze -> Silver Transform      GraphQL API
      |
      v
Silver -> Gold Builder
      |
      v
Gold Training Dataset
      |
      v
Training Service
      |
      v
MinIO ML Artifacts
```

### Simplified Flow

```text
CSV -> Producer -> Kafka -> Consumer -> MinIO Bronze -> Silver -> Gold -> Training -> Model Artifacts
                    |
                    v
                  Neo4j -> GraphQL API

CSV -> Producer -> Kafka raw_events -> Scoring -> approved_transactions / blocked_transactions
```

---

## Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| Containerization | Docker Compose | Runs the complete local environment |
| Streaming | Apache Kafka | Simulates real-time transaction events |
| Object Storage | MinIO | Stores Bronze, Silver, Gold, and ML artifacts |
| Graph Database | Neo4j | Stores transactions as graph nodes and relationships |
| Relational Database | PostgreSQL | Stores Airflow metadata |
| Orchestration | Apache Airflow | Runs and validates batch workflows |
| API Layer | FastAPI + Strawberry GraphQL | Exposes graph data through GraphQL |
| Machine Learning | scikit-learn | Trains a baseline fraud detection classifier |
| Data Processing | pandas / pyarrow | Transforms and stores structured datasets |
| Model Serialization | joblib | Saves trained model artifacts |

---

## Repository Structure

```text
fraud-detection-streaming-graph/
│
├── airflow/
│   ├── dags/
│   │   └── fraud_stream_batch_validation.py
│   ├── Dockerfile
│   └── requirements.txt
│
├── postgres/
│   └── init.sql
│
├── services/
│   ├── consumer/
│   │   ├── app.py
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   │
│   ├── data_ingestion/
│   │   ├── download_data.py
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   │
│   ├── gold_builder/
│   │   ├── build_gold_from_silver.py
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   │
│   ├── graphql_api/
│   │   ├── app.py
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   │
│   ├── producer/
│   │   ├── app.py
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   │
│   ├── scoring/
│   │   ├── app.py
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   │
│   ├── training/
│   │   ├── train_from_gold.py
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   │
│   └── transform/
│       ├── transform_bronze_to_silver.py
│       ├── Dockerfile
│       └── requirements.txt
│
├── docker-compose.yml
├── .gitignore
└── README.md
```

---

## Main Services

### PostgreSQL

PostgreSQL is used as the metadata database for Airflow.

Container:

```text
mlops_postgres
```

Default database:

```text
mlops
```

The repository includes:

```text
postgres/init.sql
```

This script creates the Airflow database automatically when PostgreSQL starts from a fresh volume.

---

### MinIO

MinIO is used as a local S3-compatible object store.

It stores:

- Raw Bronze transaction events
- Cleaned Silver datasets
- Gold training datasets
- Model artifacts
- Metrics
- Reports
- Pipeline state files

Container:

```text
mlops_minio
```

Console:

```text
http://localhost:9001
```

---

### Kafka

Kafka is used as the streaming backbone.

Container:

```text
mlops_kafka
```

Main topics:

```text
raw_events
approved_transactions
blocked_transactions
```

The producer publishes transaction events to:

```text
raw_events
```

The scoring service consumes from:

```text
raw_events
```

Then it routes transactions to:

```text
approved_transactions
blocked_transactions
```

---

### Data Ingestion Service

The data ingestion service is responsible for downloading or preparing the dataset.

Container:

```text
fraud_data_ingestion
```

It uses the Kaggle API credentials mounted from:

```text
kaggle.json
```

The expected output is:

```text
data/creditcard.csv
```

---

### Producer Service

The producer reads rows from:

```text
data/creditcard.csv
```

It converts each row into a JSON transaction event and sends it to Kafka.

Container:

```text
mlops_producer
```

Output topic:

```text
raw_events
```

Each event contains:

- `transaction_id`
- `time`
- `amount`
- `class`
- `V1` to `V28`

Useful environment variables:

```text
KAFKA_BOOTSTRAP_SERVERS=kafka:9092
KAFKA_TOPIC=raw_events
PRODUCER_INTERVAL_SECONDS=0.2
CSV_PATH=/data/creditcard.csv
```

Example manual execution with limited rows:

```bash
docker exec -w /app mlops_producer sh -c "MAX_ROWS=500 PRODUCER_INTERVAL_SECONDS=0 python -u app.py"
```

---

### Main Consumer Service

The main consumer reads transaction events from Kafka and writes them to two destinations:

1. MinIO Bronze layer
2. Neo4j graph database

Container:

```text
mlops_consumer
```

Input topic:

```text
raw_events
```

Bronze path:

```text
fraud-lake/bronze/transactions/
```

---

### Approved Consumer

The approved consumer reads transactions that the scoring service classified as approved.

Container:

```text
mlops_consumer_approved
```

Input topic:

```text
approved_transactions
```

This allows approved transactions to be persisted separately after scoring.

---

### Blocked Consumer

The blocked consumer reads transactions that the scoring service classified as blocked or high risk.

Container:

```text
mlops_consumer_blocked
```

Input topic:

```text
blocked_transactions
```

This allows suspicious transactions to be persisted separately after scoring.

---

### Scoring Service

The scoring service performs real-time inference over streaming transactions.

Container:

```text
mlops_scoring
```

It consumes from:

```text
raw_events
```

It loads the trained model from MinIO:

```text
ml-artifacts/models/fraud_model/latest/model.joblib
```

Then it routes transactions to:

```text
approved_transactions
blocked_transactions
```

The fraud threshold is configured with:

```text
FRAUD_THRESHOLD=0.80
```

A transaction with a fraud probability above the threshold is routed to the blocked topic. Otherwise, it is routed to the approved topic.

---

### Transform Service

The transform service reads Bronze JSON files from MinIO, cleans and normalizes them, then writes Silver files.

Container:

```text
mlops_transform
```

Input:

```text
fraud-lake/bronze/transactions/
```

Output:

```text
fraud-lake/silver/transactions_clean/
```

State object:

```text
fraud-lake/_state/bronze_to_silver/processed_files.json
```

The state object prevents already processed files from being transformed again.

Run manually:

```bash
docker compose up transform
```

---

### Gold Builder

The Gold builder reads Silver data and creates versioned training datasets.

Container:

```text
mlops_gold_builder
```

Input:

```text
fraud-lake/silver/transactions_clean/
```

Output:

```text
fraud-lake/gold/training_features/version=v1/
```

State object:

```text
fraud-lake/_state/silver_to_gold/processed_files.json
```

Run manually:

```bash
docker compose up gold_builder
```

---

### Training Service

The training service reads Gold data from MinIO and trains a baseline fraud detection model.

Container:

```text
mlops_training
```

Current model:

```text
StandardScaler + LogisticRegression
```

Features:

- `Time`
- `Amount`
- `V1` to `V28`

Target:

```text
Class
```

Artifacts are written to:

```text
ml-artifacts/models/fraud_model/
ml-artifacts/metrics/fraud_model/
ml-artifacts/reports/fraud_model/
```

Run manually:

```bash
docker compose up training
```

---

### Neo4j

Neo4j stores a graph representation of transactions.

Container:

```text
mlops_neo4j
```

Browser:

```text
http://localhost:7474
```

Neo4j allows queries over:

- Transaction labels
- Amount buckets
- Time buckets
- Latent feature groups
- Fraud-related graph patterns

---

### GraphQL API

The GraphQL API exposes Neo4j data through a query layer.

Container:

```text
mlops_graphql_api
```

Endpoint:

```text
http://localhost:8000/graphql
```

Health endpoint:

```text
http://localhost:8000/health
```

---

### Airflow

Airflow is used to orchestrate and validate parts of the batch flow.

Containers:

```text
airflow_webserver
airflow_scheduler
airflow_dag_processor
airflow_init
```

UI:

```text
http://localhost:8080
```

Current DAG:

```text
fraud_stream_batch_validation
```

The DAG validates that:

1. The producer can run in batch mode
2. Bronze files exist in MinIO
3. Transaction nodes exist in Neo4j

Expected task order:

```text
run_producer_batch -> validate_minio -> validate_neo4j
```

---

## Data Lake Layers

### Bronze

Raw ingested transaction events.

Path:

```text
fraud-lake/bronze/transactions/
```

Purpose:

- Preserve raw incoming events
- Store the original event payload
- Keep the ingestion layer immutable
- Provide replayability for downstream transformations

---

### Silver

Cleaned transaction data.

Path:

```text
fraud-lake/silver/transactions_clean/
```

Purpose:

- Normalize fields
- Validate schema
- Prepare data for feature generation
- Remove inconsistencies from raw event files

---

### Gold

Training-ready datasets.

Path:

```text
fraud-lake/gold/training_features/version=v1/
```

Purpose:

- Store model-ready feature tables
- Create train, validation, and test splits
- Provide stable input for the training service
- Version datasets used for model training

---

## ML Artifacts

The training service writes outputs to the `ml-artifacts` bucket.

Expected layout:

```text
ml-artifacts/
├── models/
│   └── fraud_model/
│       ├── latest/
│       │   └── model.joblib
│       └── version=v1/
│           └── run=<run_id>/
│               └── model.joblib
│
├── metrics/
│   └── fraud_model/
│       └── version=v1/
│           └── run=<run_id>/
│               └── metrics.json
│
└── reports/
    └── fraud_model/
        └── version=v1/
            └── run=<run_id>/
                └── report.json
```

Metrics can include:

- Accuracy
- Precision
- Recall
- F1-score
- ROC AUC
- Classification report
- Confusion matrix

---

## Graph Model

Transactions are stored as graph nodes in Neo4j.

Typical node types:

```text
Transaction
Label
TimeBucket
AmountBucket
LatentGroupA
LatentGroupB
LatentGroupC
```

Typical relationships:

```text
(:Transaction)-[:HAS_LABEL]->(:Label)
(:Transaction)-[:IN_TIME_BUCKET]->(:TimeBucket)
(:Transaction)-[:IN_AMOUNT_BUCKET]->(:AmountBucket)
(:Transaction)-[:HAS_GROUP_A]->(:LatentGroupA)
(:Transaction)-[:HAS_GROUP_B]->(:LatentGroupB)
(:Transaction)-[:HAS_GROUP_C]->(:LatentGroupC)
```

This graph structure allows queries such as:

- Fraud distribution by amount bucket
- Fraud distribution by time bucket
- Transactions connected to specific latent feature groups
- High-risk transaction groups
- Aggregated transaction patterns

---

## Requirements

You need:

- Docker
- Docker Compose
- Kaggle account
- Kaggle API token

No external cloud provider is required. Everything runs locally through Docker Compose.

---

## Environment Setup

### 1. Kaggle Credentials

Create a `kaggle.json` file in the project root:

```json
{
  "username": "your_kaggle_username",
  "key": "your_kaggle_api_key"
}
```

This file should be ignored by Git.

Expected path:

```text
fraud-detection-streaming-graph/kaggle.json
```

---

### 2. Environment Variables

Create a `.env` file in the project root.

Example:

```env
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin123

MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin123
MINIO_BUCKET=fraud-lake

NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password123
```

The Docker Compose file also defines several service-level variables directly.

---

### 3. Dataset Location

The producer expects:

```text
data/creditcard.csv
```

If the ingestion service is not used, download the file manually from Kaggle and place it there.

---

## Running the Project

### 1. Clone the Repository

```bash
git clone https://github.com/medinalautaro/fraud-detection-streaming-graph.git
cd fraud-detection-streaming-graph
```

### 2. Add Kaggle Credentials

Place `kaggle.json` in the project root:

```text
fraud-detection-streaming-graph/kaggle.json
```

### 3. Create `.env`

Create the `.env` file using the variables shown above.

### 4. Start All Services

```bash
docker compose up --build
```

Or run in detached mode:

```bash
docker compose up --build -d
```

### 5. Check Running Containers

```bash
docker ps
```

Expected containers include:

- `mlops_postgres`
- `mlops_minio`
- `mlops_kafka`
- `mlops_neo4j`
- `mlops_graphql_api`
- `airflow_webserver`
- `airflow_scheduler`
- `airflow_dag_processor`
- `mlops_producer`
- `mlops_consumer`
- `mlops_consumer_approved`
- `mlops_consumer_blocked`
- `mlops_scoring`
- `mlops_transform`
- `mlops_gold_builder`
- `mlops_training`

Some batch containers may finish and exit successfully. That is expected for services with `restart: "no"`.

---

## Useful URLs

| Service | URL |
|---|---|
| MinIO Console | http://localhost:9001 |
| Neo4j Browser | http://localhost:7474 |
| GraphQL API | http://localhost:8000/graphql |
| GraphQL Health | http://localhost:8000/health |
| Airflow UI | http://localhost:8080 |

---

## Default Credentials

### MinIO

```text
Username: minioadmin
Password: minioadmin123
```

### Neo4j

```text
Username: neo4j
Password: password123
```

---

## Common Commands

### Start Everything

```bash
docker compose up --build
```

### Start Everything in Detached Mode

```bash
docker compose up --build -d
```

### Stop Everything

```bash
docker compose down
```

### Stop and Delete Volumes

Use this for a completely fresh start:

```bash
docker compose down -v
docker compose up --build
```

### View All Logs

```bash
docker compose logs -f
```

### View Producer Logs

```bash
docker logs -f mlops_producer
```

### View Main Consumer Logs

```bash
docker logs -f mlops_consumer
```

### View Scoring Logs

```bash
docker logs -f mlops_scoring
```

### View Approved Consumer Logs

```bash
docker logs -f mlops_consumer_approved
```

### View Blocked Consumer Logs

```bash
docker logs -f mlops_consumer_blocked
```

### View Transform Logs

```bash
docker logs -f mlops_transform
```

### View Gold Builder Logs

```bash
docker logs -f mlops_gold_builder
```

### View Training Logs

```bash
docker logs -f mlops_training
```

### Run Producer Manually With Limited Rows

```bash
docker exec -w /app mlops_producer sh -c "MAX_ROWS=500 PRODUCER_INTERVAL_SECONDS=0 python -u app.py"
```

### Run Bronze to Silver Manually

```bash
docker compose up transform
```

### Run Silver to Gold Manually

```bash
docker compose up gold_builder
```

### Run Training Manually

```bash
docker compose up training
```

---

## Validation

### Validate Kafka Producer

```bash
docker logs -f mlops_producer
```

Expected messages:

```text
[PRODUCER] Sent transaction_id=...
[PRODUCER] Delivered transaction_id=...
```

---

### Validate Main Consumer

```bash
docker logs -f mlops_consumer
```

Expected messages:

```text
[CONSUMER] Received transaction_id=...
[CONSUMER] Uploaded to MinIO: ...
[CONSUMER] Upserted into Neo4j: transaction_id=...
```

---

### Validate Scoring

```bash
docker logs -f mlops_scoring
```

The scoring service should consume transactions from `raw_events`, load the model from MinIO, and route transactions to either `approved_transactions` or `blocked_transactions`.

---

### Validate MinIO

Open:

```text
http://localhost:9001
```

Log in with:

```text
minioadmin / minioadmin123
```

Expected buckets:

```text
fraud-lake
ml-artifacts
```

Expected folders in `fraud-lake`:

```text
bronze/
silver/
gold/
_state/
```

Expected folders in `ml-artifacts`:

```text
models/
metrics/
reports/
```

---

### Validate Neo4j

Open:

```text
http://localhost:7474
```

Run:

```cypher
MATCH (t:Transaction)
RETURN count(t) AS transactions;
```

Run:

```cypher
MATCH (t:Transaction)-[:HAS_LABEL]->(l:Label)
RETURN l.value AS label, count(t) AS count
ORDER BY label;
```

---

### Validate GraphQL

Open:

```text
http://localhost:8000/graphql
```

Run:

```graphql
query {
  fraudTransactions(limit: 5) {
    transactionId
    amount
    classValue
  }
}
```

---

### Validate Airflow

Open:

```text
http://localhost:8080
```

Find the DAG:

```text
fraud_stream_batch_validation
```

Trigger it manually.

Expected task order:

```text
run_producer_batch -> validate_minio -> validate_neo4j
```

---

## Example GraphQL Queries

### Get One Transaction

```graphql
query {
  transaction(transactionId: 1) {
    transactionId
    time
    amount
    classValue
    label {
      value
    }
    timeBucket {
      bucketId
      startTime
      endTime
    }
    amountBucket {
      bucketId
      minAmount
      maxAmount
    }
  }
}
```

### Get Latest Transactions

```graphql
query {
  transactions(limit: 10) {
    transactionId
    amount
    classValue
    label {
      value
    }
  }
}
```

### Get Fraud Transactions

```graphql
query {
  fraudTransactions(limit: 10) {
    transactionId
    amount
    classValue
    label {
      value
    }
  }
}
```

---

## Example Neo4j Queries

### Count Transactions

```cypher
MATCH (t:Transaction)
RETURN count(t) AS transactions;
```

### Get Fraud Transactions

```cypher
MATCH (t:Transaction)-[:HAS_LABEL]->(:Label {value: 1})
RETURN t.transaction_id, t.amount, t.time
ORDER BY t.transaction_id DESC
LIMIT 25;
```

### Count Transactions by Amount Bucket

```cypher
MATCH (t:Transaction)-[:IN_AMOUNT_BUCKET]->(ab:AmountBucket)
RETURN ab.bucket_id, count(t) AS transactions
ORDER BY transactions DESC;
```

### Count Transactions by Time Bucket

```cypher
MATCH (t:Transaction)-[:IN_TIME_BUCKET]->(tb:TimeBucket)
RETURN tb.bucket_id, count(t) AS transactions
ORDER BY tb.bucket_id;
```

### Fraud Distribution by Amount Bucket

```cypher
MATCH (t:Transaction)-[:IN_AMOUNT_BUCKET]->(ab:AmountBucket),
      (t)-[:HAS_LABEL]->(l:Label)
RETURN ab.bucket_id AS amount_bucket,
       l.value AS label,
       count(t) AS transactions
ORDER BY amount_bucket, label;
```

---

## Troubleshooting

### MinIO Is Empty

Check producer and consumer logs:

```bash
docker logs -f mlops_producer
docker logs -f mlops_consumer
```

Make sure the dataset exists at:

```text
data/creditcard.csv
```

---

### Dataset Is Missing

Make sure the dataset exists at:

```text
data/creditcard.csv
```

Or make sure `kaggle.json` exists in the project root before running the ingestion service.

---

### Gold Folder Does Not Appear in MinIO

Run:

```bash
docker compose up transform
docker compose up gold_builder
```

Then refresh MinIO.

---

### Training Fails Because Gold Data Is Missing

Run the pipeline in this order:

```bash
docker compose up producer consumer
docker compose up transform
docker compose up gold_builder
docker compose up training
```

---

### Scoring Fails Because the Model Is Missing

The scoring service expects the model at:

```text
ml-artifacts/models/fraud_model/latest/model.joblib
```

Run training first:

```bash
docker compose up training
```

Then restart scoring:

```bash
docker compose restart scoring
```

---

### Airflow Database Does Not Exist

The repository includes:

```text
postgres/init.sql
```

This creates the Airflow database automatically when PostgreSQL starts from a fresh volume.

If the volume already existed before this file was added, reset volumes:

```bash
docker compose down -v
docker compose up --build
```

---

### Kafka Topic Does Not Exist

The `kafka-init` service creates the `raw_events` topic automatically.

To inspect topics manually:

```bash
docker exec -it mlops_kafka /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server kafka:9092 \
  --list
```

If approved or blocked topics are missing, create them manually:

```bash
docker exec -it mlops_kafka /opt/kafka/bin/kafka-topics.sh \
  --create \
  --if-not-exists \
  --topic approved_transactions \
  --bootstrap-server kafka:9092 \
  --partitions 1 \
  --replication-factor 1
```

```bash
docker exec -it mlops_kafka /opt/kafka/bin/kafka-topics.sh \
  --create \
  --if-not-exists \
  --topic blocked_transactions \
  --bootstrap-server kafka:9092 \
  --partitions 1 \
  --replication-factor 1
```

---

### Neo4j Has No Transactions

Make sure the consumer is running and the producer has sent events.

```bash
docker logs -f mlops_consumer
docker logs -f mlops_producer
```

Then run in Neo4j:

```cypher
MATCH (t:Transaction)
RETURN count(t);
```

---

### GraphQL Returns Empty Results

GraphQL reads from Neo4j. First check that Neo4j contains transactions:

```cypher
MATCH (t:Transaction)
RETURN count(t);
```

Then check the GraphQL health endpoint:

```text
http://localhost:8000/health
```

---

## Development Notes

This project is designed for local experimentation and portfolio demonstration.

It is not production-ready as-is because:

- Credentials are simple local defaults
- Services run in a single Docker Compose environment
- The GraphQL API has no authentication layer
- There is no CI/CD pipeline yet
- The model is a baseline classifier
- Monitoring and alerting are not implemented yet
- Secrets are handled only for local development

---

## Possible Improvements

Useful next improvements:

- Add MLflow for experiment tracking
- Add a model registry
- Add batch inference
- Add real-time inference monitoring
- Add Prometheus and Grafana
- Add data quality checks with Great Expectations
- Add unit and integration tests
- Add GitHub Actions CI
- Add a dashboard for fraud analytics
- Add GraphQL aggregate queries
- Add gRPC model serving
- Add model promotion between versions
- Add drift detection
- Add Makefile commands for easier operation
- Add `.env.example`
- Add architecture image to the README

---

## License

This project is intended for educational and portfolio purposes. Add a license file if you want to define explicit usage permissions.

---

## Author

Lautaro Medina
