# Fraud Detection Streaming Graph

End-to-end MLOps project for fraud detection using simulated streaming data, a local data lake, graph storage, batch orchestration, model training, and a GraphQL query layer.

The project uses the Credit Card Fraud Detection dataset to simulate transaction events. Events are streamed through Kafka, persisted in MinIO as a lakehouse-style storage layer, represented in Neo4j as a graph, transformed into Silver and Gold datasets, and used to train a baseline fraud detection model.

## Set up

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Download the data for this dataset and paste creditcard.csv in the data folder.



## Project Goals

This repository focuses on MLOps architecture rather than model complexity. The main objective is to demonstrate how different production-oriented components can work together in a local, reproducible environment.

The project covers:

- Streaming data ingestion with Kafka
- Object storage with MinIO
- Bronze, Silver, and Gold data lake layers
- Graph modeling with Neo4j
- Workflow orchestration with Apache Airflow
- Batch validation pipelines
- Baseline ML training from Gold data
- Artifact storage for models, metrics, and reports
- GraphQL API for querying transaction data

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
      v
Consumer Service
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

## Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| Containerization | Docker Compose | Run the full local environment |
| Streaming | Apache Kafka | Simulate real-time transaction events |
| Object Storage | MinIO | Store Bronze, Silver, Gold, and ML artifacts |
| Graph Database | Neo4j | Store transaction relationships and query graph structure |
| Relational Database | PostgreSQL | Airflow metadata database |
| Orchestration | Apache Airflow | Run validation and batch workflows |
| API Layer | FastAPI + Strawberry GraphQL | Query fraud and transaction graph data |
| ML | scikit-learn | Train a baseline fraud classifier |
| Data Processing | pandas / pyarrow | Transform and store data files |

## Repository Structure

```text
fraud-detection-streaming-graph/
│
├── airflow/
│   ├── dags/
│   │   └── fraud_stream_batch_validation.py
│   ├── dockerfile
│   └── requirements.txt
│
├── postgres/
│   └── init.sql
│
├── services/
│   ├── consumer/
│   │   ├── app.py
│   │   ├── dockerfile
│   │   └── requirements.txt
│   │
│   ├── data_ingestion/
│   │   ├── download_data.py
│   │   ├── dockerfile
│   │   └── requirements.txt
│   │
│   ├── gold_builder/
│   │   ├── build_gold_from_silver.py
│   │   ├── dockerfile
│   │   └── requirements.txt
│   │
│   ├── graphql_api/
│   │   ├── app.py
│   │   ├── dockerfile
│   │   └── requirements.txt
│   │
│   ├── producer/
│   │   ├── app.py
│   │   ├── dockerfile
│   │   └── requirements.txt
│   │
│   ├── training/
│   │   ├── train_from_gold.py
│   │   ├── dockerfile
│   │   └── requirements.txt
│   │
│   └── transform/
│       ├── transform_bronze_to_silver.py
│       ├── dockerfile
│       └── requirements.txt
│
├── docker-compose.yml
├── .gitignore
└── README.md
```

## Main Services

### Kafka

Kafka is used as the streaming backbone. The producer publishes transaction events to the `raw_events` topic, and the consumer reads from that topic.

### Producer

The producer reads rows from `data/creditcard.csv`, converts each row into a JSON transaction event, and sends it to Kafka.

Each event contains:

- `transaction_id`
- `time`
- `amount`
- `class`
- `V1` to `V28`

The producer supports limiting the number of rows with `MAX_ROWS`.

Example:

```bash
docker exec -w /app mlops_producer sh -c "MAX_ROWS=500 PRODUCER_INTERVAL_SECONDS=0 python -u app.py"
```

### Consumer

The consumer reads transaction events from Kafka and writes them to two destinations:

1. MinIO Bronze layer as JSON files
2. Neo4j as graph nodes and relationships

Bronze objects are stored under:

```text
fraud-lake/bronze/transactions/
```

The Neo4j graph contains:

- `Transaction`
- `TimeBucket`
- `AmountBucket`
- `LatentGroupA`
- `LatentGroupB`
- `LatentGroupC`
- `Label`

Relationships include:

- `IN_TIME_BUCKET`
- `IN_AMOUNT_BUCKET`
- `HAS_GROUP_A`
- `HAS_GROUP_B`
- `HAS_GROUP_C`
- `HAS_LABEL`

### MinIO Data Lake

MinIO acts as the local data lake.

Expected lake layout:

```text
fraud-lake/
├── bronze/
│   └── transactions/
│
├── silver/
│   └── transactions_clean/
│
├── gold/
│   └── training_features/
│
└── _state/
    ├── bronze_to_silver/
    └── silver_to_gold/
```

ML artifacts are stored in a separate bucket:

```text
ml-artifacts/
├── models/
├── metrics/
└── reports/
```

### Transform Service

The transform service reads Bronze transaction JSON files from MinIO, cleans and normalizes them, then writes Silver files.

Input:

```text
fraud-lake/bronze/transactions/
```

Output:

```text
fraud-lake/silver/transactions_clean/
```

The service also keeps state in MinIO to avoid reprocessing already processed files.

### Gold Builder

The Gold builder reads Silver data and creates versioned training datasets.

Input:

```text
fraud-lake/silver/transactions_clean/
```

Output:

```text
fraud-lake/gold/training_features/version=v1/
```

The Gold layer includes train, validation, and test splits.

### Training Service

The training service reads Gold data from MinIO and trains a baseline fraud detection model.

Current model:

```text
Logistic Regression + StandardScaler
```

The model uses:

- `Time`
- `Amount`
- `V1` to `V28`

The label column is:

```text
Class
```

The training service stores:

```text
ml-artifacts/models/fraud_model/
ml-artifacts/metrics/fraud_model/
ml-artifacts/reports/fraud_model/
```

Metrics include:

- Accuracy
- Precision
- Recall
- F1-score
- ROC AUC
- Classification report
- Confusion matrix

### Neo4j

Neo4j stores a graph representation of transactions. This allows querying transactions by label, amount buckets, time buckets, and latent feature groups.

Neo4j Browser:

```text
http://localhost:7474
```

Credentials:

```text
Username: neo4j
Password: password123
```

Example Cypher queries:

```cypher
MATCH (t:Transaction)
RETURN t
LIMIT 25;
```

```cypher
MATCH (t:Transaction)-[:HAS_LABEL]->(:Label {value: 1})
RETURN t.transaction_id, t.amount, t.time
ORDER BY t.transaction_id DESC
LIMIT 25;
```

```cypher
MATCH (t:Transaction)-[:IN_AMOUNT_BUCKET]->(ab:AmountBucket)
RETURN ab.bucket_id, count(t) AS transactions
ORDER BY transactions DESC;
```

```cypher
MATCH (t:Transaction)-[:IN_TIME_BUCKET]->(tb:TimeBucket)
RETURN tb.bucket_id, count(t) AS transactions
ORDER BY tb.bucket_id;
```

### GraphQL API

The GraphQL API exposes transaction data from Neo4j.

GraphQL endpoint:

```text
http://localhost:8000/graphql
```

Health endpoint:

```text
http://localhost:8000/health
```

Example query: get one transaction

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

Example query: latest transactions

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

Example query: fraud transactions

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

### Airflow

Airflow is used to orchestrate and validate the batch flow.

Airflow UI:

```text
http://localhost:8080
```

Current DAG:

```text
fraud_stream_batch_validation
```

The DAG performs:

1. Runs the producer in batch mode
2. Validates that Bronze files exist in MinIO
3. Validates that Transaction nodes exist in Neo4j

## Requirements

You need:

- Docker
- Docker Compose
- Kaggle account
- Kaggle API token

No external cloud provider is required. Everything runs locally through Docker Compose.

## Dataset

This project expects the Credit Card Fraud Detection dataset from Kaggle.

The producer expects the dataset at:

```text
data/creditcard.csv
```

This file is ignored by Git because datasets should not be committed to the repository.

## Kaggle Setup

Create a `kaggle.json` file in the project root:

```text
kaggle.json
```

The file should contain your Kaggle API credentials:

```json
{
  "username": "your_kaggle_username",
  "key": "your_kaggle_api_key"
}
```

This file is ignored by Git.

## Environment Variables

Create a `.env` file in the project root:

```env
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin123
MINIO_BUCKET=fraud-lake

NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password123
```

The current Docker Compose file also defines several service-level variables directly.

## Running the Project from Zero

### 1. Clone the repository

```bash
git clone https://github.com/medinalautaro/fraud-detection-streaming-graph.git
cd fraud-detection-streaming-graph
```

### 2. Add Kaggle credentials

Place your `kaggle.json` file in the project root.

```text
fraud-detection-streaming-graph/kaggle.json
```

### 3. Create `.env`

```bash
cp .env.example .env
```

If `.env.example` does not exist yet, create `.env` manually using the variables shown above.

### 4. Start all services

```bash
docker compose up --build
```

Or run in detached mode:

```bash
docker compose up --build -d
```

### 5. Check running containers

```bash
docker ps
```

You should see containers for:

- PostgreSQL
- MinIO
- Kafka
- Kafka init
- Neo4j
- Airflow
- Producer
- Consumer
- GraphQL API
- Transform
- Gold builder
- Training

Some batch containers may finish and exit successfully. That is expected for services with `restart: "no"`.

## Useful URLs

| Service | URL |
|---|---|
| MinIO Console | http://localhost:9001 |
| Neo4j Browser | http://localhost:7474 |
| GraphQL API | http://localhost:8000/graphql |
| GraphQL Health | http://localhost:8000/health |
| Airflow UI | http://localhost:8080 |

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

## Common Commands

### Start everything

```bash
docker compose up --build
```

### Start in detached mode

```bash
docker compose up --build -d
```

### Stop everything

```bash
docker compose down
```

### Stop and delete volumes

Use this when you want a completely fresh start:

```bash
docker compose down -v
```

### View logs

```bash
docker compose logs -f
```

### View producer logs

```bash
docker logs -f mlops_producer
```

### View consumer logs

```bash
docker logs -f mlops_consumer
```

### View transform logs

```bash
docker logs -f mlops_transform
```

### View Gold builder logs

```bash
docker logs -f mlops_gold_builder
```

### View training logs

```bash
docker logs -f mlops_training
```

### Run producer manually with limited rows

```bash
docker exec -w /app mlops_producer sh -c "MAX_ROWS=500 PRODUCER_INTERVAL_SECONDS=0 python -u app.py"
```

### Run Bronze to Silver manually

```bash
docker compose up transform
```

### Run Silver to Gold manually

```bash
docker compose up gold_builder
```

### Run training manually

```bash
docker compose up training
```

## Validating the Pipeline

### Validate Kafka producer and consumer

Check producer logs:

```bash
docker logs -f mlops_producer
```

Expected messages:

```text
[PRODUCER] Sent transaction_id=...
[PRODUCER] Delivered transaction_id=...
```

Check consumer logs:

```bash
docker logs -f mlops_consumer
```

Expected messages:

```text
[CONSUMER] Received transaction_id=...
[CONSUMER] Uploaded to MinIO: fraud-lake/bronze/transactions/...
[CONSUMER] Upserted into Neo4j: transaction_id=...
```

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

If `gold/` does not appear, run:

```bash
docker compose up gold_builder
```

Then run:

```bash
docker compose up training
```

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

## Data Lake Layers

### Bronze

Raw ingested JSON events.

```text
bronze/transactions/
```

Purpose:

- Preserve raw incoming event data
- Keep ingestion metadata
- Avoid modifying the original event payload

### Silver

Cleaned transaction data.

```text
silver/transactions_clean/
```

Purpose:

- Normalize fields
- Validate schema
- Prepare data for feature generation

### Gold

Training-ready dataset.

```text
gold/training_features/version=v1/
```

Purpose:

- Store feature tables
- Include train, validation, and test splits
- Provide stable input for model training

## ML Artifacts

The training service writes outputs to the `ml-artifacts` bucket.

Expected layout:

```text
ml-artifacts/
├── models/
│   └── fraud_model/
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

## Current Model

The current training service uses a simple baseline model:

```text
StandardScaler + LogisticRegression
```

Model configuration:

```text
solver: lbfgs
max_iter: 5000
class_weight: balanced
random_state: 42
```

This is intentionally simple because the project focuses on the MLOps pipeline, not on maximizing predictive performance.

## Why Neo4j Is Used

Neo4j is used to represent transactions as a graph.

Instead of storing each transaction only as a flat row, each transaction is connected to:

- Its fraud label
- Its time bucket
- Its amount bucket
- Groups of latent PCA-like features

This makes it possible to query relationships such as:

- Fraud distribution by amount bucket
- Fraud distribution by time bucket
- Transactions connected to specific feature groups
- High-risk graph neighborhoods
- Aggregated graph patterns

## Why MinIO Is Used

MinIO provides an S3-compatible local object store. In this project it works as a local data lake.

It is useful because it separates:

- Raw data
- Cleaned data
- Training features
- Model artifacts
- Metrics
- Reports

This is closer to a production ML architecture than storing everything as local folders inside one container.

## Development Notes

This project is designed for local experimentation and portfolio demonstration.

It is not production-ready as-is because:

- Credentials are simple local defaults
- Services run on a single Docker Compose environment
- There is no authentication layer on the GraphQL API
- There is no CI/CD pipeline yet
- The model is a baseline classifier
- Monitoring and alerting are not implemented yet

## Possible Improvements

Useful next improvements:

- Add MLflow for experiment tracking
- Add model registry behavior
- Add batch inference service
- Add real-time inference consumer
- Add Prometheus and Grafana monitoring
- Add data quality checks with Great Expectations
- Add unit and integration tests
- Add GitHub Actions CI
- Add a dashboard for fraud analytics
- Add GraphQL queries for aggregate fraud metrics
- Add gRPC service for model inference
- Add versioned model promotion workflow
- Add Makefile commands for easier operation

## Troubleshooting

### Gold folder does not appear in MinIO

Run:

```bash
docker compose up gold_builder
```

Then refresh MinIO.

### Training fails because no Gold files exist

Run the pipeline in this order:

```bash
docker compose up producer consumer
docker compose up transform
docker compose up gold_builder
docker compose up training
```

### Airflow database does not exist

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

### Dataset is missing

Make sure the dataset exists at:

```text
data/creditcard.csv
```

Or make sure `kaggle.json` exists in the project root before running the ingestion service.

### Kafka topic does not exist

The `kafka-init` service creates the `raw_events` topic automatically.

To inspect topics manually:

```bash
docker exec -it mlops_kafka /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server kafka:9092 \
  --list
```

### Neo4j has no transactions

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

### GraphQL returns empty results

Check that Neo4j contains transactions first.

```cypher
MATCH (t:Transaction)
RETURN count(t);
```

Then check the GraphQL API health endpoint:

```text
http://localhost:8000/health
```

## License

This project is intended for educational and portfolio purposes. Add a license file if you want to define explicit usage permissions.

## Author

Lautaro Medina
