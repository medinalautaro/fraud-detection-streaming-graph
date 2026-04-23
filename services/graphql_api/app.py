import os
from typing import List, Optional

import strawberry
from fastapi import FastAPI
from neo4j import GraphDatabase
from strawberry.fastapi import GraphQLRouter


NEO4J_URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD),
)


@strawberry.type
class TimeBucketType:
    bucket_id: str
    start_time: float
    end_time: float


@strawberry.type
class AmountBucketType:
    bucket_id: str
    min_amount: float
    max_amount: Optional[float]


@strawberry.type
class LabelType:
    value: int


@strawberry.type
class TransactionType:
    transaction_id: int
    time: float
    amount: float
    class_value: int

    time_bucket: Optional[TimeBucketType]
    amount_bucket: Optional[AmountBucketType]
    label: Optional[LabelType]


def to_transaction(record) -> TransactionType:
    t = record["t"]
    return TransactionType(
        transaction_id=t["transaction_id"],
        time=float(t["time"]),
        amount=float(t["amount"]),
        class_value=int(t["class"]),
    )


def get_transaction_by_id(transaction_id: int) -> Optional[TransactionType]:
    query = """
    MATCH (t:Transaction {transaction_id: $transaction_id})
    OPTIONAL MATCH (t)-[:IN_TIME_BUCKET]->(tb:TimeBucket)
    OPTIONAL MATCH (t)-[:IN_AMOUNT_BUCKET]->(ab:AmountBucket)
    OPTIONAL MATCH (t)-[:HAS_LABEL]->(l:Label)
    RETURN t, tb, ab, l
    LIMIT 1
    """

    with driver.session() as session:
        r = session.run(query, {"transaction_id": transaction_id}).single()

        if not r:
            return None

        t = r["t"]
        tb = r["tb"]
        ab = r["ab"]
        l = r["l"]

        return TransactionType(
            transaction_id=t["transaction_id"],
            time=float(t["time"]),
            amount=float(t["amount"]),
            class_value=int(t["class"]),
            time_bucket=TimeBucketType(
                bucket_id=tb["bucket_id"],
                start_time=float(tb["start_time"]),
                end_time=float(tb["end_time"]),
            ) if tb else None,
            amount_bucket=AmountBucketType(
                bucket_id=ab["bucket_id"],
                min_amount=float(ab["min_amount"]),
                max_amount=float(ab["max_amount"]) if ab.get("max_amount") is not None else None,
            ) if ab else None,
            label=LabelType(
                value=int(l["value"])
            ) if l else None,
        )


def get_transactions(limit: int) -> List[TransactionType]:
    query = """
    MATCH (t:Transaction)
    OPTIONAL MATCH (t)-[:IN_TIME_BUCKET]->(tb:TimeBucket)
    OPTIONAL MATCH (t)-[:IN_AMOUNT_BUCKET]->(ab:AmountBucket)
    OPTIONAL MATCH (t)-[:HAS_LABEL]->(l:Label)
    RETURN t, tb, ab, l
    ORDER BY t.transaction_id DESC
    LIMIT $limit
    """

    with driver.session() as session:
        records = session.run(query, {"limit": limit})
        result = []

        for r in records:
            t = r["t"]
            tb = r["tb"]
            ab = r["ab"]
            l = r["l"]

            result.append(
                TransactionType(
                    transaction_id=t["transaction_id"],
                    time=float(t["time"]),
                    amount=float(t["amount"]),
                    class_value=int(t["class"]),
                    time_bucket=TimeBucketType(
                        bucket_id=tb["bucket_id"],
                        start_time=tb["start_time"],
                        end_time=tb["end_time"],
                    ) if tb else None,
                    amount_bucket=AmountBucketType(
                        bucket_id=ab["bucket_id"],
                        min_amount=ab["min_amount"],
                        max_amount=ab.get("max_amount"),
                    ) if ab else None,
                    label=LabelType(value=l["value"]) if l else None,
                )
            )

        return result


def get_fraud_transactions(limit: int) -> List[TransactionType]:
    query = """
    MATCH (t:Transaction)-[:HAS_LABEL]->(l:Label {value: 1})
    OPTIONAL MATCH (t)-[:IN_TIME_BUCKET]->(tb:TimeBucket)
    OPTIONAL MATCH (t)-[:IN_AMOUNT_BUCKET]->(ab:AmountBucket)
    RETURN t, tb, ab, l
    ORDER BY t.transaction_id DESC
    LIMIT $limit
    """

    with driver.session() as session:
        records = session.run(query, {"limit": limit})
        result = []

        for r in records:
            t = r["t"]
            tb = r["tb"]
            ab = r["ab"]
            l = r["l"]

            result.append(
                TransactionType(
                    transaction_id=t["transaction_id"],
                    time=float(t["time"]),
                    amount=float(t["amount"]),
                    class_value=int(t["class"]),
                    time_bucket=TimeBucketType(
                        bucket_id=tb["bucket_id"],
                        start_time=float(tb["start_time"]),
                        end_time=float(tb["end_time"]),
                    ) if tb else None,
                    amount_bucket=AmountBucketType(
                        bucket_id=ab["bucket_id"],
                        min_amount=float(ab["min_amount"]),
                        max_amount=float(ab["max_amount"]) if ab.get("max_amount") is not None else None,
                    ) if ab else None,
                    label=LabelType(
                        value=int(l["value"])
                    ) if l else None,
                )
            )

        return result

@strawberry.type
class Query:
    @strawberry.field
    def transaction(self, transaction_id: int) -> Optional[TransactionType]:
        return get_transaction_by_id(transaction_id)

    @strawberry.field
    def transactions(self, limit: int = 20) -> List[TransactionType]:
        return get_transactions(limit)

    @strawberry.field
    def fraud_transactions(self, limit: int = 20) -> List[TransactionType]:
        return get_fraud_transactions(limit)


schema = strawberry.Schema(query=Query)
graphql_router = GraphQLRouter(schema)

app = FastAPI(title="Fraud GraphQL API")
app.include_router(graphql_router, prefix="/graphql")


@app.get("/health")
def health():
    with driver.session() as session:
        session.run("RETURN 1").single()
    return {"status": "ok"}