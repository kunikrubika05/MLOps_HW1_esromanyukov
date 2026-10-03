import os
from typing import Any

import psycopg
from psycopg.rows import dict_row


def connect() -> psycopg.Connection:
    return psycopg.connect(
        host=os.getenv('POSTGRES_HOST', 'postgres'),
        dbname=os.getenv('POSTGRES_DB', 'fraud'),
        user=os.getenv('POSTGRES_USER', 'fraud'),
        password=os.getenv('POSTGRES_PASSWORD', 'local-demo-password'),
        connect_timeout=10,
        autocommit=True,
        row_factory=dict_row,
    )


def recent_results() -> tuple[list[dict[str, Any]], list[float]]:
    with connect() as connection:
        with connection.transaction():
            connection.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ')
            fraud = connection.execute(
                'SELECT transaction_id::text AS transaction_id, score, fraud_flag '
                'FROM scores '
                'WHERE fraud_flag = 1 ORDER BY id DESC LIMIT 10',
            ).fetchall()
            scores = connection.execute(
                'SELECT score FROM scores ORDER BY id DESC LIMIT 100',
            ).fetchall()
    return fraud, [row['score'] for row in scores]


def stored_ids(
    ids: list[str],
) -> set[str]:
    with connect() as connection:
        rows = connection.execute(
            'SELECT transaction_id::text AS transaction_id FROM scores '
            'WHERE transaction_id = ANY(%s::uuid[])',
            (ids,),
        ).fetchall()
    return {row['transaction_id'] for row in rows}
