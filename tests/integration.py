import json
import time
from uuid import uuid4

import pandas as pd

from fraud_detection.database import connect, recent_results
from fraud_detection.messaging import create_consumer, create_producer, publish
from fraud_detection.scorer import Scorer


def wait_for_rows(
    ids: list[str],
) -> dict[str, dict]:
    deadline = time.monotonic() + 60
    with connect() as connection:
        while time.monotonic() < deadline:
            rows = connection.execute(
                'SELECT transaction_id, score, fraud_flag FROM scores '
                'WHERE transaction_id = ANY(%s::uuid[])',
                (ids,),
            ).fetchall()
            if len(rows) == len(ids):
                return {str(row['transaction_id']): row for row in rows}
            time.sleep(0.5)
    raise TimeoutError(f'Only {len(rows)} of {len(ids)} transactions stored')


def main() -> None:
    producer = create_producer()
    consumer = create_consumer('verification-' + str(uuid4()), 'scores')
    model = Scorer()
    rows = pd.read_csv('data/demo.csv').to_dict('records')
    ids = [str(uuid4()) for _ in rows]
    expected = {key: model.score(row) for key, row in zip(ids, rows, strict=True)}
    try:
        for key, row in zip(ids, rows, strict=True):
            publish(producer, 'transactions', {'transaction_id': key, 'data': row})
        stored = wait_for_rows(ids)
        for key, (score, flag) in expected.items():
            assert abs(stored[key]['score'] - score) < 1e-10
            assert stored[key]['fraud_flag'] == flag
        observed = set()
        deadline = time.monotonic() + 30
        while len(observed) < len(ids) and time.monotonic() < deadline:
            message = consumer.poll(1)
            if message is None:
                continue
            if message.error():
                raise RuntimeError(str(message.error()))
            payload = json.loads(message.value())
            if payload['transaction_id'] in expected:
                assert set(payload) == {'transaction_id', 'score', 'fraud_flag'}
                observed.add(payload['transaction_id'])
        assert len(observed) == len(ids), 'Missing scores in Kafka'
        publish(producer, 'transactions', {'transaction_id': ids[0], 'data': rows[0]})
        invalid_id = str(uuid4())
        publish(producer, 'transactions', {'transaction_id': invalid_id, 'data': {}})
        marker_id = str(uuid4())
        publish(
            producer, 'transactions', {'transaction_id': marker_id, 'data': rows[0]}
        )
        wait_for_rows([marker_id])
        with connect() as connection:
            assert (
                connection.execute(
                    'SELECT COUNT(*) AS count FROM scores WHERE transaction_id = %s',
                    (ids[0],),
                ).fetchone()['count']
                == 1
            )
            assert (
                connection.execute(
                    'SELECT COUNT(*) AS count FROM scores WHERE transaction_id = %s',
                    (invalid_id,),
                ).fetchone()['count']
                == 0
            )
        fraud, scores = recent_results()
        assert len(fraud) == 10 and all(row['fraud_flag'] == 1 for row in fraud)
        assert len(scores) == 100
        print(
            'OK: scoring, Kafka, PostgreSQL, latest 10/100, duplicates, invalid input'
        )
    finally:
        consumer.close()


if __name__ == '__main__':
    main()
