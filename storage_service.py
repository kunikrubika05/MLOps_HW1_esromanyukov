import json
import logging
import math
from uuid import UUID

from confluent_kafka import KafkaException

from database import connect
from messaging import create_consumer


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    consumer = create_consumer('score-storage', 'scores')
    try:
        with connect() as connection:
            logging.info('Storage service ready')
            while True:
                messages = consumer.consume(100, timeout=0.2)
                if not messages:
                    continue
                rows = []
                for message in messages:
                    if message.error():
                        raise KafkaException(message.error())
                    try:
                        payload = json.loads(message.value())
                        transaction_id = UUID(payload['transaction_id'])
                        score = float(payload['score'])
                        flag = payload['fraud_flag']
                        if (
                            not math.isfinite(score)
                            or not 0 <= score <= 1
                            or flag not in (0, 1)
                        ):
                            raise ValueError('Invalid score or fraud flag')
                    except (ValueError, TypeError, KeyError) as error:
                        logging.warning('Invalid scoring result: %s', error)
                        continue
                    rows.append((transaction_id, score, int(flag)))
                with connection.transaction():
                    with connection.cursor() as cursor:
                        cursor.executemany(
                            'INSERT INTO scores (transaction_id, score, fraud_flag) '
                            'VALUES (%s, %s, %s) '
                            'ON CONFLICT (transaction_id) DO NOTHING',
                            rows,
                        )
                consumer.commit(asynchronous=False)
                logging.info('Stored batch of %s transactions', len(rows))
    finally:
        consumer.close()


if __name__ == '__main__':
    main()
