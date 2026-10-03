import json
import os
from typing import Any

from confluent_kafka import Consumer, KafkaError, KafkaException, Producer


def create_consumer(
    group: str,
    topic: str,
) -> Consumer:
    consumer = Consumer(
        {
            'bootstrap.servers': os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092'),
            'group.id': group,
            'auto.offset.reset': 'earliest',
            'enable.auto.commit': False,
        }
    )
    consumer.subscribe([topic])
    return consumer


def create_producer() -> Producer:
    return Producer(
        {
            'bootstrap.servers': os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092'),
            'enable.idempotence': True,
            'delivery.timeout.ms': 10000,
        }
    )


def publish(
    producer: Producer,
    topic: str,
    message: dict[str, Any],
) -> None:
    publish_many(producer, topic, [message])


def publish_many(
    producer: Producer,
    topic: str,
    messages: list[dict[str, Any]],
) -> None:
    errors: list[KafkaError] = []
    for message in messages:
        producer.produce(
            topic,
            key=str(message['transaction_id']),
            value=json.dumps(message, allow_nan=False).encode(),
            on_delivery=lambda error, _: errors.append(error) if error else None,
        )
    if producer.flush(15):
        raise TimeoutError('Kafka delivery timed out')
    if errors:
        raise KafkaException(errors[0])
