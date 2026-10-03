import unittest
from unittest.mock import Mock

from confluent_kafka import KafkaError

from fraud_detection.messaging import DeliveryError, publish_many


class MessagingTests(unittest.TestCase):
    def test_failed_delivery_preserves_confirmed_ids(
        self,
    ) -> None:
        producer = Mock()

        def deliver(
            timeout: float,
        ) -> int:
            for index, call in enumerate(producer.produce.call_args_list):
                error = KafkaError(KafkaError._MSG_TIMED_OUT) if index else None
                message = Mock(key=Mock(return_value=call.kwargs['key'].encode()))
                call.kwargs['on_delivery'](error, message)
            return 0

        producer.flush.side_effect = deliver
        with self.assertRaises(DeliveryError) as caught:
            publish_many(
                producer,
                'transactions',
                [{'transaction_id': 'first'}, {'transaction_id': 'second'}],
            )
        self.assertEqual(caught.exception.delivered_ids, ['first'])


if __name__ == '__main__':
    unittest.main()
