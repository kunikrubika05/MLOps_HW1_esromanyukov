import json
import unittest
from unittest.mock import Mock
from uuid import uuid4

import numpy as np
import pandas as pd

from fraud_detection.preprocessing import preprocess
from fraud_detection.scorer import Scorer
from fraud_detection.services.scoring import score_messages


class ScorerTests(unittest.TestCase):
    def setUp(
        self,
    ) -> None:
        self.scorer = Scorer()
        self.rows = pd.read_csv('data/demo.csv').head(10)

    def test_batch_and_stream_agree(
        self,
    ) -> None:
        expected = self.scorer.model.predict_proba(
            preprocess(self.rows, self.scorer.state),
            thread_count=1,
        )[:, 1]
        actual = [self.scorer.score(row)[0] for row in self.rows.to_dict('records')]
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
        batch = self.scorer.score_many(self.rows.to_dict('records'))
        np.testing.assert_allclose([row[0] for row in batch], expected, atol=1e-12)
        self.assertEqual([row[1] for row in batch], [int(x > 0.98) for x in expected])

    def test_invalid_message_inside_batch(
        self,
    ) -> None:
        ids = [str(uuid4()) for _ in range(3)]
        row = self.rows.iloc[0].to_dict()
        messages = [
            Mock(
                value=Mock(return_value=json.dumps(payload).encode()),
                error=Mock(return_value=None),
            )
            for payload in [
                {'transaction_id': ids[0], 'data': row},
                {'transaction_id': ids[1], 'data': {}},
                {'transaction_id': ids[2], 'data': row},
            ]
        ]
        results = score_messages(self.scorer, messages)
        self.assertEqual([x['transaction_id'] for x in results], [ids[0], ids[2]])
        self.assertEqual(results[0]['score'], results[1]['score'])

    def test_missing_values_and_unknown_category(
        self,
    ) -> None:
        row = self.rows.iloc[0].to_dict()
        row.update(amount=None, gender=None, merch='unseen-merchant', lat=None)
        score, flag = self.scorer.score(row)
        self.assertTrue(0 <= score <= 1)
        self.assertIn(flag, (0, 1))

    def test_invalid_transaction(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            self.scorer.score({})
        row = self.rows.iloc[0].to_dict()
        row['amount'] = -1
        with self.assertRaises(ValueError):
            self.scorer.score(row)

    def test_mixed_timezones_are_rejected(
        self,
    ) -> None:
        rows = self.rows.head(2).to_dict('records')
        rows[0]['transaction_time'] = '2025-01-01T00:00:00+00:00'
        rows[1]['transaction_time'] = '2025-01-01T00:00:00+03:00'
        with self.assertRaisesRegex(ValueError, 'consistent timezone'):
            self.scorer.score_many(rows)


if __name__ == '__main__':
    unittest.main()
