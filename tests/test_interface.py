import unittest
from pathlib import Path
from time import perf_counter
from unittest.mock import patch
from uuid import uuid4

from streamlit.testing.v1 import AppTest


class InterfaceTests(unittest.TestCase):
    def check_results(
        self,
        fraud: list[dict],
        scores: list[float],
    ) -> AppTest:
        with patch(
            'fraud_detection.database.recent_results', return_value=(fraud, scores)
        ):
            app = AppTest.from_file(
                str(Path(__file__).parents[1] / 'fraud_detection/services/interface.py')
            ).run()
            app.button[0].click().run()
        self.assertEqual(len(app.exception), 0)
        return app

    def test_empty_database(
        self,
    ) -> None:
        app = self.check_results([], [])
        self.assertEqual(len(app.info), 2)

    def test_no_fraud_and_less_than_100(
        self,
    ) -> None:
        app = self.check_results([], [0.01] * 17)
        self.assertEqual(len(app.info), 1)
        self.assertIn('17', app.subheader[1].value)
        self.assertEqual(len(app.get('arrow_vega_lite_chart')), 1)

    def test_fraud_table(
        self,
    ) -> None:
        app = self.check_results(
            [{'transaction_id': str(uuid4()), 'score': 0.99, 'fraud_flag': 1}],
            [0.1, 0.99],
        )
        self.assertEqual(len(app.dataframe), 1)
        self.assertEqual(len(app.info), 0)

    def test_scoring_progress_tracks_current_upload(
        self,
    ) -> None:
        ids = [str(uuid4()), str(uuid4())]
        app = AppTest.from_file(
            str(Path(__file__).parents[1] / 'fraud_detection/services/interface.py')
        )
        app.session_state['batch'] = {
            'pending': ids,
            'total': 2,
            'stored': 0,
            'started': perf_counter(),
            'elapsed': None,
            'error': None,
        }
        with patch(
            'fraud_detection.database.stored_ids', return_value={ids[0]}
        ) as lookup:
            app.run()
            lookup.assert_called_once_with(ids)
        self.assertEqual(app.session_state['batch']['stored'], 1)
        self.assertEqual(app.get('progress')[0].proto.value, 50)
        self.assertEqual(len(app.success), 0)
        with patch(
            'fraud_detection.database.stored_ids', return_value={ids[1]}
        ) as lookup:
            app.run()
            lookup.assert_called_once_with([ids[1]])
        self.assertEqual(app.session_state['batch']['stored'], 2)
        self.assertEqual(app.get('progress')[0].proto.value, 100)
        self.assertEqual(len(app.success), 1)

    def test_progress_continues_after_partial_send_error(
        self,
    ) -> None:
        ids = [str(uuid4()), str(uuid4())]
        app = AppTest.from_file(
            str(Path(__file__).parents[1] / 'fraud_detection/services/interface.py')
        )
        app.session_state['batch'] = {
            'pending': ids,
            'total': 3,
            'stored': 0,
            'started': perf_counter(),
            'elapsed': None,
            'error': 'Отправка остановлена: Kafka delivery failed',
        }
        with patch('fraud_detection.database.stored_ids', return_value=set(ids)):
            app.run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state['batch']['stored'], 2)
        self.assertEqual(app.get('progress')[0].proto.value, 66)
        self.assertEqual(len(app.error), 1)
        self.assertFalse(app.success)


if __name__ == '__main__':
    unittest.main()
