"""UI integration checks; require optional UI dependencies and local experiment."""
import importlib.util
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
READY=importlib.util.find_spec('streamlit') is not None and (ROOT/'models/msft/ridge__market.joblib').exists() and (ROOT/'reports/msft_interpretability.json').exists()


@unittest.skipUnless(READY,'Dashboard dependencies and prepared local experiment required')
class DashboardTests(unittest.TestCase):
    def app(self):
        from streamlit.testing.v1 import AppTest
        return AppTest.from_file(str(ROOT/'scripts/dashboard_msft.py'),default_timeout=60).run()

    def test_filters_and_arbitrary_date_explanation(self):
        app=self.app()
        self.assertFalse(app.exception)
        app.date_input(key='start_date').set_value(date(2026,10,1))
        app.date_input(key='end_date').set_value(date(2026,10,15))
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(next(m.value for m in app.metric if m.label=='Торговых сессий'),'11')
        app.selectbox(key='explain_date').select('2026-10-02').run()
        self.assertFalse(app.exception)
        self.assertTrue(any('Проверка суммы' in x.value for x in app.markdown))
        app.selectbox(key='pdp_feature').select('log_volume').run()
        app.selectbox(key='importance_period').select('Вторая половина').run()
        self.assertFalse(app.exception)

    def test_empty_and_reversed_date_intervals(self):
        app=self.app()
        app.date_input(key='start_date').set_value(date(2026,10,10))
        app.date_input(key='end_date').set_value(date(2026,10,11))
        app.run()
        self.assertFalse(app.exception)
        self.assertTrue(any('нет торговых сессий' in w.value for w in app.warning))
        app.date_input(key='end_date').set_value(date(2026,10,9)).run()
        self.assertTrue(any('не позже конца' in w.value for w in app.warning))

    def test_failed_update_is_visible_without_breaking_research(self):
        app=self.app()
        with patch('stocks_forecast.msft_live.update',side_effect=OSError('test outage')):
            app.button(key='refresh_live').click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any('test outage' in e.value for e in app.error))
        self.assertEqual(next(m.value for m in app.metric if m.label=='Торговых сессий'),'252')


if __name__=='__main__':
    unittest.main()
