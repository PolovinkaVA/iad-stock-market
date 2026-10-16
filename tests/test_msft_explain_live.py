"""Numerical explanations and operational failure isolation."""
import json
import shutil
import sys
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from stocks_forecast.msft_features import PRICE_FEATURES, VOLUME_FEATURES
from stocks_forecast.msft_modeling import model_pipeline, predict_close
from stocks_forecast.msft_explain import (reference, explain_row, partial_dependence,
                                         permutation_importance, block_indices)
from stocks_forecast.msft_live import calculation_timing, update


class ExplanationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cfg=json.loads((ROOT/'config/msft_experiment.json').read_text())
        rng=np.random.default_rng(123)
        frame=pd.DataFrame({k:rng.normal(0,.01,180) for k in PRICE_FEATURES+VOLUME_FEATURES})
        frame['weekday']=[str(i%5) for i in range(len(frame))]
        frame['month']=[str(i%12+1) for i in range(len(frame))]
        frame['close_at_forecast']=100.
        frame['target_return']=2*frame['return_1']
        frame['target_close']=100*(1+frame['target_return'])
        frame['target_date']='2026-10-16'
        frame['forecast_at']='2026-10-16T13:00:00Z'
        cls.model=model_pipeline('ridge','market',cfg,alpha=.0001)
        cls.model.fit(frame.iloc[:120],frame['target_return'].iloc[:120])
        cls.fit=frame.iloc[:120];cls.test=frame.iloc[120:]
        cls.center=reference(cls.model,cls.fit)

    def test_contributions_reconstruct_unseen_predictions(self):
        predictions=predict_close(self.model,self.test)
        for row,expected in zip(self.test.to_dict('records'),predictions):
            detail=explain_row(self.model,row,self.center)
            total=detail['base_price']+sum(v['contribution_usd'] for v in detail['contributions'])
            self.assertAlmostEqual(total,expected,places=10)

    def test_permutation_detects_known_signal_and_is_repeatable(self):
        first=permutation_importance(self.model,self.test,repeats=3,block_size=10)
        second=permutation_importance(self.model,self.test,repeats=3,block_size=10)
        self.assertEqual(first,second)
        ranked=sorted(first,key=lambda r:r['delta_mae_usd'],reverse=True)
        self.assertEqual(ranked[0]['feature'],'return_1')
        self.assertGreater(ranked[0]['delta_mae_usd'],1.)
        self.assertLess(max(abs(r['delta_mae_usd']) for r in ranked[1:]),.001)

    def test_pdp_recovers_known_linear_response(self):
        curve=partial_dependence(self.model,self.fit,'return_1')
        slope=(curve[-1]['mean_predicted_change_bps']-curve[0]['mean_predicted_change_bps'])/(curve[-1]['value']-curve[0]['value'])
        self.assertAlmostEqual(slope,20000.,delta=1.)

    def test_block_shuffle_retains_every_observation_once(self):
        indices=block_indices(53,np.random.default_rng(42),20)
        self.assertEqual(sorted(indices.tolist()),list(range(53)))
        for block in (list(range(20)),list(range(20,40)),list(range(40,53))):
            start=indices.tolist().index(block[0])
            self.assertEqual(indices[start:start+len(block)].tolist(),block)


class OperationalTests(unittest.TestCase):
    def test_timing_marks_late_and_exact_cutoff_calculations(self):
        cutoff='2026-10-16T09:00:00-04:00'
        self.assertEqual(calculation_timing('2026-10-16T12:59:59Z',cutoff),'before_cutoff')
        self.assertEqual(calculation_timing('2026-10-16T13:00:00Z',cutoff),'after_cutoff')
        self.assertEqual(calculation_timing('2026-10-16T14:00:00Z',cutoff),'after_cutoff')

    def test_failed_source_refresh_preserves_previous_estimate_and_logs_failure(self):
        base=(ROOT/'tmp/live_tests').resolve();base.mkdir(parents=True,exist_ok=True)
        root=base/uuid.uuid4().hex;root.mkdir()
        try:
            (root/'config').mkdir()
            (root/'config/msft.json').write_text('{}')
            folder=root/'data/processed/msft_live';folder.mkdir(parents=True)
            latest=folder/'latest.json';latest.write_text('{"previous": true}')
            with patch('stocks_forecast.msft_live.Ingestion') as factory, patch('stocks_forecast.msft_live.calculate') as calculate:
                factory.return_value.update_prices.side_effect=OSError('source unavailable')
                factory.return_value.finish.return_value={'status':'partial'}
                with self.assertRaises(RuntimeError):
                    update({},root=root,refresh=True)
                calculate.assert_not_called()
            self.assertEqual(latest.read_text(),'{"previous": true}')
            record=json.loads((folder/'runs.jsonl').read_text())
            self.assertEqual(record['status'],'failed')
            self.assertIn('incomplete',record['error'])
        finally:
            assert root.resolve().parent==base
            shutil.rmtree(root)


if __name__=='__main__':
    unittest.main()
