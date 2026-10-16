"""Restoring artifacts must preserve published results and reject mismatches."""
import json
import shutil
import sys
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from stocks_forecast.msft_features import sha, write_json
from stocks_forecast.msft_modeling import evaluate_test, versions


class RebuildTests(unittest.TestCase):
    def setUp(self):
        self.base=(ROOT/'tmp/rebuild_tests').resolve();self.base.mkdir(parents=True,exist_ok=True)
        self.root=self.base/uuid.uuid4().hex;self.root.mkdir()
        self.cfg={'reports_dir':'reports','processed_dir':'processed','model_dir':'models'}
        (self.root/'processed').mkdir()
        self.metadata={'snapshot_sha256':'fixture','splits':{'test':{'rows':2}}}
        self.protocol={'dataset':self.metadata,'config':self.cfg,'versions':versions(),
                       'candidates':[{'name':'naive_last_close','family':'naive'},
                                     {'name':'ridge__market','family':'ridge','group':'market','alpha':1000},
                                     {'name':'ridge__market_text','family':'ridge','group':'market_text','alpha':1000},
                                     {'name':'random_forest__market','family':'random_forest','group':'market','alpha':None},
                                     {'name':'random_forest__market_text','family':'random_forest','group':'market_text','alpha':None}],
                       'selected_model':'naive_last_close','best_learned_model':'ridge__market'}
        self.protocol['protocol_sha256']=sha(self.protocol)
        write_json(self.root/'reports/msft_protocol.json',self.protocol)
        self.rows=[{'price_date':'2026-10-14','target_date':'2026-10-15','forecast_at':'2026-10-15T13:00Z',
                    'target_close':101.,'close_at_forecast':100.,'target_return':.01}]*2

    def tearDown(self):
        assert self.root.resolve().parent==self.base
        shutil.rmtree(self.root)

    def run_evaluation(self,rebuild):
        from sklearn.dummy import DummyRegressor
        with patch('stocks_forecast.msft_modeling.input_data',return_value=(self.rows,self.metadata)), \
             patch('stocks_forecast.msft_modeling.split_rows',return_value={'train':self.rows,'validation':self.rows,'test':self.rows}), \
             patch('stocks_forecast.msft_modeling.model_pipeline',return_value=DummyRegressor(strategy='constant',constant=0)):
            return evaluate_test(self.cfg,self.root,rebuild_artifacts=rebuild)

    def test_restore_missing_models_keeps_existing_test_report(self):
        self.run_evaluation(False)
        before=(self.root/'reports/msft_test.json').read_bytes()
        for p in (self.root/'models').glob('*.joblib'): p.unlink()
        self.run_evaluation(True)
        self.assertEqual(len(list((self.root/'models').glob('*.joblib'))),4)
        self.assertEqual(before,(self.root/'reports/msft_test.json').read_bytes())

    def test_mismatch_does_not_replace_saved_model(self):
        self.run_evaluation(False)
        model=self.root/'models/ridge__market.joblib';before=model.read_bytes()
        test=json.loads((self.root/'reports/msft_test.json').read_text())
        test['results'][0]['metrics']['mae_usd']=999
        write_json(self.root/'reports/msft_test.json',test)
        with self.assertRaisesRegex(ValueError,'disagree'):
            self.run_evaluation(True)
        self.assertEqual(before,model.read_bytes())


if __name__=='__main__': unittest.main()
