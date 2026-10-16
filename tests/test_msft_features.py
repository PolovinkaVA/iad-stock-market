"""Temporal safety and train/inference parity for the MSFT experiment."""
import copy
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from stocks_forecast.msft_features import (build_rows, check_actions, feature_at,
                                          schedule, split_rows, validate_prices)
from stocks_forecast.msft_modeling import model_pipeline, predict_close


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT/'config/msft_experiment.json').read_text())
        self.config.update(start='2024-01-02',end='2024-03-15',validation_sessions=3,test_sessions=3)
        self.sessions = schedule(self.config)
        self.prices = []
        for i, day in enumerate(self.sessions.loc[self.config['start']:self.config['end']].index):
            c = 100+i
            self.prices.append({'trade_date':day.date().isoformat(),'open':c-0.5,'high':c+1,'low':c-1,
                                'close':c,'volume':10000+i*100,'adjusted_close':c*0.99})

    def test_calendar_holidays_early_close_and_dst(self):
        cal = mcal.get_calendar('NASDAQ')
        self.assertTrue(cal.schedule('2025-01-09','2025-01-09').empty)
        self.assertTrue(cal.schedule('2018-12-05','2018-12-05').empty)
        early = cal.schedule('2024-11-29','2024-11-29').iloc[0]['market_close']
        self.assertEqual(early.tz_convert('America/New_York').hour,13)
        march = cal.schedule('2024-03-08','2024-03-11')
        self.assertEqual(march.iloc[0]['market_open'].hour,14)
        self.assertEqual(march.iloc[1]['market_open'].hour,13)

    def test_missing_session_is_not_silently_bridged(self):
        with self.assertRaises(ValueError):
            validate_prices(self.prices[:25]+self.prices[26:],self.sessions,self.config)

    def test_training_row_equals_unlabeled_prefix_row(self):
        full = build_rows({'prices':self.prices,'articles':[]},self.sessions,self.config)
        prefix = build_rows({'prices':self.prices[:26],'articles':[]},self.sessions,self.config)[-1]
        historic = full[5]
        for key,value in prefix.items():
            if key not in ('target_close','target_return'):
                self.assertEqual(value,historic[key],key)
        self.assertIsNone(prefix['target_close'])
        self.assertIsNotNone(historic['target_close'])

    def test_future_prices_do_not_change_past_features(self):
        before = build_rows({'prices':self.prices,'articles':[]},self.sessions,self.config)
        modified = copy.deepcopy(self.prices)
        for row in modified[35:]:
            for field in ('open','high','low','close'):
                row[field] *= 5
        after = build_rows({'prices':modified,'articles':[]},self.sessions,self.config)
        self.assertEqual(before[:14],after[:14])

    def test_text_window_includes_weekend_but_excludes_future_publication(self):
        cutoff = pd.Timestamp('2024-03-18T09:00:00-04:00')
        def item(name, when):
            return {'url':name,'title':'Product launch','text':name,'published_at':when.isoformat()}
        articles = [item('weekend',cutoff-pd.Timedelta(days=1)),item('on_cutoff',cutoff),
                    item('future',cutoff+pd.Timedelta(seconds=1)),item('too_old',cutoff-pd.Timedelta(days=7))]
        row = feature_at(self.prices,articles,cutoff,'2024-03-18',self.config)
        self.assertEqual(row['article_urls'],['weekend','on_cutoff'])
        self.assertEqual(row['news_count_7d'],2)
        self.assertEqual(row['event_category'],'product')

    def test_current_target_price_rejected(self):
        with self.assertRaises(ValueError):
            feature_at(self.prices,[],'2024-03-15T09:00:00-04:00','2024-03-15',self.config)

    def test_splits_require_explicit_adjustment_policy(self):
        payload = {'date':int(pd.Timestamp('2024-02-01T14:30Z').timestamp()),'numerator':2,'denominator':1}
        with self.assertRaises(ValueError):
            check_actions([{'kind':'splits','payload_json':json.dumps(payload)}],self.prices,self.sessions,self.config)

    def test_label_boundary_rejects_unavailable_training_target(self):
        cfg = dict(self.config,validation_sessions=3,test_sessions=3)
        rows=[]
        for day in pd.date_range('2020-01-01',periods=260):
            rows.append({'target_date':day.date().isoformat(),'target_close':10,
                         'forecast_at':(day.tz_localize('UTC')+pd.Timedelta(hours=13)).isoformat(),
                         'target_available_at':(day.tz_localize('UTC')+pd.Timedelta(hours=20)).isoformat()})
        parts=split_rows(rows,cfg)
        self.assertEqual(len(parts['train']),254)
        rows[253]['target_available_at']=rows[254]['forecast_at']
        with self.assertRaises(ValueError):
            split_rows(rows,cfg)

    def test_transforms_do_not_learn_validation_vocabulary(self):
        rows=build_rows({'prices':self.prices,'articles':[]},self.sessions,self.config)
        train=pd.DataFrame(rows[:10]); future=pd.DataFrame(rows[10:12])
        train['news_text']='cloud growth earnings'
        future['news_text']='unseenfutureword'
        cfg=copy.deepcopy(self.config); cfg['tfidf'].update(min_df=1,max_df=1.0)
        model=model_pipeline('ridge','market_text',cfg,alpha=1.0)
        model.fit(train,train['target_return'])
        vocabulary=dict(model.named_steps['features'].named_transformers_['text'].vocabulary_)
        values=predict_close(model,future)
        self.assertTrue(np.isfinite(values).all())
        self.assertNotIn('unseenfutureword',vocabulary)
        self.assertEqual(vocabulary,model.named_steps['features'].named_transformers_['text'].vocabulary_)
        self.assertEqual(model.named_steps['features'].named_transformers_['numeric'].n_samples_seen_,10)


if __name__ == '__main__':
    unittest.main()
