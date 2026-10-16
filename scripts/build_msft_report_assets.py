"""Generate report tables and static SVG plots from retained project results.

Run with the document workspace Python runtime (pandas, reportlab).
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from reportlab.graphics.shapes import Drawing, String, Line, Rect, Polygon
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.charts.barcharts import HorizontalBarChart
from reportlab.graphics import renderSVG
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/figures';OUT.mkdir(exist_ok=True)
pdfmetrics.registerFont(TTFont('Arial',r'C:/Windows/Fonts/arial.ttf'))
BLUE=colors.HexColor('#2874a6');ORANGE=colors.HexColor('#b9781a');DARK=colors.HexColor('#243645');GREEN=colors.HexColor('#4d917e')
def load(p):return json.loads((ROOT/p).read_text(encoding='utf-8'))
cfg=load('config/msft_experiment.json');snap=load('data/processed/msft/snapshot.json')
prices=pd.DataFrame(snap['prices']);pred=pd.read_csv(ROOT/'data/processed/msft/test_predictions.csv')
quality=load('reports/msft_data_quality.json');val=load('reports/msft_validation.json');test=load('reports/msft_test.json')
ex=load('reports/msft_interpretability.json');repro=load('reports/msft_reproduction.json')
captions={}

def text(d,x,y,t,size=14,anchor='start',color=DARK):
    d.add(String(x,y,str(t),fontName='Arial',fontSize=size,textAnchor=anchor,fillColor=color))
def save(name,d,caption):
    d.contents.insert(0,Rect(0,0,d.width,d.height,fillColor=colors.white,strokeColor=None))
    renderSVG.drawToFile(d,str(OUT/(name+'.svg')))
    captions[name]=caption
def lineplot(name,series,dates,ylabel,caption):
    d=Drawing(900,385);c=LinePlot();c.x=75;c.y=65;c.width=795;c.height=270
    c.data=[list(enumerate(np.asarray(v,dtype=float))) for _,v,_ in series]
    for i,(_,_,color) in enumerate(series):c.lines[i].strokeColor=color;c.lines[i].strokeWidth=1.3
    c.xValueAxis.valueMin=0;c.xValueAxis.valueMax=len(dates)-1
    c.xValueAxis.valueSteps=np.linspace(0,len(dates)-1,5).round().astype(int).tolist()
    c.xValueAxis.labelTextFormat=lambda v:pd.Timestamp(dates[min(round(v),len(dates)-1)]).strftime('%m.%Y')
    for ax in (c.xValueAxis,c.yValueAxis):
        ax.labels.fontName='Arial';ax.labels.fontSize=13;ax.strokeColor=colors.HexColor('#9ba8b3')
    c.yValueAxis.visibleGrid=True;c.yValueAxis.gridStrokeColor=colors.HexColor('#e2e7eb')
    c.yValueAxis.labelTextFormat=lambda v:f'{v:g}'
    d.add(c);text(d,75,360,ylabel,14)
    xpos=75
    for label,_,col in series:
        d.add(Line(xpos,22,xpos+24,22,strokeColor=col,strokeWidth=2));text(d,xpos+30,17,label,13);xpos+=max(170,len(label)*8+40)
    save(name,d,caption)

def barplot(name,labels,values,xlabel,caption,second=None):
    d=Drawing(900,max(340,len(labels)*31+100));h=d.height
    c=HorizontalBarChart();c.x=330;c.y=60;c.width=540;c.height=h-100
    c.data=[list(values)]+([list(second)] if second is not None else [])
    all_values=list(values)+(list(second) if second is not None else [])
    c.valueAxis.valueMin=min(0,min(all_values)*1.1)
    c.valueAxis.valueMax=max(0,max(all_values)*1.08)
    c.categoryAxis.categoryNames=list(labels);c.categoryAxis.labels.fontName='Arial';c.categoryAxis.labels.fontSize=13
    c.categoryAxis.labels.boxAnchor='e';c.categoryAxis.labels.dx=-8
    c.categoryAxis.labelAxisMode='low'
    c.valueAxis.labels.fontName='Arial';c.valueAxis.labels.fontSize=13
    c.valueAxis.visibleGrid=True;c.valueAxis.gridStrokeColor=colors.HexColor('#e2e7eb')
    c.valueAxis.labelTextFormat=lambda v:f'{v:.3g}'
    c.bars[0].fillColor=BLUE;c.bars[0].strokeColor=None
    if second is not None:c.bars[1].fillColor=GREEN;c.bars[1].strokeColor=None
    c.groupSpacing=9;c.barSpacing=2;d.add(c);text(d,600,20,xlabel,14,anchor='middle')
    if second is not None:
        text(d,340,h-18,'Синий — первая половина; зелёный — вторая',12)
    save(name,d,caption)

lineplot('price_history',[('Цена закрытия',prices.close,BLUE)],prices.trade_date,'USD','Дневная цена закрытия MSFT на всём исследуемом интервале')
lineplot('test_prices',[('Факт',pred.target_close,DARK),('Последнее закрытие',pred.naive_last_close,ORANGE),('Ridge',pred.ridge__market,BLUE)],pred.target_date,'USD','Факт и прогнозы на отложенном тестовом периоде')
lineplot('test_errors',[('Последнее закрытие',abs(pred.target_close-pred.naive_last_close),ORANGE),('Ridge',abs(pred.target_close-pred.ridge__market),BLUE)],pred.target_date,'Абсолютная ошибка, USD','Абсолютные ошибки двух прогнозов по тестовым датам')
barplot('text_coverage',['Train','Validation','Test'],[100*1090/1684,100*122/252,100*138/252],'Строки с текстом в семидневном окне, %','Покрытие временных частей текстовой информацией')
short={'naive_last_close':'Последнее закрытие','ridge__price_only':'Ridge — цены','ridge__market':'Ridge — рынок','ridge__market_text':'Ridge — рынок и тексты','random_forest__price_only':'Лес — цены','random_forest__market':'Лес — рынок','random_forest__market_text':'Лес — рынок и тексты'}
barplot('model_mae',[short[r['name']] for r in reversed(test['results'])],[r['metrics']['mae_usd'] for r in reversed(test['results'])],'MAE, USD','Средняя абсолютная ошибка всех вариантов на test')
labels={'return_1':'Доходность за 1 сессию','return_2':'Доходность за 2 сессии','return_5':'Доходность за 5 сессий','return_20':'Доходность за 20 сессий',
        'return_lag_1':'Доходность с лагом 1','return_lag_2':'Доходность с лагом 2','return_lag_5':'Доходность с лагом 5',
        'close_to_mean_5':'Цена к средней за 5 сессий','close_to_mean_20':'Цена к средней за 20 сессий','volatility_5':'Волатильность 5 сессий','volatility_20':'Волатильность 20 сессий',
        'intraday_range':'Диапазон свечи','intraday_body':'Тело свечи','opening_gap':'Разрыв открытия','log_volume':'Логарифм объёма',
        'volume_to_mean_5':'Объём к средней за 5 сессий','volume_to_mean_20':'Объём к средней за 20 сессий','volume_change':'Изменение объёма','weekday':'День недели','month':'Месяц'}
for key,metric,caption,axis in [('contribution_importance','mean_abs_usd','Средний абсолютный вклад основных признаков Ridge','Средний модуль вклада, USD'),('permutation_importance','delta_mae_usd','Блочная перестановочная важность на полном test','Изменение MAE, USD')]:
    selected=sorted(ex['periods']['full'][key],key=lambda r:r[metric],reverse=True)[:10][::-1]
    barplot(key,[labels[r['feature']] for r in selected],[r[metric] for r in selected],axis,caption)
local=ex['local_explanations']['example'];rows=local['contributions'][:8][::-1]
barplot('local_explanation',[labels[r['feature']] for r in rows],[r['contribution_usd'] for r in rows],'Вклад, USD','Крупнейшие вклады в прогноз закрытия 15 октября 2026 года')
curve=ex['pdp']['return_1'];d=Drawing(900,355);c=LinePlot();c.x=80;c.y=65;c.width=780;c.height=240
c.data=[[(100*r['value'],r['mean_predicted_change_bps']) for r in curve]];c.lines[0].strokeColor=BLUE;c.lines[0].strokeWidth=2
for ax in (c.xValueAxis,c.yValueAxis):ax.labels.fontName='Arial';ax.labels.fontSize=14
c.yValueAxis.visibleGrid=True;c.yValueAxis.gridStrokeColor=colors.HexColor('#e2e7eb');d.add(c)
text(d,80,330,'Среднее прогнозируемое изменение, б.п.',14);text(d,470,20,'Доходность предыдущей сессии, %',14,'middle');save('pdp',d,'Частичная зависимость прогноза Ridge от однодневной доходности')
first={r['feature']:r['mean_abs_bps'] for r in ex['periods']['first_half']['contribution_importance']};second={r['feature']:r['mean_abs_bps'] for r in ex['periods']['second_half']['contribution_importance']}
keys=sorted(first,key=lambda k:first[k]+second[k],reverse=True)[:8][::-1]
barplot('period_importance',[labels[k] for k in keys],[first[k] for k in keys],'Средний модуль вклада, б.п.','Сопоставление вкладов одной модели в двух половинах test',[second[k] for k in keys])
d=Drawing(900,300)
for x,title,sub in [(15,'Источники','Yahoo и Microsoft'),(195,'Загрузка','Версии и SQLite'),(375,'Признаки','Время и окна'),(555,'Эксперимент','Выбор и test'),(735,'Интерфейс','Ошибки и вклады')]:
    d.add(Rect(x,140,150,82,fillColor=colors.HexColor('#eef3f7'),strokeColor=BLUE));text(d,x+75,190,title,15,'middle');text(d,x+75,165,sub,11,'middle')
    if x<735:d.add(Line(x+150,180,x+179,180,strokeColor=BLUE));d.add(Polygon([x+179,180,x+170,184,x+170,176],fillColor=BLUE,strokeColor=BLUE))
text(d,450,80,'Замороженный эксперимент и рабочий расчёт сохраняются отдельно',16,'middle');save('architecture',d,'Основные компоненты конвейера MSFT')
d=Drawing(900,300);d.add(Line(70,150,830,150,strokeColor=DARK,strokeWidth=2))
for x,title,sub in [(100,'Предыдущее закрытие','OHLCV завершённой сессии'),(430,'09:00 Нью-Йорк','Отсечка текстов и прогноз'),(780,'Следующее закрытие','Появление целевой цены')]:
    d.add(Line(x,140,x,160,strokeColor=BLUE,strokeWidth=2));text(d,x,190,title,14,'middle');text(d,x,115,sub,11,'middle')
text(d,450,55,'Текстовое окно — семь календарных дней до отсечки включительно',15,'middle');save('timeline',d,'Временная доступность входов и целевого закрытия')

tables={}
def table(key,caption,header,rows):tables[key]={'caption':caption,'header':header,'rows':rows}
def num(x,n=4):return f'{x:.{n}f}'.replace('.',',')
table('requirements','Проверяемые результаты технической части',['Требование','Реализация'],[
['Два разнородных источника','Котировки JSON и публикации HTML'],['Обновление и хранение','Инкрементальный сбор, версии и исходные ответы'],['Единые признаки','Одна функция для истории и рабочего расчёта'],['Сравнение моделей','Baseline, Ridge, случайный лес'],['Качество прогноза','MAE и RMSE в USD, дополнительная MAPE'],['Интерпретируемость','Два метода важности, отдельный прогноз, PDP, периоды'],['Интерфейс','Фильтры, графики, объяснения и обновление'],['Воспроизводимость','Зафиксированные зависимости, снимок и протокол']])
table('sources','Характеристики двух источников',['Характеристика','Котировки','Пресс-релизы'],[['Формат','JSON','HTML'],['Объём снимка','2 209 сессий','526 публикаций'],['Основной ключ','Тикер и дата сессии','URL публикации'],['Временные поля','Дата свечи, время получения','Публикация, изменение, получение'],['Использование','Цель и рыночные признаки','Статистики окна, категория, TF-IDF'],['Ограничение','Исправления истории','Текущие редакции; только новости компании']])
table('storage','Основные сущности хранения',['Сущность','Назначение'],[['raw_responses','Метаданные и хеш исходного ответа'],['observations','Наблюдения загрузки источника'],['price_versions','Версии дневных котировок'],['article_versions','Версии заголовка, текста и дат'],['article_queue','Очередь новых и неудачных URL'],['corporate_actions','Дивиденды и дробления'],['runs и errors','Статусы запусков и причины ошибок'],['prices и articles','Представления актуальных наблюдавшихся версий']])
table('quality','Проверка исследовательского снимка',['Показатель','Результат'],[['Ожидаемые и полученные сессии','2 209 / 2 209'],['Пропуски и лишние даты','0 / 0'],['Проверка диапазона OHLCV','Пройдена'],['Дивидендные события','35'],['Дробления в периоде','0'],['Использование adjusted close','Исключён из модели'],['Размеченные строки после прогрева','2 188'],['Последняя строка без метки','Закрытие 16.10.2026']])
table('feature_groups','Группы признаков для сравнения',['Группа','Состав','Числовые входы'],[['price_only','Ценовые лаги и окна','14'],['market','Цены, объём, день недели и месяц','18 и 2 категории'],['market_text','Рыночные и текстовые признаки','22, 3 категории и до 300 терминов']])
table('models','Конфигурации моделей',['Модель','Настройки','Обучаемая цель'],[['Последнее закрытие','Без параметров обучения','Не требуется'],['Ridge','alpha ∈ {1, 100, 1000}; solver lsqr','Изменение следующей цены'],['Случайный лес','200 деревьев; глубина 8; лист 15; seed 42','Изменение следующей цены']])
table('splits','Временное разбиение по целевой сессии',['Часть','Начало','Конец','Строк'],[['Train','01.02.2018','10.10.2024','1 684'],['Validation','11.10.2024','14.10.2025','252'],['Test','15.10.2025','15.10.2026','252']])
table('validation','Результаты выбора на validation',['Вариант','alpha','MAE USD','RMSE USD'],[[short[r['name']],'-' if r['alpha'] is None else str(int(r['alpha'])),num(r['metrics']['mae_usd']),num(r['metrics']['rmse_usd'])] for r in val['candidates']])
table('test','Итоговое сравнение на test',['Вариант','MAE USD','RMSE USD','MAPE %'],[[short[r['name']],num(r['metrics']['mae_usd']),num(r['metrics']['rmse_usd']),num(r['metrics']['mape_percent'])] for r in test['results']])
table('text_effect','Изменение MAE при добавлении текстов',['Семейство','Validation USD','Test USD'],[[f,num(val['text_effect'][key]['mae_difference_text_minus_market'],6),num(test['text_effect'][key]['mae_difference_text_minus_market'],6)] for f,key in [('Ridge','ridge'),('Случайный лес','random_forest')]])
tmp=pred.assign(error=abs(pred.target_close-pred.naive_last_close)).nlargest(3,'error')
table('largest_errors','Три наибольшие ошибки базового прогноза',['Дата','Факт USD','База USD','Ridge USD','Ошибка базы'],[[r.target_date,num(r.target_close,2),num(r.naive_last_close,2),num(r.ridge__market,2),num(r.error,2)] for r in tmp.itertuples()])
table('periods','Качество по двум половинам test',['Период','Сессий','MAE базы','MAE Ridge'],[[p['start']+' — '+p['end'],str(p['rows']),num(p['naive_metrics']['mae_usd']),num(p['ridge_metrics']['mae_usd'])] for p in [ex['periods']['first_half'],ex['periods']['second_half']]])
table('dashboard','Вопросы аналитика и элементы интерфейса',['Вопрос','Элемент'],[['Насколько точен прогноз?','MAE, график факта и прогнозов, таблица моделей'],['Когда ошибки особенно велики?','График абсолютных ошибок и фильтр дат'],['Почему получена конкретная цена?','Выбор даты, вклады и проверка суммы'],['Какие признаки заметны и полезны?','Два метода важности и PDP'],['Меняется ли поведение по периодам?','Сравнение половин года'],['Когда создан новый расчёт?','Время формирования, отсечка, последняя свеча']])
table('tests','Автоматические проверки по группам',['Группа','Число','Основной предмет'],[['Аудитор','4','Парсинг и качество источников'],['Загрузка','9','Версии, повторы и восстановление'],['Признаки','9','Календарь и временные границы'],['Объяснения и рабочее обновление','6','Точность разложения, PDP, сбой'],['Интерфейс','3','Фильтры, выбор даты, сообщение об ошибке'],['Восстановление моделей','2','Сохранность отчёта и отказ при несовпадении']])
table('reproduction','Результат восстановления в новом окружении',['Проверка','Результат'],[['Python','3.12.8'],['Установка зависимостей','Без конфликтов'],['Исходные данные','Сохранённый JSON-снимок'],['Использование готовой SQLite','Нет'],['Заново обученные модели',str(repro['model_files_rebuilt'])],['Проверенные тестовые даты',str(repro['test_rows'])],['Максимальное различие прогнозов','0,0 USD'],['Опубликованный test','Сохранён без изменения'],['Тесты в отдельной копии','33 пройдены']])
descriptions=['Изменение close за 1 сессию','Изменение close за 2 сессии','Изменение close за 5 сессий','Изменение close за 20 сессий','Однодневная доходность со сдвигом 1','Однодневная доходность со сдвигом 2','Однодневная доходность со сдвигом 5','Отношение close к средней 5 минус 1','Отношение close к средней 20 минус 1','Стандартное отклонение доходностей за 5 сессий','Стандартное отклонение доходностей за 20 сессий','(high − low) / close последней свечи','close / open − 1 последней свечи','open / предыдущий close − 1','Натуральный логарифм величины 1 + объём','Объём к средней 5 минус 1','Объём к средней 20 минус 1','Относительное изменение объёма','Категория дня недели целевой сессии','Категория месяца целевой сессии']
table('features_full','Словарь рыночных признаков',['Имя в коде','Содержание'],[[key,desc] for key,desc in zip(labels,descriptions)])
table('commands','Основные команды после установки окружения',['Этап','Скрипт и параметры'],[['Загрузка','scripts/ingest_msft.py --end 2026-10-15'],['Признаки','scripts/prepare_msft.py'],['Выбор','scripts/train_msft.py --stage validate'],['Test','scripts/train_msft.py --stage test'],['Восстановление','scripts/train_msft.py --stage test --rebuild-artifacts'],['Объяснения','scripts/explain_msft.py'],['Обновление','scripts/update_msft.py --refresh'],['Интерфейс','python -m streamlit run scripts/dashboard_msft.py'],['Проверки','python -m unittest discover -s tests -v']])
table('artifacts','Назначение основных каталогов',['Каталог','Содержимое'],[['config/','Параметры загрузки и эксперимента'],['src/stocks_forecast/','Исходный код'],['scripts/','Команды запуска'],['tests/','Автоматические проверки'],['reports/','Сводки, протокол и итоговый отчёт'],['data/raw/msft/','Локальные исходные ответы'],['data/processed/msft/','Замороженный снимок и признаки'],['models/msft/','Исследовательские модели'],['data/processed/msft_live/','Рабочая оценка и журнал'],['models/msft_live/','Рабочая модель Ridge']])
(ROOT/'reports/report_tables.json').write_text(json.dumps({'tables':tables,'figures':captions},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(f'Generated {len(tables)} tables and {len(captions)} SVG figures')
