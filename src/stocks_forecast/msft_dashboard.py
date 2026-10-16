"""Read-only research dashboard with an explicit operational refresh action."""
import json
from datetime import date
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .msft_features import ROOT, PRICE_FEATURES, VOLUME_FEATURES
from .msft_modeling import metrics, predict_close
from .msft_explain import LABELS, load_review, reference, explain_row

COLORS={'Факт':'#132c41','Последнее закрытие':'#d18a20','Ridge':'#2874c6'}


def style(fig,height=370):
    fig.update_layout(template='plotly_white',height=height,margin=dict(l=12,r=18,t=20,b=15),
                      font=dict(family='Arial',color='#19344b'),legend=dict(orientation='h',y=1.12,x=0),
                      hovermode='x unified')
    fig.update_xaxes(showgrid=False)
    return fig


def line_chart(frame, columns, title):
    fig=go.Figure()
    for label,column in columns.items():
        fig.add_trace(go.Scatter(x=frame['target_date'],y=frame[column],name=label,mode='lines',
                      line=dict(color=COLORS.get(label),width=2,dash='dot' if label=='Последнее закрытие' else 'solid')))
    fig.update_yaxes(title=title)
    return style(fig)


def input_value(feature,value):
    if feature in ('weekday','month'):
        return str(value)
    if feature=='log_volume':
        return f'{value:.3f}'
    return f'{100*value:.3f}%'


@st.cache_resource(show_spinner=False)
def resources(signature):
    config=json.loads((ROOT/'config/msft_experiment.json').read_text(encoding='utf-8'))
    data=load_review(config)
    data['center']=reference(data['model'],data['fit'])
    report=json.loads((ROOT/config['reports_dir']/'msft_interpretability.json').read_text(encoding='utf-8'))
    if report['protocol_sha256']!=data['protocol']['protocol_sha256'] or report['model_sha256']!=data['model_sha256']:
        raise ValueError('Объяснения относятся к другой модели. Повторите подготовку отчёта.')
    data['explanations']=report
    data['quality']=json.loads((ROOT/config['reports_dir']/'msft_data_quality.json').read_text(encoding='utf-8'))
    data['test_report']=json.loads((ROOT/config['reports_dir']/'msft_test.json').read_text(encoding='utf-8'))
    return config,data


def period_filter(frame):
    first=date.fromisoformat(frame['target_date'].iloc[0]);last=date.fromisoformat(frame['target_date'].iloc[-1])
    st.sidebar.subheader('Период проверки')
    start=st.sidebar.date_input('Начало периода',first,min_value=first,max_value=last,key='start_date')
    end=st.sidebar.date_input('Конец периода',last,min_value=first,max_value=last,key='end_date')
    if start>end:
        st.warning('Начало периода должно быть не позже конца.')
        st.stop()
    filtered=frame[(frame['target_date']>=start.isoformat())&(frame['target_date']<=end.isoformat())].copy()
    if filtered.empty:
        st.warning('В выбранном интервале нет торговых сессий. Измените даты.')
        st.stop()
    return filtered


def forecast_tab(frame,data):
    base=metrics(frame['target_close'],frame['naive']);ridge=metrics(frame['target_close'],frame['ridge'])
    a,b=st.columns(2)
    a.metric('Ошибка базового прогноза',f"${base['mae_usd']:.2f}")
    b.metric('Ошибка Ridge',f"${ridge['mae_usd']:.2f}")
    c,d=st.columns(2)
    change=ridge['mae_usd']-base['mae_usd']
    c.metric('Разница MAE: Ridge − база',f'{change:+.2f} $')
    d.metric('Торговых сессий',len(frame))
    st.caption('MAE — среднее абсолютное отклонение от факта на выбранном периоде. Положительная разница означает большую ошибку Ridge.')
    st.subheader('Насколько прогноз близок к фактической цене?')
    st.plotly_chart(line_chart(frame,{'Факт':'target_close','Последнее закрытие':'naive','Ridge':'ridge'},'Цена закрытия, USD'),width='stretch',key='price_chart')
    error=frame.copy()
    error['naive_error']=(error['target_close']-error['naive']).abs()
    error['ridge_error']=(error['target_close']-error['ridge']).abs()
    st.subheader('Когда прогноз ошибается сильнее?')
    st.plotly_chart(line_chart(error,{'Последнее закрытие':'naive_error','Ridge':'ridge_error'},'Абсолютная ошибка, USD'),width='stretch',key='error_chart')
    st.caption('График ошибок помогает находить трудные периоды. Историческая средняя ошибка не задаёт гарантированный интервал следующей цены.')
    with st.expander('Все варианты на полном отложенном периоде'):
        labels={'naive_last_close':'Последнее закрытие','ridge__price_only':'Ridge · цены',
                'ridge__market':'Ridge · рыночные признаки','ridge__market_text':'Ridge · рыночные признаки и тексты',
                'random_forest__price_only':'Лес · цены','random_forest__market':'Лес · рыночные признаки',
                'random_forest__market_text':'Лес · рыночные признаки и тексты'}
        table=[{'Вариант':labels[r['name']],'MAE, USD':r['metrics']['mae_usd'],
                'RMSE, USD':r['metrics']['rmse_usd'],'MAPE, %':r['metrics']['mape_percent']} for r in data['test_report']['results']]
        st.dataframe(pd.DataFrame(table).round(4),hide_index=True,width='stretch')
        st.caption('Эта таблица относится ко всему тестовому году. Модель выбрана заранее по validation; фильтр дат не меняет выбор.')
    export=frame[['price_date','target_date','forecast_at','target_close','naive','ridge']].to_csv(index=False).encode('utf-8-sig')
    st.download_button('Скачать прогнозы выбранного периода',export,'msft_predictions.csv','text/csv')


def local_tab(frame,data):
    st.subheader('Почему Ridge дал именно эту цену?')
    options=frame['target_date'].tolist()
    if st.session_state.get('explain_date') not in options:
        st.session_state['explain_date']=options[-1]
    selected=st.selectbox('Дата прогнозируемого закрытия',options,key='explain_date')
    row=frame.loc[frame['target_date']==selected].iloc[0].to_dict()
    detail=explain_row(data['model'],row,data['center'])
    a,b,c=st.columns(3)
    a.metric('Факт',f"${detail['actual']:.2f}")
    b.metric('Базовый',f"${detail['last_close']:.2f}")
    c.metric('Ridge',f"${detail['prediction']:.2f}")
    st.caption(f"Отсечка: {pd.Timestamp(detail['forecast_at']).tz_convert('America/New_York').strftime('%d.%m.%Y %H:%M')} Нью-Йорк. Последняя свеча: {row['price_date']}.")
    terms=detail['contributions'];top=terms[:10]
    rest=sum(x['contribution_usd'] for x in terms[10:])
    bar=pd.DataFrame([{'Признак':r['label'],'Вклад':r['contribution_usd']} for r in top]+[{'Признак':'Остальные признаки','Вклад':rest}]).iloc[::-1]
    fig=go.Figure(go.Bar(x=bar['Вклад'],y=bar['Признак'],orientation='h',
                        marker_color=['#2874c6' if x>=0 else '#d18a20' for x in bar['Вклад']],
                        hovertemplate='%{y}: %{x:+.4f} USD<extra></extra>'))
    fig.update_xaxes(title='Поправка к базовому значению Ridge, USD',zeroline=True,zerolinecolor='#758696')
    st.plotly_chart(style(fig,430),width='stretch',key='local_contributions')
    total=sum(x['contribution_usd'] for x in terms)
    st.markdown(f"**Проверка суммы:** последняя цена **{detail['last_close']:.4f}** + средняя поправка модели **{detail['base_adjustment_usd']:+.4f}** + вклады признаков **{total:+.4f}** = **{detail['prediction']:.4f} USD**.")
    st.caption('Вклады рассчитаны относительно средних преобразованных признаков обучающего периода. Это точное разложение линейной модели, а не доказательство причин изменения рынка. Цена последнего закрытия задаёт масштаб прогноза.')
    with st.expander('Все значения и вклады'):
        st.dataframe(pd.DataFrame([{'Признак':r['label'],'Значение':input_value(r['feature'],r['value']),
                                   'Вклад, USD':r['contribution_usd']} for r in terms]),hide_index=True,width='stretch')
    st.info('В выбранный вариант Ridge тексты не входят: по validation он оказался лучше текстового варианта. Тексты участвовали в отдельном сравнении моделей.')


def importance_chart(rows,value,title):
    df=pd.DataFrame(rows).copy();df['label']=df['feature'].map(LABELS)
    df=df.sort_values(value,ascending=False).head(12).sort_values(value)
    fig=go.Figure(go.Bar(x=df[value],y=df['label'],orientation='h',marker_color='#2874c6'))
    fig.update_xaxes(title=title,zeroline=True,zerolinecolor='#7f8c99')
    return style(fig,440)


def factors_tab(data):
    report=data['explanations']
    st.subheader('Какие признаки заметны, а какие помогают точности?')
    periods={'Весь тестовый год':'full','Первая половина':'first_half','Вторая половина':'second_half'}
    selected=st.selectbox('Период для оценки важности',list(periods),key='importance_period')
    period=report['periods'][periods[selected]]
    st.caption(f"{period['start']} — {period['end']} · {period['rows']} сессий. Здесь используются фиксированные половины тестового года; общий фильтр дат относится к первым двум вкладкам.")
    st.markdown('**Средний абсолютный вклад**')
    st.plotly_chart(importance_chart(period['contribution_importance'],'mean_abs_usd','Средний |вклад|, USD'),width='stretch',key='contribution_importance')
    st.caption('Насколько сильно признак меняет прогноз относительно обучающего среднего. Большой вклад сам по себе не означает улучшения точности.')
    st.markdown('**Изменение ошибки при перестановке**')
    st.plotly_chart(importance_chart(period['permutation_importance'],'delta_mae_usd','Изменение MAE, USD'),width='stretch',key='permutation_importance')
    st.caption('Признак переставлен блоками по 20 сессий, 10 повторов. Положительное значение: ошибка выросла без исходного порядка. Отрицательное: перестановка улучшила результат.')
    st.subheader('Как меняется зависимость от признаков между периодами?')
    first,second=report['periods']['first_half'],report['periods']['second_half']
    a=pd.DataFrame(first['contribution_importance']).set_index('feature')['mean_abs_bps']
    b=pd.DataFrame(second['contribution_importance']).set_index('feature')['mean_abs_bps']
    keys=(a+b).sort_values(ascending=False).head(10).index
    fig=go.Figure()
    fig.add_bar(y=[LABELS[k] for k in keys],x=a.loc[keys],name='Первая половина',orientation='h',marker_color='#2874c6')
    fig.add_bar(y=[LABELS[k] for k in keys],x=b.loc[keys],name='Вторая половина',orientation='h',marker_color='#7dbba8')
    fig.update_layout(barmode='group');fig.update_xaxes(title='Средний |вклад| в изменение цены, базисные пункты')
    st.plotly_chart(style(fig,440),width='stretch',key='period_importance')
    st.caption('1 базисный пункт = 0,01%. Модель и обучающее среднее одинаковы в обеих половинах; меняются входные данные. Относительные единицы уменьшают влияние уровня цены на сравнение.')
    st.dataframe(pd.DataFrame([{'Период':name,'Начало':p['start'],'Конец':p['end'],
                               'MAE базового, USD':p['naive_metrics']['mae_usd'],
                               'MAE Ridge, USD':p['ridge_metrics']['mae_usd']} for name,p in [('Первая половина',first),('Вторая половина',second)]]).round(4),hide_index=True,width='stretch')
    st.subheader('Что произойдёт с прогнозом при изменении одного признака?')
    feature=st.selectbox('Признак для графика зависимости',PRICE_FEATURES+VOLUME_FEATURES,
                          format_func=lambda k:LABELS[k],key='pdp_feature')
    curve=pd.DataFrame(report['pdp'][feature]);factor=1 if feature=='log_volume' else 100
    fig=go.Figure(go.Scatter(x=curve['value']*factor,y=curve['mean_predicted_change_bps'],mode='lines',line=dict(color='#2874c6',width=3)))
    fig.update_xaxes(title=LABELS[feature]+('' if feature=='log_volume' else ', %'))
    fig.update_yaxes(title='Средний прогноз изменения цены, б.п.')
    st.plotly_chart(style(fig,310),width='stretch',key='pdp_chart')
    st.caption('PDP: один признак изменяется в диапазоне 5–95% обучающих значений, остальные фиксируются на 100 обучающих строках. Связанные признаки могут образовать нетипичные сочетания; график не описывает причинный эффект или торговую стратегию.')


def data_tab(config,data):
    st.subheader('Данные и обновляемый расчёт')
    q=data['quality']
    st.write(f"Исследовательский снимок: **{q['price_rows']:,} торговых сессий** и **526 пресс-релизов**. Календарь NASDAQ без пропущенных сессий. Обычная цена закрытия; дивидендная корректировка не используется.")
    st.caption('Историческая таблица и модели выше зафиксированы для честного сравнения. Обновление ниже формирует отдельный расчёт и не меняет результаты теста.')
    if st.button('Обновить источники и расчёт',key='refresh_live',type='primary'):
        from .msft_live import update
        with st.spinner('Загружаются котировки и публикации, проверяются даты и обновляется расчёт…'):
            try:
                update(config,refresh=True)
                st.success('Данные и расчёт обновлены.')
            except Exception as exc:
                st.error(f'Обновление не завершено. Предыдущий расчёт сохранён. {exc}')
    latest=ROOT/'data/processed/msft_live/latest.json'
    if latest.exists():
        live=json.loads(latest.read_text(encoding='utf-8'))
        caption='Предварительный расчёт до отсечки' if live['timing']=='before_cutoff' else 'Расчёт сформирован после номинальной отсечки'
        st.markdown(f"**{caption} · закрытие {live['target_date']}**")
        a,b=st.columns(2)
        a.metric('Основной прогноз · последнее закрытие',f"${live['primary_prediction']:.2f}")
        b.metric('Ridge · обновлённая модель',f"${live['ridge_prediction']:.2f}")
        st.caption(f"Последняя свеча: {live['price_date']}. Обучение Ridge по {live['fit_through']}. Сформировано: {pd.Timestamp(live['created_at']).tz_convert('Europe/Moscow').strftime('%d.%m.%Y %H:%M')} МСК. Данные: {'обновлены из источников' if live['source_status']=='refreshed' else 'из локальной базы без нового запроса'}.")
        st.warning('Время расчёта показано явно: это учебная оценка, а не подтверждение прогноза, выпущенного в 09:00. До отсечки возможны новые публикации; после неё расчёт уже нельзя считать сделанным заранее.')
    else:
        st.info('Обновляемый расчёт ещё не сформирован. Исторический эксперимент доступен в остальных вкладках.')
    st.subheader('Границы выводов')
    st.markdown('Текущие редакции пресс-релизов привязаны к исходным датам публикации. Поздние правки и исправления котировок могут влиять на ретроспективную оценку. Корпус содержит сообщения Microsoft, а не все новости рынка. Результат относится к одному временному разбиению; причинность и статистическая значимость небольших различий не установлены.')
    with st.expander('Периоды и воспроизводимость'):
        st.json(data['metadata']['splits'])
        st.code('Снимок: '+data['metadata']['snapshot_sha256']+'\nПротокол: '+data['protocol']['protocol_sha256'])


def main():
    st.set_page_config(page_title='MSFT · прогноз и факторы',page_icon='📈',layout='wide')
    st.markdown('''<style>
      .block-container {max-width:1440px;padding-top:4.5rem;padding-left:2rem;padding-right:2rem;}
      h1,h2,h3 {letter-spacing:-.025em;} [data-testid="stMetric"] {border:1px solid #9aabb74d;border-radius:12px;padding:12px;}
      .stTabs [data-baseweb="tab"] {font-size:15px;}
    </style>''',unsafe_allow_html=True)
    st.caption('ИНТЕЛЛЕКТУАЛЬНЫЙ АНАЛИЗ ДАННЫХ · MSFT')
    st.title('Прогноз цены и его объяснение')
    st.write('Microsoft · следующее закрытие в USD · котировки и корпоративные публикации')
    try:
        paths=[ROOT/'data/processed/msft/manifest.json',ROOT/'data/processed/msft/features.json',
               ROOT/'reports/msft_protocol.json',ROOT/'reports/msft_interpretability.json',ROOT/'models/msft/ridge__market.joblib']
        config,data=resources(tuple((str(p),p.stat().st_mtime_ns) for p in paths))
    except (FileNotFoundError,ValueError) as exc:
        st.error(f'Исследовательские данные или объяснения недоступны: {exc}')
        st.info('Подготовьте эксперимент и выполните scripts/explain_msft.py. Порядок запуска описан в README.')
        st.stop()
    frame=data['test'].copy()
    frame['naive']=predict_close(None,frame);frame['ridge']=predict_close(data['model'],frame)
    filtered=period_filter(frame)
    st.sidebar.caption('Фильтр действует на прогнозы и разбор выбранной даты. Важность сравнивается на фиксированных половинах тестового года.')
    st.info('На валидации выбран прогноз последнего закрытия; на полном тестовом году он точнее Ridge. Ridge используется для исследования влияния признаков.')
    tabs=st.tabs(['Прогноз и качество','Разбор прогноза','Влияние признаков','Данные и обновление'])
    with tabs[0]: forecast_tab(filtered,data)
    with tabs[1]: local_tab(filtered,data)
    with tabs[2]: factors_tab(data)
    with tabs[3]: data_tab(config,data)


if __name__=='__main__':
    main()
