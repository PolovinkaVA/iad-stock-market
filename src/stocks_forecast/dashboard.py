"""Local Streamlit dashboard for the SBER forecasting project."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml
import joblib


ROOT = Path(__file__).resolve().parents[2]


def load_config() -> dict[str, Any]:
    with (ROOT / "config" / "config.yaml").open("r", encoding="utf-8-sig") as stream:
        return yaml.safe_load(stream)


CONFIG = load_config()
REPORTS = ROOT / CONFIG["storage"]["reports_dir"]
PROCESSED = ROOT / CONFIG["storage"]["processed_dir"]
DATABASE = ROOT / CONFIG["storage"]["database"]


@st.cache_data
def load_json(path: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


@st.cache_data
def load_latest_market_data(db_path: str, ticker: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    with sqlite3.connect(db_path) as connection:
        prices = pd.read_sql_query(
            "SELECT trade_date, close, volume FROM prices WHERE ticker=? ORDER BY trade_date",
            connection,
            params=(ticker,),
        )
        news = pd.read_sql_query(
            "SELECT seen_at, title, domain FROM news ORDER BY seen_at DESC LIMIT 10",
            connection,
        )
    return prices, news


def rub(value: float) -> str:
    return f"{value:,.2f} ₽".replace(",", " ")


st.set_page_config(page_title="SBER | прогноз цены", page_icon="📈", layout="wide")
st.title("SBER — прогноз цены закрытия")
st.caption("Учебный проект по интеллектуальному анализу данных · горизонт: следующая торговая сессия")

if st.sidebar.button("Обновить данные на странице"):
    st.cache_data.clear()

comparison_path = REPORTS / "model_comparison.json"
predictions_path = REPORTS / "test_predictions.csv"
monitoring_path = REPORTS / "monitoring.json"
feature_analysis_path = REPORTS / "feature_analysis.json"
quality_path = PROCESSED / "quality_report.json"
latest_features_path = PROCESSED / "latest_features.json"
required = [comparison_path, predictions_path, monitoring_path, quality_path, DATABASE]
missing = [str(path) for path in required if not path.exists()]
if missing:
    st.warning("Сначала выполните загрузку и расчёт модели. Не найдены: " + ", ".join(missing))
    st.code("$env:PYTHONPATH = 'src'\npython -m stocks_forecast.ingest\npython -m stocks_forecast.prepare\npython -m stocks_forecast.modeling\nstreamlit run src/stocks_forecast/dashboard.py", language="powershell")
    st.stop()

comparison = load_json(str(comparison_path))
monitoring = load_json(str(monitoring_path))
quality = load_json(str(quality_path))
feature_analysis = load_json(str(feature_analysis_path)) if feature_analysis_path.exists() else None
predictions = load_csv(str(predictions_path))
prices, news = load_latest_market_data(str(DATABASE), CONFIG["instrument"]["ticker"])
latest_features = load_json(str(latest_features_path)) if latest_features_path.exists() else {}

latest_price = float(prices.iloc[-1]["close"])
latest_date = str(prices.iloc[-1]["trade_date"])
model_name = comparison["selected_model"]
if model_name == "naive_last_close":
    next_estimate = latest_price
    forecast_description = "Выбранная модель переносит последнее закрытие на следующую сессию."
else:
    model_file = comparison.get("deployment_model_file")
    if model_file and latest_features:
        artifact = joblib.load(ROOT / CONFIG["storage"]["model_dir"] / model_file)
        latest_vector = [[float(latest_features[name]) for name in artifact["features"]]]
        next_estimate = float(artifact["model"].predict(latest_vector)[0])
        forecast_description = f"Прогноз выбранной модели {model_name} по признакам на {latest_date}."
    else:
        next_estimate = latest_price
        forecast_description = "Файл модели или текущие признаки не найдены; показан наивный ориентир."

metric_a, metric_b, metric_c, metric_d = st.columns(4)
metric_a.metric("Последнее закрытие", rub(latest_price), help=f"По данным MOEX за {latest_date}")
metric_b.metric("Ориентир на следующую сессию", rub(next_estimate), help=forecast_description)
metric_c.metric("MAE на тесте", rub(float(comparison["untouched_test_metrics"]["mae_rub"])))
metric_d.metric("MAPE на тесте", f"{comparison['untouched_test_metrics']['mape_percent']:.2f}%")
st.caption(f"Модель: `{model_name}` · последняя доступная котировка: {latest_date}. Значение не является инвестиционной рекомендацией.")

left, right = st.columns([1.55, 1])
with left:
    st.subheader("Котировка SBER")
    history_days = st.selectbox("Показать историю", [90, 252, 756, 2200], index=1, format_func=lambda n: f"Последние {n} сессий")
    shown = prices.tail(history_days)
    chart = go.Figure()
    chart.add_trace(go.Scatter(x=shown["trade_date"], y=shown["close"], mode="lines", name="Цена закрытия", line={"color": "#1769aa", "width": 2}))
    chart.update_layout(height=390, margin={"l": 12, "r": 12, "t": 18, "b": 12}, yaxis_title="Рублей", xaxis_title="", hovermode="x unified", legend={"orientation": "h", "y": 1.08})
    st.plotly_chart(chart, use_container_width=True)

with right:
    st.subheader("Качество и покрытие данных")
    coverage = comparison.get("news_coverage", {})
    st.write(f"**Модель:** `{model_name}`")
    st.write(f"**Объяснение:** {comparison.get('interpretability', {}).get('selected_model_explanation', 'Нет описания.')}")
    st.write(f"**Новости:** {coverage.get('sessions_with_news', 0)} сессий с новостями из {coverage.get('feature_rows', 0)} ({coverage.get('coverage_percent', 0):.2f}%).")
    st.write(f"**Порог включения новостей:** {coverage.get('minimum_to_include_percent', 20):.0f}% покрытия; в обучении {'использованы' if coverage.get('included_in_model') else 'не использованы'}.")
    st.write(f"**Контроль качества:** {quality.get('prices_clean_rows', 0)} нормализованных сессий; объединено выходных свечей: {quality.get('prices_weekend_candles_merged_into_next_weekday', 0)}.")

tab_backtest, tab_errors, tab_news = st.tabs(["Тестовый прогноз", "Мониторинг ошибки", "Новости RSS"])
with tab_backtest:
    st.subheader("Прогноз и факт на отложенном тесте")
    test_chart = go.Figure()
    test_chart.add_trace(go.Scatter(x=predictions["target_date"], y=predictions["actual_close"], mode="lines", name="Факт", line={"color": "#1769aa"}))
    test_chart.add_trace(go.Scatter(x=predictions["target_date"], y=predictions["predicted_close"], mode="lines", name="Прогноз", line={"color": "#e67e22"}))
    test_chart.update_layout(height=430, yaxis_title="Цена закрытия, ₽", xaxis_title="", hovermode="x unified", legend={"orientation": "h", "y": 1.08})
    st.plotly_chart(test_chart, use_container_width=True)
    st.caption(f"Период теста: {comparison['test_period']['start']} — {comparison['test_period']['end']}; он не использовался для выбора модели.")
    if model_name == "naive_last_close" and not predictions.empty:
        selected_target = st.selectbox("Объяснить прогноз на дату", predictions["target_date"].tolist(), key="explain_forecast_date")
        selected_prediction = predictions.loc[predictions["target_date"] == selected_target].iloc[0]
        st.info(
            f"Для SBER на {selected_prediction['forecast_date']} прогноз на {selected_target} равен последнему известному закрытию: "
            f"{rub(float(selected_prediction['predicted_close']))}. В наивной модели используется только признак `close_at_forecast`."
        )
    elif comparison.get("deployment_model_file") and latest_features:
        artifact = joblib.load(ROOT / CONFIG["storage"]["model_dir"] / comparison["deployment_model_file"])
        training_features = load_csv(str(PROCESSED / "features.csv"))
        training_features = training_features[training_features["trade_date"] <= comparison["train_period"]["end"]]
        medians = training_features[artifact["features"]].median()
        reference_vector = [[float(medians[name]) for name in artifact["features"]]]
        current_vector = [[float(latest_features[name]) for name in artifact["features"]]]
        reference_prediction = float(artifact["model"].predict(reference_vector)[0])
        st.write(f"Прогноз при медианных значениях признаков train — {rub(reference_prediction)}; текущий прогноз — {rub(next_estimate)}.")
        effects = []
        for index, name in enumerate(artifact["features"]):
            perturbed = [row.copy() for row in reference_vector]
            perturbed[0][index] = current_vector[0][index]
            value = float(artifact["model"].predict(perturbed)[0])
            effects.append({"Признак": name, "Текущее значение": current_vector[0][index], "Отклонение от прогноза на медианах, ₽": value - reference_prediction})
        st.caption("Локальная диагностика: по очереди подставляем текущее значение каждого признака вместо его медианы обучающего набора. Эффекты по одному признаку не обязаны суммироваться при нелинейной модели.")
        st.dataframe(pd.DataFrame(effects).sort_values("Отклонение от прогноза на медианах, ₽", key=lambda values: values.abs(), ascending=False), hide_index=True, use_container_width=True)

with tab_errors:
    st.subheader("Ошибка во времени")
    error_chart = go.Figure()
    error_chart.add_trace(go.Scatter(x=predictions["target_date"], y=predictions["rolling_20_session_mae_rub"], mode="lines", name="MAE, окно 20 сессий", line={"color": "#8e44ad"}))
    error_chart.add_hline(y=monitoring["reference_validation_mae_rub"], line_dash="dash", line_color="#555", annotation_text="MAE на валидации")
    error_chart.update_layout(height=350, yaxis_title="MAE, ₽", xaxis_title="", hovermode="x unified")
    st.plotly_chart(error_chart, use_container_width=True)
    st.write(f"Статус: **{monitoring['status']}** · MAE последних {monitoring['recent_window_sessions']} сессий — {rub(monitoring['recent_window_mae_rub'])}; контрольное отношение — {monitoring['recent_to_validation_mae_ratio']:.2f} при пороге {monitoring['warning_threshold_ratio']:.1f}.")
    monthly = pd.DataFrame([{"Месяц": month, "Сессий": values["rows"], "MAE, ₽": values["mae_rub"]} for month, values in monitoring["monthly_mae"].items()])
    st.dataframe(monthly, hide_index=True, use_container_width=True)
    st.caption(monitoring["interpretation"])

with tab_news:
    st.subheader("Последние публикации Банка России")
    if news.empty:
        st.info("В базе пока нет новостей. Обновите данные загрузчиком.")
    else:
        for item in news.itertuples(index=False):
            title = " ".join(str(item.title).split())
            if len(title) > 450:
                title = title[:447] + "..."
            st.markdown(f"**{item.seen_at or 'Дата не указана'}** · {item.domain or 'источник неизвестен'}  \n{title}")
    st.caption("Официальная RSS-лента не покрывает всю историю котировок; новостные признаки не вошли в обучение при покрытии ниже заданного порога.")

if feature_analysis is not None:
    st.subheader("Влияние групп признаков")
    st.caption("Сравнение по MAE на валидации; тест не использовался для выбора признаков или модели.")
    rows = []
    for model, variants in feature_analysis["validation_ablation"].items():
        for variant, metrics in variants.items():
            rows.append({"Модель": model, "Набор признаков": variant, "MAE, ₽": metrics["mae_rub"], "RMSE, ₽": metrics["rmse_rub"]})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    importance_model = st.selectbox("Модель для анализа важности", list(feature_analysis["tree_model_importance"]), key="importance_model")
    importance_rows = feature_analysis["tree_model_importance"][importance_model]["permutation_importance"]["validation"][:10]
    importance_frame = pd.DataFrame(importance_rows).rename(columns={"feature": "Признак", "mae_increase_rub": "Рост MAE при перемешивании, ₽", "std_rub": "Разброс, ₽"})
    st.bar_chart(importance_frame.set_index("Признак")["Рост MAE при перемешивании, ₽"])
    st.dataframe(importance_frame, hide_index=True, use_container_width=True)
    st.caption("Положительное значение означает, что перемешивание признака ухудшило MAE. Важность — диагностическая оценка, не причинный эффект.")
