"""Quality checks and point-in-time feature generation for SBER forecasts."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import logging
import math
import re
import sqlite3
import statistics
import sys
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml

LOGGER = logging.getLogger("stocks_forecast.prepare")
TOKEN_RE = re.compile(r"[a-zа-яё0-9]+", flags=re.IGNORECASE)
POSITIVE_WORDS = {"рост", "растет", "прибыль", "рекорд", "роста", "выручка", "growth", "profit", "record"}
NEGATIVE_WORDS = {"падение", "снижение", "убыток", "убытки", "риск", "санкции", "падения", "loss", "risk"}


def load_config(path: Path) -> tuple[dict[str, Any], Path]:
    path = path.resolve()
    with path.open("r", encoding="utf-8-sig") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {path}")
    return config, path.parent.parent


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def parse_seen_at(value: str | None, local_timezone: ZoneInfo) -> datetime | None:
    if not value:
        return None
    try:
        # GDELT seendate format: YYYYMMDDTHHMMSSZ.
        parsed = datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return parsed.astimezone(local_timezone)


def clean_prices(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows.sort(key=lambda row: row["trade_date"])
    seen: set[str] = set()
    valid_rows: list[dict[str, Any]] = []
    duplicate_count = 0
    invalid_count = 0
    missing_volume_count = 0

    for row in rows:
        trade_date = str(row["trade_date"])[:10]
        if trade_date in seen:
            duplicate_count += 1
            continue
        seen.add(trade_date)
        numeric: dict[str, float | None] = {}
        for field in ("open", "high", "low", "close", "volume"):
            try:
                numeric[field] = float(row[field]) if row[field] is not None else None
            except (TypeError, ValueError):
                numeric[field] = None
        opening, high, low, close = (numeric[key] for key in ("open", "high", "low", "close"))
        volume = numeric["volume"]
        if (
            opening is None or high is None or low is None or close is None
            or close <= 0 or opening <= 0 or low <= 0 or high <= 0
            or high < low or high < max(opening, close) or low > min(opening, close)
            or (volume is not None and volume < 0)
        ):
            invalid_count += 1
            continue
        if volume is None:
            missing_volume_count += 1
            volume = 0.0
        normalized = {"trade_date": trade_date, **numeric, "volume": volume}
        valid_rows.append(normalized)

    # MOEX weekend auction candles are part of the following weekday trading session.
    clean: list[dict[str, Any]] = []
    weekend_pending: list[dict[str, Any]] = []
    weekend_merged = 0
    for row in valid_rows:
        weekday = date.fromisoformat(row["trade_date"]).weekday()
        if weekday >= 5:
            weekend_pending.append(row)
            continue
        parts = weekend_pending + [row]
        if weekend_pending:
            weekend_merged += len(weekend_pending)
        clean.append(
            {
                **row,
                "open": parts[0]["open"],
                "high": max(part["high"] for part in parts),
                "low": min(part["low"] for part in parts),
                "close": row["close"],
                "volume": sum(float(part["volume"]) for part in parts),
                "raw_rows_merged": len(parts),
            }
        )
        weekend_pending = []

    suspicious_moves: list[dict[str, Any]] = []
    for previous, current in zip(clean, clean[1:]):
        move = float(current["close"]) / float(previous["close"]) - 1.0
        if abs(move) >= 0.20:
            suspicious_moves.append({"trade_date": current["trade_date"], "return": round(move, 6)})

    report = {
        "prices_input_rows": len(rows),
        "prices_duplicate_dates_dropped": duplicate_count,
        "prices_invalid_ohlcv_rows_dropped": invalid_count,
        "prices_missing_volume_filled_zero": missing_volume_count,
        "prices_weekend_candles_merged_into_next_weekday": weekend_merged,
        "prices_trailing_weekend_candles_without_next_weekday": len(weekend_pending),
        "prices_suspicious_daily_moves_flagged_not_removed": suspicious_moves,
        "prices_clean_rows": len(clean),
        "prices_first_date": clean[0]["trade_date"] if clean else None,
        "prices_last_date": clean[-1]["trade_date"] if clean else None,
    }
    return clean, report


def aggregate_news(
    news: list[dict[str, Any]],
    prices: list[dict[str, Any]],
    local_timezone: ZoneInfo,
    close_time: time,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cutoffs: list[datetime] = []
    session_dates: list[date] = []
    for row in prices:
        session_date = date.fromisoformat(row["trade_date"])
        session_dates.append(session_date)
        cutoffs.append(datetime.combine(session_date, close_time, tzinfo=local_timezone))

    per_session: list[list[dict[str, Any]]] = [[] for _ in prices]
    invalid_timestamps = 0
    outside_session_windows = 0
    for article in news:
        seen_at = parse_seen_at(article.get("seen_at"), local_timezone)
        if seen_at is None:
            invalid_timestamps += 1
            continue
        idx = bisect.bisect_left(cutoffs, seen_at)
        if idx >= len(cutoffs):
            outside_session_windows += 1
            continue
        lower_bound = (
            datetime.combine(session_dates[0], time.min, tzinfo=local_timezone)
            if idx == 0 else cutoffs[idx - 1]
        )
        # A post-close story is assigned to the next session; one known at today's close belongs to today.
        if seen_at <= lower_bound:
            outside_session_windows += 1
            continue
        per_session[idx].append(article)

    daily: list[dict[str, Any]] = []
    for articles in per_session:
        tokens_by_article = [TOKEN_RE.findall(article.get("title", "").lower()) for article in articles]
        domains = {article.get("domain") for article in articles if article.get("domain")}
        languages = [str(article.get("language") or "").lower() for article in articles]
        all_tokens = [token for group in tokens_by_article for token in group]
        daily.append(
            {
                "news_count": len(articles),
                "news_distinct_domains": len(domains),
                "news_title_chars_mean": round(statistics.mean([len(a.get("title", "")) for a in articles]), 4) if articles else 0.0,
                "news_title_words_mean": round(statistics.mean([len(tokens) for tokens in tokens_by_article]), 4) if articles else 0.0,
                "news_russian_count": sum("russian" in language for language in languages),
                "news_english_count": sum("english" in language for language in languages),
                "news_positive_terms": sum(token in POSITIVE_WORDS for token in all_tokens),
                "news_negative_terms": sum(token in NEGATIVE_WORDS for token in all_tokens),
            }
        )
    return daily, {
        "news_rows_in_database": len(news),
        "news_rows_with_invalid_timestamps": invalid_timestamps,
        "news_rows_outside_price_session_windows": outside_session_windows,
        "news_rows_assigned_to_sessions": sum(len(group) for group in per_session),
        "news_data_loaded": bool(news),
        "news_session_close_local": close_time.isoformat(),
    }


def rolling_mean(values: list[float], end: int, window: int) -> float | None:
    if end + 1 < window:
        return None
    return statistics.mean(values[end - window + 1 : end + 1])


def rolling_std(values: list[float], end: int, window: int) -> float | None:
    if end + 1 < window:
        return None
    sample = values[end - window + 1 : end + 1]
    return statistics.stdev(sample) if len(sample) >= 2 else 0.0


def build_features(
    prices: list[dict[str, Any]],
    news: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    clean, qa_report = clean_prices(prices)
    if len(clean) < 21:
        raise ValueError(f"Need at least 21 valid trading days; got {len(clean)}")
    local_timezone = ZoneInfo(config["project"].get("timezone", "Europe/Moscow"))
    close_time = time.fromisoformat(config["instrument"].get("session_close_local", "18:50"))
    daily_news, news_report = aggregate_news(news, clean, local_timezone, close_time)

    closes = [float(row["close"]) for row in clean]
    volumes = [float(row["volume"]) for row in clean]
    returns = [0.0] + [closes[i] / closes[i - 1] - 1.0 for i in range(1, len(closes))]
    output: list[dict[str, Any]] = []
    feature_missing_rows = 0

    for i, row in enumerate(clean[:-1]):
        trade_day = date.fromisoformat(row["trade_date"])
        target = clean[i + 1]
        feature: dict[str, Any] = {
            "trade_date": row["trade_date"],
            "target_date": target["trade_date"],
            "close_at_forecast": closes[i],
            "volume_at_forecast": volumes[i],
            "return_1d": returns[i],
            "return_lag_1d": returns[i - 1] if i >= 1 else None,
            "return_lag_2d": returns[i - 2] if i >= 2 else None,
            "return_lag_5d": returns[i - 5] if i >= 5 else None,
            "close_lag_1d": closes[i - 1] if i >= 1 else None,
            "close_lag_5d": closes[i - 5] if i >= 5 else None,
            "close_mean_5d": rolling_mean(closes, i, 5),
            "close_mean_20d": rolling_mean(closes, i, 20),
            "return_std_5d": rolling_std(returns, i, 5),
            "volume_mean_5d": rolling_mean(volumes, i, 5),
            # Cyclical encoding preserves the calendar wrap-around
            # (Sunday is adjacent to Monday; December to January).
            "weekday_sin": round(math.sin(2 * math.pi * trade_day.weekday() / 7), 8),
            "weekday_cos": round(math.cos(2 * math.pi * trade_day.weekday() / 7), 8),
            "month_sin": round(math.sin(2 * math.pi * (trade_day.month - 1) / 12), 8),
            "month_cos": round(math.cos(2 * math.pi * (trade_day.month - 1) / 12), 8),
            "day_of_month_sin": round(math.sin(2 * math.pi * (trade_day.day - 1) / 31), 8),
            "day_of_month_cos": round(math.cos(2 * math.pi * (trade_day.day - 1) / 31), 8),
            "news_source_loaded": int(news_report["news_data_loaded"]),
            **daily_news[i],
            "target_close_next": closes[i + 1],
        }
        if any(value is None for key, value in feature.items() if key not in {"target_date"}):
            feature_missing_rows += 1
            continue
        output.append(feature)

    # Current inference row has no next-session label yet, so it belongs in a
    # separate file and is never added to the training table.
    i = len(clean) - 1
    row = clean[i]
    trade_day = date.fromisoformat(row["trade_date"])
    latest_feature = {
        "trade_date": row["trade_date"],
        "close_at_forecast": closes[i],
        "volume_at_forecast": volumes[i],
        "return_1d": returns[i],
        "return_lag_1d": returns[i - 1] if i >= 1 else 0.0,
        "return_lag_2d": returns[i - 2] if i >= 2 else 0.0,
        "return_lag_5d": returns[i - 5] if i >= 5 else 0.0,
        "close_lag_1d": closes[i - 1] if i >= 1 else closes[i],
        "close_lag_5d": closes[i - 5] if i >= 5 else closes[i],
        "close_mean_5d": rolling_mean(closes, i, 5),
        "close_mean_20d": rolling_mean(closes, i, 20),
        "return_std_5d": rolling_std(returns, i, 5),
        "volume_mean_5d": rolling_mean(volumes, i, 5),
        "weekday_sin": round(math.sin(2 * math.pi * trade_day.weekday() / 7), 8),
        "weekday_cos": round(math.cos(2 * math.pi * trade_day.weekday() / 7), 8),
        "month_sin": round(math.sin(2 * math.pi * (trade_day.month - 1) / 12), 8),
        "month_cos": round(math.cos(2 * math.pi * (trade_day.month - 1) / 12), 8),
        "day_of_month_sin": round(math.sin(2 * math.pi * (trade_day.day - 1) / 31), 8),
        "day_of_month_cos": round(math.cos(2 * math.pi * (trade_day.day - 1) / 31), 8),
        "news_source_loaded": int(news_report["news_data_loaded"]),
        **daily_news[i],
    }

    report = {
        **qa_report,
        **news_report,
        "feature_rows_written": len(output),
        "feature_rows_dropped_for_history": feature_missing_rows,
        "latest_session_dropped_without_next_target": 1,
        "feature_rows_dropped_total": feature_missing_rows + 1,
        "target": "next trading session close (RUB)",
        "feature_cutoff": f"{close_time.isoformat()} Europe local time",
        "leakage_note": "Rows are ordered by trading session; label is the following session close. News seen after a close is assigned to a later session.",
        "news_text_method": "Headline token counts from a small transparent RU/EN lexicon plus title length, word count, domain and language aggregation; not a full semantic sentiment model.",
        "_latest_inference_features": latest_feature,
    }
    return output, report


def run_prepare(config_path: Path) -> dict[str, Any]:
    config, root = load_config(config_path)
    database = resolve_path(root, config["storage"]["database"])
    processed_dir = resolve_path(root, config["storage"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    try:
        ticker = config["instrument"]["ticker"]
        price_rows = [dict(row) for row in connection.execute(
            "SELECT trade_date,open,high,low,close,volume FROM prices WHERE ticker=? ORDER BY trade_date",
            (ticker,),
        )]
        news_rows = [dict(row) for row in connection.execute(
            "SELECT seen_at,title,domain,language FROM news ORDER BY seen_at"
        )]
    finally:
        connection.close()
    if not price_rows:
        raise ValueError(f"No price history found for {ticker}. Run the ingestion command first.")

    features, report = build_features(price_rows, news_rows, config)
    latest_feature_path = processed_dir / "latest_features.json"
    latest_features = report.pop("_latest_inference_features")
    with latest_feature_path.open("w", encoding="utf-8") as stream:
        json.dump(latest_features, stream, ensure_ascii=False, indent=2)
    features_path = processed_dir / "features.csv"
    report_path = processed_dir / "quality_report.json"
    fields = list(features[0].keys())
    with features_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(features)
    report["features_file"] = str(features_path.relative_to(root))
    with report_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    LOGGER.info("Prepared %s rows -> %s", len(features), features_path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate market data and build point-in-time features")
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        report = run_prepare(args.config)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except Exception:
        LOGGER.exception("Feature preparation failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
