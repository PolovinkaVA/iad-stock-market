"""Idempotent ingestion of daily MOEX candles and Bank of Russia RSS news."""

from __future__ import annotations

import argparse
import email.utils
import html
import json
import logging
import re
import sqlite3
import sys
import time
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo
import xml.etree.ElementTree as ET

import requests
import yaml

LOGGER = logging.getLogger("stocks_forecast.ingest")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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


def connect_database(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS prices (
            ticker TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            value REAL,
            trades INTEGER,
            raw_json TEXT NOT NULL,
            fetched_at TEXT NOT NULL,
            PRIMARY KEY (ticker, trade_date)
        );
        CREATE TABLE IF NOT EXISTS news (
            url TEXT PRIMARY KEY,
            seen_at TEXT,
            title TEXT NOT NULL,
            domain TEXT,
            language TEXT,
            source_country TEXT,
            raw_json TEXT NOT NULL,
            fetched_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS ingestion_runs (
            run_id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            status TEXT NOT NULL,
            rows_fetched INTEGER NOT NULL DEFAULT 0,
            rows_written INTEGER NOT NULL DEFAULT 0,
            parameters_json TEXT NOT NULL,
            error TEXT
        );
        """
    )
    columns = {row[1] for row in connection.execute("PRAGMA table_info(ingestion_runs)")}
    if "rows_inserted" in columns and "rows_written" not in columns:
        connection.execute("ALTER TABLE ingestion_runs RENAME COLUMN rows_inserted TO rows_written")
        connection.commit()
    news_columns = {row[1] for row in connection.execute("PRAGMA table_info(news)")}
    if "published_at" in news_columns and "seen_at" not in news_columns:
        connection.execute("ALTER TABLE news RENAME COLUMN published_at TO seen_at")
        connection.commit()
    connection.execute("CREATE INDEX IF NOT EXISTS idx_news_seen_at ON news(seen_at)")
    connection.commit()
    return connection


def begin_run(connection: sqlite3.Connection, source: str, parameters: dict[str, Any]) -> str:
    run_id = str(uuid.uuid4())
    connection.execute(
        "INSERT INTO ingestion_runs(run_id,source,started_at,status,parameters_json) VALUES(?,?,?,?,?)",
        (run_id, source, utc_now(), "running", json.dumps(parameters, ensure_ascii=False, sort_keys=True)),
    )
    connection.commit()
    return run_id


def finish_run(
    connection: sqlite3.Connection,
    run_id: str,
    status: str,
    fetched: int,
    inserted: int,
    error: str | None = None,
) -> None:
    connection.execute(
        "UPDATE ingestion_runs SET finished_at=?,status=?,rows_fetched=?,rows_written=?,error=? WHERE run_id=?",
        (utc_now(), status, fetched, inserted, error, run_id),
    )
    connection.commit()


def get_json(url: str, params: dict[str, Any], timeout: int, retries: int) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.get(
                url,
                params=params,
                timeout=timeout,
                headers={"User-Agent": "sber-semester-forecast/0.1 (educational project)"},
            )
            if response.status_code == 429 or response.status_code >= 500:
                retry_after = response.headers.get("Retry-After")
                try:
                    requested_delay = int(retry_after) if retry_after else 0
                except ValueError:
                    requested_delay = 0
                minimum_delay = 6 if response.status_code == 429 else 2**attempt
                delay = min(max(requested_delay, minimum_delay), 30)
                LOGGER.warning("HTTP %s from %s; retry in %ss", response.status_code, url, delay)
                time.sleep(delay)
                last_error = RuntimeError(f"HTTP {response.status_code}: {response.text[:300]}")
                continue
            response.raise_for_status()
            try:
                return response.json()
            except requests.exceptions.JSONDecodeError as exc:
                raise ValueError(f"Expected JSON from {response.url}; got invalid response") from exc
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                delay = min(2**attempt, 8)
                LOGGER.warning("Request attempt %s/%s failed: %s; retry in %ss", attempt + 1, retries, exc, delay)
                time.sleep(delay)
    raise RuntimeError(f"Request failed after {retries} attempts: {last_error}") from last_error


def parse_moex_candles(payload: dict[str, Any]) -> list[dict[str, Any]]:
    block = payload.get("candles")
    if not isinstance(block, dict):
        raise ValueError("MOEX response has no 'candles' table")
    columns = block.get("columns", [])
    records = block.get("data", [])
    result = []
    for values in records:
        row = dict(zip(columns, values))
        trade_date = str(row.get("begin", ""))[:10]
        close = row.get("close")
        if not trade_date or close is None:
            continue
        result.append(
            {
                "trade_date": trade_date,
                "open": row.get("open"),
                "high": row.get("high"),
                "low": row.get("low"),
                "close": close,
                "volume": row.get("volume"),
                "value": row.get("value"),
                "trades": row.get("trades"),
                "raw_json": json.dumps(row, ensure_ascii=False, separators=(",", ":")),
            }
        )
    return result


def latest_price_date(connection: sqlite3.Connection, ticker: str) -> str | None:
    row = connection.execute("SELECT MAX(trade_date) FROM prices WHERE ticker=?", (ticker,)).fetchone()
    return row[0] if row and row[0] else None


def fetch_moex(config: dict[str, Any], connection: sqlite3.Connection) -> tuple[int, int]:
    source = config["sources"]["moex_iss"]
    instrument = config["instrument"]
    ticker = instrument["ticker"]
    latest = latest_price_date(connection, ticker)
    # Revisit the last few calendar days to include a candle published late or corrected by the exchange.
    start_date = (
        (date.fromisoformat(latest) - timedelta(days=5)).isoformat()
        if latest
        else source["start_date"]
    )
    moscow_today = datetime.now(ZoneInfo(config["project"].get("timezone", "Europe/Moscow"))).date()
    end_date = (moscow_today - timedelta(days=1)).isoformat()  # Exclude today's possibly unfinished candle.
    parameters = {"ticker": ticker, "from": start_date, "till": end_date, "interval": source["candles_interval"]}
    run_id = begin_run(connection, "moex_iss", parameters)
    fetched = inserted = 0
    try:
        if start_date > end_date:
            finish_run(connection, run_id, "success", 0, 0)
            return 0, 0
        base = source["base_url"].rstrip("/")
        endpoint = (
            f"{base}/engines/{instrument['engine']}/markets/{instrument['market']}"
            f"/boards/{instrument['board']}/securities/{ticker}/candles.json"
        )
        offset = 0
        now = utc_now()
        while True:
            params = {
                "from": start_date,
                "till": end_date,
                "interval": source["candles_interval"],
                "start": offset,
                "limit": source["page_size"],
                "iss.meta": "off",
            }
            payload = get_json(endpoint, params, source["timeout_seconds"], source["retries"])
            rows = parse_moex_candles(payload)
            fetched += len(rows)
            for row in rows:
                cursor = connection.execute(
                    """INSERT INTO prices(ticker,trade_date,open,high,low,close,volume,value,trades,raw_json,fetched_at)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?)
                       ON CONFLICT(ticker,trade_date) DO UPDATE SET
                         open=excluded.open,high=excluded.high,low=excluded.low,close=excluded.close,
                         volume=excluded.volume,value=excluded.value,trades=excluded.trades,
                         raw_json=excluded.raw_json,fetched_at=excluded.fetched_at
                       WHERE prices.open IS NOT excluded.open OR prices.high IS NOT excluded.high
                          OR prices.low IS NOT excluded.low OR prices.close IS NOT excluded.close
                          OR prices.volume IS NOT excluded.volume OR prices.value IS NOT excluded.value
                          OR prices.trades IS NOT excluded.trades OR prices.raw_json IS NOT excluded.raw_json""",
                    (ticker, row["trade_date"], row["open"], row["high"], row["low"], row["close"],
                     row["volume"], row["value"], row["trades"], row["raw_json"], now),
                )
                inserted += max(cursor.rowcount, 0)
            connection.commit()
            if len(rows) < source["page_size"]:
                break
            offset += len(rows)
        finish_run(connection, run_id, "success", fetched, inserted)
        return fetched, inserted
    except Exception as exc:
        connection.rollback()
        finish_run(connection, run_id, "failed", fetched, inserted, str(exc))
        raise


def fetch_cbr_rss(config: dict[str, Any], connection: sqlite3.Connection) -> tuple[int, int]:
    """Collect public Bank of Russia press-release headlines from its official RSS 2.0 feed."""
    source = config["sources"]["cbr_rss"]
    parameters = {"url": source["url"], "max_records": source["max_records"]}
    run_id = begin_run(connection, "cbr_rss", parameters)
    fetched = inserted = 0
    try:
        response = None
        last_error: Exception | None = None
        for attempt in range(source["retries"]):
            try:
                response = requests.get(
                    source["url"], timeout=source["timeout_seconds"],
                    headers={"User-Agent": "sber-semester-forecast/0.1 (educational project)", "Accept": "application/rss+xml, application/xml, text/xml"},
                )
                if response.status_code == 429 or response.status_code >= 500:
                    retry_after = response.headers.get("Retry-After")
                    try:
                        requested_delay = int(retry_after) if retry_after else 0
                    except ValueError:
                        requested_delay = 0
                    delay = min(max(requested_delay, 6 if response.status_code == 429 else 2**attempt), 30)
                    LOGGER.warning("HTTP %s from CBR RSS; retry in %ss", response.status_code, delay)
                    time.sleep(delay)
                    last_error = RuntimeError(f"HTTP {response.status_code}: {response.text[:300]}")
                    continue
                response.raise_for_status()
                break
            except requests.RequestException as exc:
                last_error = exc
                if attempt + 1 < source["retries"]:
                    delay = min(2**attempt, 8)
                    LOGGER.warning("CBR RSS request failed: %s; retry in %ss", exc, delay)
                    time.sleep(delay)
        if response is None or response.status_code >= 400:
            raise RuntimeError(f"CBR RSS request failed after {source['retries']} attempts: {last_error}")

        root = ET.fromstring(response.content)
        entries = root.findall(".//item")
        if not entries:
            entries = root.findall(".//{*}entry")
        entries = entries[: int(source["max_records"])]
        fetched = len(entries)
        now = utc_now()

        def child_text(element: ET.Element, name: str) -> str:
            child = element.find(name)
            if child is None:
                child = element.find(f"{{*}}{name}")
            return (child.text or "").strip() if child is not None else ""

        for entry in entries:
            title = child_text(entry, "title")
            link_element = entry.find("link")
            if link_element is None:
                link_element = entry.find("{*}link")
            link = (link_element.get("href", "") if link_element is not None and link_element.get("href") else
                    (link_element.text or "").strip() if link_element is not None else "")
            guid = child_text(entry, "guid") or child_text(entry, "id")
            url = link or guid
            description = child_text(entry, "description") or child_text(entry, "summary") or child_text(entry, "content")
            description = re.sub(r"<[^>]+>", " ", html.unescape(description))
            description = " ".join(description.split())
            text = " ".join(part for part in (title, description) if part)
            date_text = child_text(entry, "pubDate") or child_text(entry, "published") or child_text(entry, "updated")
            if not url or not text:
                continue
            try:
                parsed_date = email.utils.parsedate_to_datetime(date_text)
                if parsed_date.tzinfo is None:
                    parsed_date = parsed_date.replace(tzinfo=timezone.utc)
                seen_at = parsed_date.astimezone(timezone.utc).isoformat(timespec="seconds")
            except (TypeError, ValueError, OverflowError):
                seen_at = None
            record = {"title": text, "link": url, "date": date_text, "description": description}
            cursor = connection.execute(
                """INSERT INTO news(url,seen_at,title,domain,language,source_country,raw_json,fetched_at)
                   VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(url) DO NOTHING""",
                (url, seen_at, text, "cbr.ru", "Russian", "Russia",
                 json.dumps(record, ensure_ascii=False, separators=(",", ":")), now),
            )
            inserted += max(cursor.rowcount, 0)
        connection.commit()
        finish_run(connection, run_id, "success", fetched, inserted)
        return fetched, inserted
    except Exception as exc:
        connection.rollback()
        finish_run(connection, run_id, "failed", fetched, inserted, str(exc))
        raise


def run_ingestion(config_path: Path) -> int:
    config, root = load_config(config_path)
    storage = config["storage"]
    for key in ("raw_dir", "processed_dir", "model_dir", "reports_dir"):
        resolve_path(root, storage[key]).mkdir(parents=True, exist_ok=True)
    connection = connect_database(resolve_path(root, storage["database"]))
    failures = 0
    try:
        for source_name, function in (("MOEX ISS", fetch_moex), ("Bank of Russia RSS", fetch_cbr_rss)):
            try:
                fetched, inserted = function(config, connection)
                LOGGER.info("%s: received %s records; inserted or changed %s", source_name, fetched, inserted)
            except Exception:
                failures += 1
                LOGGER.exception("%s ingestion failed; see ingestion_runs for details", source_name)
    finally:
        connection.close()
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Incrementally ingest MOEX candles and Bank of Russia RSS news")
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        return run_ingestion(args.config)
    except Exception:
        LOGGER.exception("Ingestion could not start")
        return 2


if __name__ == "__main__":
    sys.exit(main())
