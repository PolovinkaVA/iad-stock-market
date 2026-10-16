"""CLI entry point; paths in config are relative to the project root."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from stocks_forecast.msft_ingest import main

if __name__ == '__main__':
    raise SystemExit(main())
