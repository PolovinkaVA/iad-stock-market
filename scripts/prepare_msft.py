"""Build the frozen MSFT research dataset."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from stocks_forecast.msft_features import main

if __name__ == '__main__':
    raise SystemExit(main())
