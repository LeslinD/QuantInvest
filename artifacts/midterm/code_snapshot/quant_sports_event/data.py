from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List
from urllib.error import URLError
from urllib.request import Request, urlopen

import pandas as pd


class DataProviderError(RuntimeError):
    pass


class SinaDailyProvider:
    """Small dependency-free daily K-line provider.

    Sina returns current point-in-time daily bars for A-share symbols such as
    ``sh600060`` and index symbols such as ``sh000300``. We use it as a fast
    public data source and freeze the raw response for audit.
    """

    endpoint = "https://quotes.sina.cn/cn/api/json_v2.php/CN_MarketDataService.getKLineData"

    def __init__(self, datalen: int = 1500, timeout: int = 20, sleep_seconds: float = 0.03):
        self.datalen = datalen
        self.timeout = timeout
        self.sleep_seconds = sleep_seconds

    def fetch_daily(self, sina_symbol: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        url = (
            f"{self.endpoint}?symbol={sina_symbol}&scale=240&ma=no&datalen={self.datalen}"
        )
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        last_error: Exception | None = None
        for _ in range(3):
            try:
                text = urlopen(req, timeout=self.timeout).read().decode("utf-8")
                data = json.loads(text)
                if not data:
                    raise DataProviderError(f"No data returned for {sina_symbol}")
                df = pd.DataFrame(data)
                break
            except (URLError, TimeoutError, json.JSONDecodeError, DataProviderError) as exc:
                last_error = exc
                time.sleep(0.4)
        else:
            raise DataProviderError(f"Failed to fetch {sina_symbol}: {last_error}")

        df = df.rename(columns={"day": "date"})
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["date"] = pd.to_datetime(df["date"])
        df["sina_symbol"] = sina_symbol
        df["amount"] = df["close"] * df["volume"]
        df = df.dropna(subset=["date", "open", "high", "low", "close"]).sort_values("date")
        if start:
            df = df[df["date"] >= pd.to_datetime(start)]
        if end:
            df = df[df["date"] <= pd.to_datetime(end)]
        return df.reset_index(drop=True)

    def fetch_many(self, symbols: Iterable[str], start: str | None = None, end: str | None = None) -> Dict[str, pd.DataFrame]:
        out: Dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            out[symbol] = self.fetch_daily(symbol, start=start, end=end)
            time.sleep(self.sleep_seconds)
        return out


def freeze_market_snapshot(
    provider: SinaDailyProvider,
    symbols: List[str],
    as_of_date: str,
    snapshot_dir: str | Path,
    benchmark_symbols: List[str] | None = None,
) -> Dict[str, str]:
    snapshot_dir = Path(snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    all_symbols = list(dict.fromkeys(symbols + (benchmark_symbols or [])))
    frames = provider.fetch_many(all_symbols, end=as_of_date)
    paths: Dict[str, str] = {}
    for symbol, df in frames.items():
        if df.empty:
            raise DataProviderError(f"{symbol} has no data through {as_of_date}")
        max_date = df["date"].max().date().isoformat()
        if max_date > as_of_date:
            raise DataProviderError(f"{symbol} returned future data {max_date} > {as_of_date}")
        path = snapshot_dir / f"{symbol}_{as_of_date}.csv"
        df.to_csv(path, index=False)
        paths[symbol] = str(path)
    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "as_of_date": as_of_date,
        "provider": "sina",
        "symbols": all_symbols,
        "paths": paths,
    }
    meta_path = snapshot_dir / f"snapshot_meta_{as_of_date}.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    paths["_meta"] = str(meta_path)
    return paths


def load_snapshot(paths: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    frames: Dict[str, pd.DataFrame] = {}
    for symbol, path in paths.items():
        if symbol.startswith("_"):
            continue
        df = pd.read_csv(path, parse_dates=["date"])
        frames[symbol] = df
    return frames
