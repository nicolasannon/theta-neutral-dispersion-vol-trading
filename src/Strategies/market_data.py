from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Protocol

import pandas as pd

from .instruments import OptionContract, OptionQuote, OptionType


class SpotProvider(Protocol):
    def get_spot(self, symbol: str) -> float:
        ...


class RateProvider(Protocol):
    def get_annual_rate(self, asof: date) -> float:
        """
        Annual simple rate as decimal (e.g. 0.03).
        """
        ...


class OptionMarketDataProvider(SpotProvider, Protocol):
    def get_dividend_yield(self, symbol: str) -> float:
        ...

    def list_expirations(self, symbol: str) -> list[date]:
        ...

    def get_option_chain(self, contract: OptionContract) -> pd.DataFrame:
        ...

    def get_option_quote(self, contract: OptionContract) -> OptionQuote:
        ...

    def get_option_mark(self, contract: OptionContract) -> float | None:
        ...


@dataclass
class LocalParquetMarketData(OptionMarketDataProvider):
    """
    Local historical market data provider based on:
    - aapl_2016_2023.parquet
    - spy_2020_2022.parquet

    It exposes a Yahoo-like interface to minimize downstream changes.
    """
    data_root: Path | str
    asof: date
    default_dividend_yield: float = 0.0
    use_mid_as_last_price: bool = True

    _df_all: pd.DataFrame = field(init=False, repr=False)

    def __post_init__(self) -> None:
        root = Path(self.data_root)

        aapl_path = root / "aapl_2016_2023.parquet"
        spy_path = root / "spy_2020_2022.parquet"

        if not aapl_path.exists():
            raise ValueError(f"Missing file: {aapl_path}")
        if not spy_path.exists():
            raise ValueError(f"Missing file: {spy_path}")

        aapl = pd.read_parquet(aapl_path)
        spy = pd.read_parquet(spy_path)

        aapl = self._normalize_option_frame(aapl, forced_ticker=None)
        spy = self._normalize_option_frame(spy, forced_ticker="SPY")

        self._df_all = pd.concat([aapl, spy], ignore_index=True)

    def set_asof(self, asof: date) -> None:
        self.asof = asof

    def list_common_dates(self, symbols: list[str]) -> list[date]:
        if not symbols:
            return []

        common_dates: set[date] | None = None
        for symbol in symbols:
            dates = set(self._df_all.loc[self._df_all["ticker"] == symbol.upper(), "date"].unique().tolist())
            common_dates = dates if common_dates is None else common_dates.intersection(dates)

        if common_dates is None:
            return []
        return sorted(common_dates)

    @staticmethod
    def _to_cp(option_type: OptionType) -> str:
        return "C" if option_type == OptionType.CALL else "P"

    @staticmethod
    def _normalize_option_frame(df: pd.DataFrame, forced_ticker: str | None) -> pd.DataFrame:
        required = {
            "date",
            "spot",
            "strike",
            "expiration",
            "implied_volatility",
            "bid",
            "ask",
            "mid",
            "volume",
            "call_put",
            "option_id",
        }

        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"Missing required columns in parquet: {sorted(missing)}")

        out = df.copy()
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
        out["expiration"] = pd.to_datetime(out["expiration"], errors="coerce").dt.date
        out["call_put"] = out["call_put"].astype(str).str.strip().str.upper()
        out["option_id"] = out["option_id"].astype(str).str.strip()

        if forced_ticker is not None:
            out["ticker"] = forced_ticker
        elif "ticker" in out.columns:
            out["ticker"] = out["ticker"].astype(str).str.strip().str.upper()
        else:
            raise ValueError("Ticker column missing and no forced ticker provided")

        for col in ["spot", "strike", "implied_volatility", "bid", "ask", "mid", "volume"]:
            out[col] = pd.to_numeric(out[col], errors="coerce")

        out = out.dropna(subset=["date", "expiration", "ticker", "call_put", "strike", "spot"])
        return out

    def _slice_day_symbol(self, symbol: str) -> pd.DataFrame:
        sym = symbol.strip().upper()
        out = self._df_all[(self._df_all["ticker"] == sym) & (self._df_all["date"] == self.asof)]
        if out.empty:
            raise ValueError(f"No local data for symbol={sym} at asof={self.asof.isoformat()}")
        return out

    def get_spot(self, symbol: str) -> float:
        df = self._slice_day_symbol(symbol)
        return float(df["spot"].median())

    def get_dividend_yield(self, symbol: str) -> float:
        return float(self.default_dividend_yield)

    def list_expirations(self, symbol: str) -> list[date]:
        df = self._slice_day_symbol(symbol)
        expiries = sorted(df["expiration"].dropna().unique().tolist())
        return [d for d in expiries]

    def get_option_chain(self, contract: OptionContract) -> pd.DataFrame:
        df = self._slice_day_symbol(contract.underlying.symbol)
        call_put = self._to_cp(contract.option_type)

        out = df[(df["expiration"] == contract.expiry) & (df["call_put"] == call_put)].copy()
        if out.empty:
            raise ValueError(
                f"No option chain for {contract.underlying.symbol} "
                f"expiry={contract.expiry.isoformat()} cp={call_put} asof={self.asof.isoformat()}"
            )

        out["strike"] = out["strike"].astype(float)
        out["impliedVolatility"] = out["implied_volatility"].astype(float)
        out["lastPrice"] = out["mid"].astype(float) if self.use_mid_as_last_price else out["ask"].astype(float)
        out["lastTradeDate"] = pd.NaT
        return out.sort_values("strike")

    def get_option_quote(self, contract: OptionContract) -> OptionQuote:
        chain = self.get_option_chain(contract)
        target_strike = float(contract.strike)

        row = chain.loc[chain["strike"] == target_strike]
        if row.empty:
            row = chain.iloc[(chain["strike"] - target_strike).abs().argsort()[:1]]

        first_row = row.iloc[0].to_dict()

        def _f(value: object) -> float | None:
            try:
                if value is None or pd.isna(value):
                    return None
                return float(value)
            except Exception:
                return None

        return OptionQuote(
            last=_f(first_row.get("lastPrice")),
            bid=_f(first_row.get("bid")),
            ask=_f(first_row.get("ask")),
            implied_vol=_f(first_row.get("impliedVolatility")),
            timestamp_utc=self.asof.isoformat(),
        )

    def get_option_mark(self, contract: OptionContract) -> float | None:
        quote = self.get_option_quote(contract)
        return quote.best_price()


@dataclass
class CsvYieldCurveRateProvider(RateProvider):
    """
    Rate provider from par-yield-curve-rates-2020-2023.csv

    Returns annual simple rate as decimal.
    Example:
    - file value 1.56 means 1.56%
    - returned value is 0.0156
    """
    csv_path: Path | str
    tenor_col: str = "1 Yr"

    _df: pd.DataFrame = field(init=False, repr=False)

    def __post_init__(self) -> None:
        path = Path(self.csv_path)
        if not path.exists():
            raise ValueError(f"Missing rates file: {path}")

        df = pd.read_csv(path)

        if "date" not in df.columns:
            raise ValueError("Rates CSV must contain a 'date' column")
        if self.tenor_col not in df.columns:
            raise ValueError(f"Rates CSV missing tenor column '{self.tenor_col}'")

        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
        df[self.tenor_col] = pd.to_numeric(df[self.tenor_col], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")

        if df.empty:
            raise ValueError("Rates CSV has no valid rows after date parsing")

        self._df = df

    def get_annual_rate(self, asof: date) -> float:
        hist = self._df[self._df["date"] <= asof]
        if hist.empty:
            raise ValueError(f"No available rate on or before asof={asof.isoformat()}")

        row = hist.iloc[-1]
        raw = row[self.tenor_col]

        if pd.isna(raw):
            hist2 = hist.dropna(subset=[self.tenor_col])
            if hist2.empty:
                raise ValueError(f"No valid '{self.tenor_col}' value on or before asof={asof.isoformat()}")
            raw = hist2.iloc[-1][self.tenor_col]

        return float(raw) / 100.0


@dataclass
class ConstantRateProvider(RateProvider):
    annual_rate_simple: float = 0.02

    def get_annual_rate(self, asof: date) -> float:
        return float(self.annual_rate_simple)
