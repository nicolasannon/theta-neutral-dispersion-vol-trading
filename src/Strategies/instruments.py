from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from enum import Enum


class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"


@dataclass(frozen=True)
class Stock:
    symbol: str


@dataclass(frozen=True)
class OptionContract:
    """
    Simple representation of a listed equity/ETF option (vanilla).
    Note: in real markets, equity options are often American style.
    For this educational project we price them with Black–Scholes (European approximation).
    """
    underlying: Stock
    option_type: OptionType
    strike: float
    expiry: date

    def pretty_symbol(self) -> str:
        return f"{self.underlying.symbol} {self.expiry.isoformat()} {self.option_type.value.upper()} K={self.strike:g}"


@dataclass(frozen=True)
class OptionQuote:
    """
    Snapshot quote for one listed option strike.
    Prices are per share (not per contract).
    """
    last: float | None
    bid: float | None
    ask: float | None
    implied_vol: float | None  # decimal, e.g. 0.25 for 25%
    timestamp_utc: str | None = None

    def mid(self) -> float | None:
        if self.bid is not None and self.ask is not None and self.bid > 0 and self.ask > 0:
            return 0.5 * (self.bid + self.ask)
        return None

    def best_price(self) -> float | None:
        """
        Prefer mid if available; fallback to last.
        """
        m = self.mid()
        if m is not None:
            return m
        return self.last