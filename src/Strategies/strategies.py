from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from typing import Literal

from .instruments import OptionContract
from .portfolio import Portfolio
from .pricing import BlackScholesPricer


MatchMetric = Literal["theta_notional", "vega_notional"]


@dataclass(frozen=True)
class DispersionSizingResult:
    qty_spy_straddle: float
    qty_aapl_straddle: float
    metric: MatchMetric
    spy_metric_value: float
    aapl_metric_value: float


@dataclass
class DispersionSizer:
    """
    Match the size of the two legs using:
    - theta_notional: $/day
    - vega_notional: $ per 1.00 vol
    """
    pricer: BlackScholesPricer

    def match(
        self,
        metric: MatchMetric,
        qty_spy_straddle: float,
        spy_metric_per_straddle: float,
        aapl_metric_per_straddle: float,
    ) -> DispersionSizingResult:
        if abs(aapl_metric_per_straddle) < 1e-12:
            raise ValueError("AAPL metric is ~0, impossible to match.")

        qty_aapl = -(qty_spy_straddle * spy_metric_per_straddle) / aapl_metric_per_straddle

        return DispersionSizingResult(
            qty_spy_straddle=qty_spy_straddle,
            qty_aapl_straddle=qty_aapl,
            metric=metric,
            spy_metric_value=qty_spy_straddle * spy_metric_per_straddle,
            aapl_metric_value=qty_aapl * aapl_metric_per_straddle,
        )


@dataclass(frozen=True)
class DeltaHedgeTrade:
    symbol: str
    trade_shares: float
    new_total_shares: float


@dataclass
class DeltaHedger:
    """
    Dynamic stock hedge:
    adjust stock positions so that delta_shares = 0 for each underlying.
    """
    pricer: BlackScholesPricer

    def hedge_to_zero_delta(
        self,
        portfolio: Portfolio,
        asof: date,
        spot_by_symbol: dict[str, float],
        rate_cc: float,
        dividend_yield_cc_by_symbol: dict[str, float],
        vol_by_option: dict[OptionContract, float],
    ) -> list[DeltaHedgeTrade]:
        delta_map = portfolio.delta_by_symbol(
            asof=asof,
            spot_by_symbol=spot_by_symbol,
            rate_cc=rate_cc,
            dividend_yield_cc_by_symbol=dividend_yield_cc_by_symbol,
            vol_by_option=vol_by_option,
            pricer=self.pricer,
        )

        trades: list[DeltaHedgeTrade] = []
        for symbol, total_delta_shares in delta_map.items():
            trade_shares = -total_delta_shares
            if abs(trade_shares) < 1e-8:
                continue

            current_stock = portfolio.get_stock_quantity(symbol)
            trades.append(
                DeltaHedgeTrade(
                    symbol=symbol,
                    trade_shares=trade_shares,
                    new_total_shares=current_stock + trade_shares,
                )
            )

        return trades

    def apply_hedge_trades(
        self,
        portfolio: Portfolio,
        trades: list[DeltaHedgeTrade],
        execution_spot_by_symbol: dict[str, float],
    ) -> None:
        for trade in trades:
            portfolio.apply_stock_trade(
                symbol=trade.symbol,
                shares=trade.trade_shares,
                price_per_share=execution_spot_by_symbol[trade.symbol],
            )
