from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date
from typing import Literal
import logging

import pandas as pd

from .dispersion import DispersionTradeBuilder, DispersionTradeSetup
from .instruments import OptionContract
from .market_data import LocalParquetMarketData, RateProvider
from .portfolio import Portfolio
from .pricing import BlackScholesPricer
from .strategies import DeltaHedger, DeltaHedgeTrade, DispersionSizer, MatchMetric
from .logger import logger


@dataclass(frozen=True)
class BacktestConfig:
    start_date: date
    metric: MatchMetric = "theta_notional"
    base_qty_spy_straddle: float = -1.0
    min_days: int = 20
    max_days: int = 45
    rebalance_delta_daily: bool = True
    stock_transaction_cost_bps: float = 1.0
    stock_half_spread_bps: float = 0.0
    liquidate_at_expiry: bool = True


@dataclass(frozen=True)
class BatchBacktestConfig:
    start_date_from: date
    start_date_to: date | None = None
    frequency: Literal["daily", "weekly", "monthly"] = "monthly"
    max_entries: int | None = 12
    metric: MatchMetric = "theta_notional"
    base_qty_spy_straddle: float = -1.0
    min_days: int = 20
    max_days: int = 45
    rebalance_delta_daily: bool = True
    stock_transaction_cost_bps: float = 1.0
    stock_half_spread_bps: float = 0.0
    liquidate_at_expiry: bool = True
    verbose: bool = False


@dataclass(frozen=True)
class DailyMarketState:
    asof: date
    expiry: date
    rate_cc: float
    spot_by_symbol: dict[str, float]
    dividend_yield_cc_by_symbol: dict[str, float]
    vol_by_option: dict[OptionContract, float]
    option_mark_by_option: dict[OptionContract, float]


@dataclass(frozen=True)
class StockTradeExecutionCostModel:
    proportional_cost_bps: float = 1.0
    half_spread_bps: float = 0.0

    def estimate(self, shares: float, execution_price: float) -> float:
        notional = abs(shares) * execution_price
        total_bps = self.proportional_cost_bps + self.half_spread_bps
        return notional * total_bps / 10000.0


@dataclass(frozen=True)
class DailyBacktestRow:
    asof: date
    expiry: date
    lifecycle_event: str
    gross_portfolio_value_before_costs: float
    portfolio_value: float
    cash: float
    option_market_value: float
    stock_market_value: float
    pnl_options: float
    pnl_stocks: float
    pnl_total: float
    pnl_cumulative: float
    transaction_costs: float
    transaction_costs_cumulative: float
    theta_total: float
    spy_delta_pre_hedge: float
    aapl_delta_pre_hedge: float
    spy_delta_post_hedge: float
    aapl_delta_post_hedge: float
    spy_spot: float
    aapl_spot: float
    spy_stock_position: float
    aapl_stock_position: float
    hedge_trade_spy: float
    hedge_trade_aapl: float
    hedge_notional_spy: float
    hedge_notional_aapl: float


@dataclass(frozen=True)
class DispersionBacktestSummary:
    start_date: date
    expiry: date
    num_revaluation_dates: int
    initial_gross_portfolio_value_before_costs: float
    initial_entry_costs: float
    initial_net_portfolio_value: float
    final_portfolio_value: float
    final_cumulative_pnl: float
    final_options_pnl_contribution: float
    final_stock_pnl_contribution: float
    total_transaction_costs: float
    accounting_control_value: float
    accounting_control_residual: float
    final_lifecycle_event: str
    final_spy_stock_hedge: float
    final_aapl_stock_hedge: float
    max_abs_spy_delta_pre_hedge: float
    max_abs_aapl_delta_pre_hedge: float
    max_abs_spy_stock_position: float
    max_abs_aapl_stock_position: float
    total_stock_turnover_notional: float
    avg_daily_stock_turnover_notional: float
    max_drawdown: float


@dataclass(frozen=True)
class BatchBacktestSummaryRow:
    start_date: date
    expiry: date | None
    num_revaluation_dates: int
    final_cumulative_pnl: float
    options_pnl_contribution: float
    stock_pnl_contribution: float
    total_transaction_costs: float
    accounting_control_residual: float
    max_abs_spy_stock_position: float
    max_abs_aapl_stock_position: float
    total_stock_turnover_notional: float
    max_drawdown: float
    status: str
    error_message: str = ""


@dataclass
class DispersionBacktestResult:
    config: BacktestConfig
    trade_setup: DispersionTradeSetup
    rows: list[DailyBacktestRow] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([row.__dict__ for row in self.rows])

    def summary(self) -> DispersionBacktestSummary | None:
        frame = self.to_frame()
        if frame.empty:
            return None

        first_row = frame.iloc[0]
        last_row = frame.iloc[-1]
        options_pnl = float(frame["pnl_options"].sum())
        stocks_pnl = float(frame["pnl_stocks"].sum())
        total_costs = float(frame["transaction_costs"].sum())
        accounting_control_value = options_pnl + stocks_pnl - total_costs
        accounting_control_residual = accounting_control_value - float(last_row["pnl_cumulative"])
        total_stock_turnover_notional = float((frame["hedge_notional_spy"].abs() + frame["hedge_notional_aapl"].abs()).sum())
        running_max = frame["pnl_cumulative"].cummax()
        drawdown = frame["pnl_cumulative"] - running_max
        max_drawdown = float(drawdown.min())

        return DispersionBacktestSummary(
            start_date=self.config.start_date,
            expiry=self.trade_setup.expiry,
            num_revaluation_dates=len(frame),
            initial_gross_portfolio_value_before_costs=float(first_row["gross_portfolio_value_before_costs"]),
            initial_entry_costs=float(first_row["transaction_costs"]),
            initial_net_portfolio_value=float(first_row["portfolio_value"]),
            final_portfolio_value=float(last_row["portfolio_value"]),
            final_cumulative_pnl=float(last_row["pnl_cumulative"]),
            final_options_pnl_contribution=options_pnl,
            final_stock_pnl_contribution=stocks_pnl,
            total_transaction_costs=total_costs,
            accounting_control_value=accounting_control_value,
            accounting_control_residual=accounting_control_residual,
            final_lifecycle_event=str(last_row["lifecycle_event"]),
            final_spy_stock_hedge=float(last_row["spy_stock_position"]),
            final_aapl_stock_hedge=float(last_row["aapl_stock_position"]),
            max_abs_spy_delta_pre_hedge=float(frame["spy_delta_pre_hedge"].abs().max()),
            max_abs_aapl_delta_pre_hedge=float(frame["aapl_delta_pre_hedge"].abs().max()),
            max_abs_spy_stock_position=float(frame["spy_stock_position"].abs().max()),
            max_abs_aapl_stock_position=float(frame["aapl_stock_position"].abs().max()),
            total_stock_turnover_notional=total_stock_turnover_notional,
            avg_daily_stock_turnover_notional=total_stock_turnover_notional / len(frame),
            max_drawdown=max_drawdown,
        )


@dataclass
class DispersionBatchBacktestResult:
    config: BatchBacktestConfig
    rows: list[BatchBacktestSummaryRow] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([row.__dict__ for row in self.rows])

    def aggregate_statistics(self) -> dict[str, float]:
        frame = self.to_frame()
        if frame.empty:
            return {}

        good = frame[frame["status"] == "ok"].copy()
        if good.empty:
            return {"num_runs": 0.0}

        final_pnl = pd.to_numeric(good["final_cumulative_pnl"], errors="coerce")
        return {
            "num_runs": float(len(good)),
            "mean_final_pnl": float(final_pnl.mean()),
            "median_final_pnl": float(final_pnl.median()),
            "std_final_pnl": float(final_pnl.std(ddof=0)),
            "min_final_pnl": float(final_pnl.min()),
            "max_final_pnl": float(final_pnl.max()),
            "win_rate": float((final_pnl > 0.0).mean()),
            "mean_total_transaction_costs": float(pd.to_numeric(good["total_transaction_costs"], errors="coerce").mean()),
            "mean_turnover_notional": float(pd.to_numeric(good["total_stock_turnover_notional"], errors="coerce").mean()),
            "mean_max_drawdown": float(pd.to_numeric(good["max_drawdown"], errors="coerce").mean()),
            "worst_max_drawdown": float(pd.to_numeric(good["max_drawdown"], errors="coerce").min()),
        }


@dataclass
class DispersionBacktester:
    market_data: LocalParquetMarketData
    rates: RateProvider
    pricer: BlackScholesPricer
    sizer: DispersionSizer
    hedger: DeltaHedger
    trade_builder: DispersionTradeBuilder | None = None

    def __post_init__(self) -> None:
        if self.trade_builder is None:
            self.trade_builder = DispersionTradeBuilder(pricer=self.pricer, sizer=self.sizer)

    def run(self, config: BacktestConfig) -> DispersionBacktestResult:
        setup = self.trade_builder.build_theta_neutral_trade(
            asof=config.start_date,
            market_data=self.market_data,
            rates=self.rates,
            metric=config.metric,
            base_qty_spy_straddle=config.base_qty_spy_straddle,
            min_days=config.min_days,
            max_days=config.max_days,
        )

        portfolio = Portfolio()
        portfolio.add_option_trade(setup.spy_straddle.call, setup.qty_spy_straddle, setup.option_mark_by_option[setup.spy_straddle.call])
        portfolio.add_option_trade(setup.spy_straddle.put, setup.qty_spy_straddle, setup.option_mark_by_option[setup.spy_straddle.put])
        portfolio.add_option_trade(setup.aapl_straddle.call, setup.qty_aapl_straddle, setup.option_mark_by_option[setup.aapl_straddle.call])
        portfolio.add_option_trade(setup.aapl_straddle.put, setup.qty_aapl_straddle, setup.option_mark_by_option[setup.aapl_straddle.put])

        rows: list[DailyBacktestRow] = []
        common_dates = self._get_relevant_dates(config.start_date, setup.expiry)
        if not common_dates:
            return DispersionBacktestResult(config=config, trade_setup=setup, rows=[])

        execution_cost_model = StockTradeExecutionCostModel(
            proportional_cost_bps=config.stock_transaction_cost_bps,
            half_spread_bps=config.stock_half_spread_bps,
        )

        previous_option_value = portfolio.option_market_value(setup.option_mark_by_option)
        previous_total_value = portfolio.total_market_value(setup.spot_by_symbol, setup.option_mark_by_option)
        previous_spot_by_symbol = dict(setup.spot_by_symbol)
        previous_stock_quantities = {"SPY": portfolio.get_stock_quantity("SPY"), "AAPL": portfolio.get_stock_quantity("AAPL")}
        pnl_cumulative = 0.0
        transaction_costs_cumulative = 0.0

        for idx, current_date in enumerate(common_dates):
            market_state = self._build_market_state(current_date, setup)
            is_first_day = idx == 0
            is_expiry_day = current_date >= setup.expiry

            if is_expiry_day and config.liquidate_at_expiry:
                option_value_pre_actions = portfolio.option_payoff_value(market_state.spot_by_symbol)
            else:
                option_value_pre_actions = portfolio.option_market_value(market_state.option_mark_by_option)

            stock_value_pre_actions = portfolio.stock_market_value(market_state.spot_by_symbol)
            gross_portfolio_value_before_costs = portfolio.cash + option_value_pre_actions + stock_value_pre_actions

            delta_pre = portfolio.delta_by_symbol(
                asof=current_date,
                spot_by_symbol=market_state.spot_by_symbol,
                rate_cc=market_state.rate_cc,
                dividend_yield_cc_by_symbol=market_state.dividend_yield_cc_by_symbol,
                vol_by_option=market_state.vol_by_option,
                pricer=self.pricer,
            )
            theta_total = portfolio.theta_total_per_day(
                asof=current_date,
                spot_by_symbol=market_state.spot_by_symbol,
                rate_cc=market_state.rate_cc,
                dividend_yield_cc_by_symbol=market_state.dividend_yield_cc_by_symbol,
                vol_by_option=market_state.vol_by_option,
                pricer=self.pricer,
            )

            pnl_options = 0.0 if is_first_day else option_value_pre_actions - previous_option_value
            pnl_stocks = 0.0 if is_first_day else self._stock_pnl_from_carry(
                previous_stock_quantities=previous_stock_quantities,
                previous_spot_by_symbol=previous_spot_by_symbol,
                current_spot_by_symbol=market_state.spot_by_symbol,
            )

            transaction_costs = 0.0
            hedge_trades: list[DeltaHedgeTrade] = []
            lifecycle_event = "initial_hedge" if is_first_day else "daily_rehedge"

            if is_expiry_day and config.liquidate_at_expiry:
                portfolio.settle_options_at_expiry(market_state.spot_by_symbol)
                hedge_trades, transaction_costs = self._liquidate_stock_hedges(
                    portfolio=portfolio,
                    spot_by_symbol=market_state.spot_by_symbol,
                    execution_cost_model=execution_cost_model,
                )
                lifecycle_event = "expiry_unwind"
            elif config.rebalance_delta_daily:
                hedge_trades = self.hedger.hedge_to_zero_delta(
                    portfolio=portfolio,
                    asof=current_date,
                    spot_by_symbol=market_state.spot_by_symbol,
                    rate_cc=market_state.rate_cc,
                    dividend_yield_cc_by_symbol=market_state.dividend_yield_cc_by_symbol,
                    vol_by_option=market_state.vol_by_option,
                )
                transaction_costs = self._apply_hedge_trades_with_costs(
                    portfolio=portfolio,
                    trades=hedge_trades,
                    spot_by_symbol=market_state.spot_by_symbol,
                    execution_cost_model=execution_cost_model,
                )

            transaction_costs_cumulative += transaction_costs

            delta_post = portfolio.delta_by_symbol(
                asof=current_date,
                spot_by_symbol=market_state.spot_by_symbol,
                rate_cc=market_state.rate_cc,
                dividend_yield_cc_by_symbol=market_state.dividend_yield_cc_by_symbol,
                vol_by_option=market_state.vol_by_option,
                pricer=self.pricer,
            )

            if is_expiry_day and config.liquidate_at_expiry:
                option_market_value = 0.0
                stock_market_value = 0.0
                portfolio_value = portfolio.cash
            else:
                option_market_value = portfolio.option_market_value(market_state.option_mark_by_option)
                stock_market_value = portfolio.stock_market_value(market_state.spot_by_symbol)
                portfolio_value = portfolio.cash + option_market_value + stock_market_value

            pnl_total = portfolio_value - previous_total_value
            pnl_cumulative += pnl_total

            hedge_trade_map = {trade.symbol: trade.trade_shares for trade in hedge_trades}
            rows.append(
                DailyBacktestRow(
                    asof=current_date,
                    expiry=setup.expiry,
                    lifecycle_event=lifecycle_event,
                    gross_portfolio_value_before_costs=gross_portfolio_value_before_costs,
                    portfolio_value=portfolio_value,
                    cash=portfolio.cash,
                    option_market_value=option_market_value,
                    stock_market_value=stock_market_value,
                    pnl_options=pnl_options,
                    pnl_stocks=pnl_stocks,
                    pnl_total=pnl_total,
                    pnl_cumulative=pnl_cumulative,
                    transaction_costs=transaction_costs,
                    transaction_costs_cumulative=transaction_costs_cumulative,
                    theta_total=theta_total,
                    spy_delta_pre_hedge=delta_pre.get("SPY", 0.0),
                    aapl_delta_pre_hedge=delta_pre.get("AAPL", 0.0),
                    spy_delta_post_hedge=delta_post.get("SPY", 0.0),
                    aapl_delta_post_hedge=delta_post.get("AAPL", 0.0),
                    spy_spot=market_state.spot_by_symbol["SPY"],
                    aapl_spot=market_state.spot_by_symbol["AAPL"],
                    spy_stock_position=portfolio.get_stock_quantity("SPY"),
                    aapl_stock_position=portfolio.get_stock_quantity("AAPL"),
                    hedge_trade_spy=hedge_trade_map.get("SPY", 0.0),
                    hedge_trade_aapl=hedge_trade_map.get("AAPL", 0.0),
                    hedge_notional_spy=abs(hedge_trade_map.get("SPY", 0.0)) * market_state.spot_by_symbol["SPY"],
                    hedge_notional_aapl=abs(hedge_trade_map.get("AAPL", 0.0)) * market_state.spot_by_symbol["AAPL"],
                )
            )

            previous_option_value = 0.0 if is_expiry_day and config.liquidate_at_expiry else option_value_pre_actions
            previous_total_value = portfolio_value
            previous_spot_by_symbol = dict(market_state.spot_by_symbol)
            previous_stock_quantities = {"SPY": portfolio.get_stock_quantity("SPY"), "AAPL": portfolio.get_stock_quantity("AAPL")}

        return DispersionBacktestResult(config=config, trade_setup=setup, rows=rows)

    def _get_relevant_dates(self, start_date: date, expiry: date) -> list[date]:
        available = self.market_data.list_common_dates(["SPY", "AAPL"])
        return [d for d in available if start_date <= d <= expiry]

    def _build_market_state(self, current_date: date, setup: DispersionTradeSetup) -> DailyMarketState:
        self.market_data.set_asof(current_date)

        spot_by_symbol = {"SPY": self.market_data.get_spot("SPY"), "AAPL": self.market_data.get_spot("AAPL")}
        dividend_yield_cc_by_symbol = {
            "SPY": self.pricer.to_continuous_rate(self.market_data.get_dividend_yield("SPY")),
            "AAPL": self.pricer.to_continuous_rate(self.market_data.get_dividend_yield("AAPL")),
        }
        rate_cc = self.pricer.to_continuous_rate(self.rates.get_annual_rate(current_date))

        option_contracts = [setup.spy_straddle.call, setup.spy_straddle.put, setup.aapl_straddle.call, setup.aapl_straddle.put]
        vol_by_option: dict[OptionContract, float] = {}
        option_mark_by_option: dict[OptionContract, float] = {}
        for option_contract in option_contracts:
            quote = self.market_data.get_option_quote(option_contract)
            if quote.implied_vol is None:
                raise ValueError(f"Missing implied volatility for {option_contract.pretty_symbol()} at {current_date.isoformat()}")
            mark = self.market_data.get_option_mark(option_contract)
            if mark is None:
                raise ValueError(f"Missing market price for {option_contract.pretty_symbol()} at {current_date.isoformat()}")
            vol_by_option[option_contract] = float(quote.implied_vol)
            option_mark_by_option[option_contract] = float(mark)

        return DailyMarketState(
            asof=current_date,
            expiry=setup.expiry,
            rate_cc=rate_cc,
            spot_by_symbol=spot_by_symbol,
            dividend_yield_cc_by_symbol=dividend_yield_cc_by_symbol,
            vol_by_option=vol_by_option,
            option_mark_by_option=option_mark_by_option,
        )

    @staticmethod
    def _stock_pnl_from_carry(
        previous_stock_quantities: dict[str, float],
        previous_spot_by_symbol: dict[str, float],
        current_spot_by_symbol: dict[str, float],
    ) -> float:
        total = 0.0
        for symbol, quantity in previous_stock_quantities.items():
            previous_spot = previous_spot_by_symbol.get(symbol, 0.0)
            current_spot = current_spot_by_symbol.get(symbol, previous_spot)
            total += quantity * (current_spot - previous_spot)
        return total

    @staticmethod
    def _apply_hedge_trades_with_costs(
        portfolio: Portfolio,
        trades: list[DeltaHedgeTrade],
        spot_by_symbol: dict[str, float],
        execution_cost_model: StockTradeExecutionCostModel,
    ) -> float:
        total_cost = 0.0
        for trade in trades:
            execution_price = spot_by_symbol[trade.symbol]
            transaction_cost = execution_cost_model.estimate(trade.trade_shares, execution_price)
            portfolio.apply_stock_trade(
                symbol=trade.symbol,
                shares=trade.trade_shares,
                price_per_share=execution_price,
                transaction_cost=transaction_cost,
            )
            total_cost += transaction_cost
        return total_cost

    @staticmethod
    def _liquidate_stock_hedges(
        portfolio: Portfolio,
        spot_by_symbol: dict[str, float],
        execution_cost_model: StockTradeExecutionCostModel,
    ) -> tuple[list[DeltaHedgeTrade], float]:
        trades: list[DeltaHedgeTrade] = []
        total_cost = 0.0
        for symbol in ["SPY", "AAPL"]:
            current_shares = portfolio.get_stock_quantity(symbol)
            if abs(current_shares) < 1e-12:
                continue
            unwind_shares = -current_shares
            execution_price = spot_by_symbol[symbol]
            transaction_cost = execution_cost_model.estimate(unwind_shares, execution_price)
            portfolio.apply_stock_trade(symbol=symbol, shares=unwind_shares, price_per_share=execution_price, transaction_cost=transaction_cost)
            trades.append(DeltaHedgeTrade(symbol=symbol, trade_shares=unwind_shares, new_total_shares=0.0))
            total_cost += transaction_cost
        return trades, total_cost


@dataclass
class DispersionBatchBacktester:
    single_backtester: DispersionBacktester


    def get_entry_dates(self, config: BatchBacktestConfig) -> list[date]:
        return self._select_entry_dates(config)

    def run(self, config: BatchBacktestConfig) -> DispersionBatchBacktestResult:
        entry_dates = self._select_entry_dates(config)
        rows: list[BatchBacktestSummaryRow] = []

        for idx, entry_date in enumerate(entry_dates):
            if config.verbose:
                logger.info(f"[Batch {idx + 1}/{len(entry_dates)}] Running entry date {entry_date.isoformat()}...")
            try:
                result = self.single_backtester.run(
                    BacktestConfig(
                        start_date=entry_date,
                        metric=config.metric,
                        base_qty_spy_straddle=config.base_qty_spy_straddle,
                        min_days=config.min_days,
                        max_days=config.max_days,
                        rebalance_delta_daily=config.rebalance_delta_daily,
                        stock_transaction_cost_bps=config.stock_transaction_cost_bps,
                        stock_half_spread_bps=config.stock_half_spread_bps,
                        liquidate_at_expiry=config.liquidate_at_expiry,
                    )
                )
                summary = result.summary()
                if summary is None:
                    rows.append(
                        BatchBacktestSummaryRow(
                            start_date=entry_date,
                            expiry=None,
                            num_revaluation_dates=0,
                            final_cumulative_pnl=float("nan"),
                            options_pnl_contribution=float("nan"),
                            stock_pnl_contribution=float("nan"),
                            total_transaction_costs=float("nan"),
                            accounting_control_residual=float("nan"),
                            max_abs_spy_stock_position=float("nan"),
                            max_abs_aapl_stock_position=float("nan"),
                            total_stock_turnover_notional=float("nan"),
                            max_drawdown=float("nan"),
                            status="empty",
                            error_message="",
                        )
                    )
                else:
                    rows.append(
                        BatchBacktestSummaryRow(
                            start_date=entry_date,
                            expiry=summary.expiry,
                            num_revaluation_dates=summary.num_revaluation_dates,
                            final_cumulative_pnl=summary.final_cumulative_pnl,
                            options_pnl_contribution=summary.final_options_pnl_contribution,
                            stock_pnl_contribution=summary.final_stock_pnl_contribution,
                            total_transaction_costs=summary.total_transaction_costs,
                            accounting_control_residual=summary.accounting_control_residual,
                            max_abs_spy_stock_position=summary.max_abs_spy_stock_position,
                            max_abs_aapl_stock_position=summary.max_abs_aapl_stock_position,
                            total_stock_turnover_notional=summary.total_stock_turnover_notional,
                            max_drawdown=summary.max_drawdown,
                            status="ok",
                            error_message="",
                        )
                    )
                    if config.verbose:
                        logger.info(f"[Batch {idx + 1}/{len(entry_dates)}] Completed | final pnl={summary.final_cumulative_pnl:.6f} | max drawdown={summary.max_drawdown:.6f}")
            except Exception as exc:
                rows.append(
                    BatchBacktestSummaryRow(
                        start_date=entry_date,
                        expiry=None,
                        num_revaluation_dates=0,
                        final_cumulative_pnl=float("nan"),
                        options_pnl_contribution=float("nan"),
                        stock_pnl_contribution=float("nan"),
                        total_transaction_costs=float("nan"),
                        accounting_control_residual=float("nan"),
                        max_abs_spy_stock_position=float("nan"),
                        max_abs_aapl_stock_position=float("nan"),
                        total_stock_turnover_notional=float("nan"),
                        max_drawdown=float("nan"),
                        status="error",
                        error_message=str(exc),
                    )
                )
                if config.verbose:
                    logger.error(f"[Batch {idx + 1}/{len(entry_dates)}] Error | {exc}")

        return DispersionBatchBacktestResult(config=config, rows=rows)

    def _select_entry_dates(self, config: BatchBacktestConfig) -> list[date]:
        common_dates = self.single_backtester.market_data.list_common_dates(["SPY", "AAPL"])
        eligible = [d for d in common_dates if d >= config.start_date_from and (config.start_date_to is None or d <= config.start_date_to)]
        if not eligible:
            return []

        selected: list[date] = []
        seen_keys: set[tuple[int, int] | tuple[int, int, int]] = set()
        for entry_date in eligible:
            if config.frequency == "daily":
                key = (entry_date.year, entry_date.month, entry_date.day)
            elif config.frequency == "weekly":
                iso = entry_date.isocalendar()
                key = (iso.year, iso.week)
            else:
                key = (entry_date.year, entry_date.month)

            if key in seen_keys:
                continue
            seen_keys.add(key)
            selected.append(entry_date)

            if config.max_entries is not None and len(selected) >= config.max_entries:
                break

        return selected
