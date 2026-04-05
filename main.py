from __future__ import annotations
from datetime import date
from pathlib import Path
import logging

from src.Strategies.backtest import (
    BacktestConfig,
    BatchBacktestConfig,
    DispersionBacktester,
    DispersionBatchBacktester,
)
from src.Strategies.dispersion import DispersionTradeBuilder
from src.Strategies.market_data import CsvYieldCurveRateProvider, LocalParquetMarketData
from src.Strategies.portfolio import Portfolio
from src.Strategies.pricing import BlackScholesPricer
from src.Strategies.reporting import BacktestArtifactExporter
from src.Strategies.strategies import DeltaHedger, DispersionSizer
from src.Strategies.logger import logger


def build_components(asof: date) -> tuple[LocalParquetMarketData, CsvYieldCurveRateProvider, BlackScholesPricer, DispersionSizer, DeltaHedger, DispersionTradeBuilder]:
    market_data = LocalParquetMarketData(
        data_root=Path("data"),
        asof=asof,
        default_dividend_yield=0.0,
    )
    rates = CsvYieldCurveRateProvider(
        csv_path=Path("data/par-yield-curve-rates-2020-2023.csv"),
        tenor_col="1 Yr",
    )
    pricer = BlackScholesPricer()
    sizer = DispersionSizer(pricer=pricer)
    hedger = DeltaHedger(pricer=pricer)
    trade_builder = DispersionTradeBuilder(pricer=pricer, sizer=sizer)
    return market_data, rates, pricer, sizer, hedger, trade_builder


def run_snapshot() -> None:
    asof = date(2021, 9, 1)
    metric = "theta_notional"
    base_qty_spy_straddle = -1.0
    min_days = 20
    max_days = 45

    market_data, rates, pricer, sizer, hedger, trade_builder = build_components(asof)

    setup = trade_builder.build_theta_neutral_trade(
        asof=asof,
        market_data=market_data,
        rates=rates,
        metric=metric,
        base_qty_spy_straddle=base_qty_spy_straddle,
        min_days=min_days,
        max_days=max_days,
    )

    portfolio = Portfolio()
    portfolio.add_option_trade(setup.spy_straddle.call, setup.qty_spy_straddle, setup.option_mark_by_option[setup.spy_straddle.call])
    portfolio.add_option_trade(setup.spy_straddle.put, setup.qty_spy_straddle, setup.option_mark_by_option[setup.spy_straddle.put])
    portfolio.add_option_trade(setup.aapl_straddle.call, setup.qty_aapl_straddle, setup.option_mark_by_option[setup.aapl_straddle.call])
    portfolio.add_option_trade(setup.aapl_straddle.put, setup.qty_aapl_straddle, setup.option_mark_by_option[setup.aapl_straddle.put])

    theta_total = portfolio.theta_total_per_day(
        asof=asof,
        spot_by_symbol=setup.spot_by_symbol,
        rate_cc=setup.rate_cc,
        dividend_yield_cc_by_symbol=setup.dividend_yield_cc_by_symbol,
        vol_by_option=setup.vol_by_option,
        pricer=pricer,
    )
    delta_pre = portfolio.delta_by_symbol(
        asof=asof,
        spot_by_symbol=setup.spot_by_symbol,
        rate_cc=setup.rate_cc,
        dividend_yield_cc_by_symbol=setup.dividend_yield_cc_by_symbol,
        vol_by_option=setup.vol_by_option,
        pricer=pricer,
    )
    hedge_trades = hedger.hedge_to_zero_delta(
        portfolio=portfolio,
        asof=asof,
        spot_by_symbol=setup.spot_by_symbol,
        rate_cc=setup.rate_cc,
        dividend_yield_cc_by_symbol=setup.dividend_yield_cc_by_symbol,
        vol_by_option=setup.vol_by_option,
    )
    hedger.apply_hedge_trades(portfolio=portfolio, trades=hedge_trades, execution_spot_by_symbol=setup.spot_by_symbol)
    delta_post = portfolio.delta_by_symbol(
        asof=asof,
        spot_by_symbol=setup.spot_by_symbol,
        rate_cc=setup.rate_cc,
        dividend_yield_cc_by_symbol=setup.dividend_yield_cc_by_symbol,
        vol_by_option=setup.vol_by_option,
        pricer=pricer,
    )

    logger.info("=== Dispersion Trade Snapshot ===")
    logger.info(f"As-of: {setup.asof.isoformat()}")
    logger.info(f"Expiry: {setup.expiry.isoformat()}")
    logger.info("")

    logger.info("=== Straddles ===")
    logger.info(f"SPY spot={setup.spot_by_symbol['SPY']:.4f} strike={setup.spy_straddle.call.strike:.2f} qty={setup.qty_spy_straddle:.6f}")
    logger.info(f"AAPL spot={setup.spot_by_symbol['AAPL']:.4f} strike={setup.aapl_straddle.call.strike:.2f} qty={setup.qty_aapl_straddle:.6f}")
    logger.info("")

    logger.info("=== Metric Matching ===")
    logger.info(f"Metric used: {setup.metric}")
    logger.info(f"SPY total metric: {setup.spy_metric_value:.6f}")
    logger.info(f"AAPL total metric: {setup.aapl_metric_value:.6f}")
    logger.info("")

    logger.info("=== Portfolio Theta ===")
    logger.info(f"Portfolio theta/day: {theta_total:.6f}")
    logger.info("")

    logger.info("=== Delta by Symbol Before Hedge Execution ===")
    for symbol, delta_value in delta_pre.items():
        logger.info(f"{symbol}: {delta_value:.6f} shares")
    logger.info("")

    logger.info("=== Hedge Trades Executed ===")
    for trade in hedge_trades:
        logger.info(f"{trade.symbol}: trade {trade.trade_shares:.6f} shares -> new stock position {trade.new_total_shares:.6f}")
    logger.info("")

    logger.info("=== Delta by Symbol After Hedge Execution ===")
    for symbol, delta_value in delta_post.items():
        logger.info(f"{symbol}: {delta_value:.6f} shares")
    logger.info("")

    logger.info("=== Portfolio Value After Hedge Execution ===")
    logger.info(f"Portfolio market value: {portfolio.total_market_value(setup.spot_by_symbol, setup.option_mark_by_option):.6f}")
    logger.info(f"Cash: {portfolio.cash:.6f}")
    logger.info(f"Option market value: {portfolio.option_market_value(setup.option_mark_by_option):.6f}")
    logger.info(f"Stock market value: {portfolio.stock_market_value(setup.spot_by_symbol):.6f}")
    logger.info("")


def run_backtest() -> None:
    start_date = date(2021, 9, 1)
    output_root = Path("outputs")
    market_data, rates, pricer, sizer, hedger, trade_builder = build_components(start_date)

    backtester = DispersionBacktester(
        market_data=market_data,
        rates=rates,
        pricer=pricer,
        sizer=sizer,
        hedger=hedger,
        trade_builder=trade_builder,
    )

    result = backtester.run(
        BacktestConfig(
            start_date=start_date,
            metric="theta_notional",
            base_qty_spy_straddle=-1.0,
            min_days=20,
            max_days=45,
            rebalance_delta_daily=True,
        )
    )

    frame = result.to_frame()
    summary = result.summary()
    if frame.empty or summary is None:
        print("=== Backtest Summary ===")
        print("No rows produced by the backtest.")
        return

    exporter = BacktestArtifactExporter(output_root=output_root)
    single_paths = exporter.export_single(result)

    print("=== Backtest Summary ===")
    print(f"Start date: {summary.start_date.isoformat()}")
    print(f"Expiry: {summary.expiry.isoformat()}")
    print(f"Number of revaluation dates: {summary.num_revaluation_dates}")
    print(f"Gross initial portfolio value before costs: {summary.initial_gross_portfolio_value_before_costs:.6f}")
    print(f"Initial entry costs: {summary.initial_entry_costs:.6f}")
    print(f"Net initial portfolio value: {summary.initial_net_portfolio_value:.6f}")
    print(f"Final portfolio value: {summary.final_portfolio_value:.6f}")
    print(f"Final cumulative PnL: {summary.final_cumulative_pnl:.6f}")
    print(f"Final options PnL contribution: {summary.final_options_pnl_contribution:.6f}")
    print(f"Final stock PnL contribution: {summary.final_stock_pnl_contribution:.6f}")
    print(f"Total transaction costs: {summary.total_transaction_costs:.6f}")
    print(
        "Accounting control: "
        f"options pnl + stocks pnl - costs = {summary.accounting_control_value:.6f} | "
        f"total pnl = {summary.final_cumulative_pnl:.6f} | "
        f"residual = {summary.accounting_control_residual:.6e}"
    )
    print(f"Final lifecycle event: {summary.final_lifecycle_event}")
    print(f"Final SPY stock hedge: {summary.final_spy_stock_hedge:.6f} shares")
    print(f"Final AAPL stock hedge: {summary.final_aapl_stock_hedge:.6f} shares")
    print()

    print("=== Trading Cost Assumptions ===")
    print(f"Stock proportional cost: {result.config.stock_transaction_cost_bps:.2f} bps")
    print(f"Stock half-spread add-on: {result.config.stock_half_spread_bps:.2f} bps")
    print()

    print("=== Robustness Diagnostics ===")
    print(f"Max |SPY delta pre-hedge|: {summary.max_abs_spy_delta_pre_hedge:.6f} shares")
    print(f"Max |AAPL delta pre-hedge|: {summary.max_abs_aapl_delta_pre_hedge:.6f} shares")
    print(f"Max |SPY stock hedge|: {summary.max_abs_spy_stock_position:.6f} shares")
    print(f"Max |AAPL stock hedge|: {summary.max_abs_aapl_stock_position:.6f} shares")
    print(f"Total stock turnover notional: {summary.total_stock_turnover_notional:.6f}")
    print(f"Average daily stock turnover notional: {summary.avg_daily_stock_turnover_notional:.6f}")
    print(f"Max drawdown: {summary.max_drawdown:.6f}")
    print()

    print("=== First 5 Backtest Rows ===")
    print(
        frame[[
            "asof",
            "lifecycle_event",
            "gross_portfolio_value_before_costs",
            "portfolio_value",
            "pnl_options",
            "pnl_stocks",
            "transaction_costs",
            "pnl_total",
            "pnl_cumulative",
            "spy_stock_position",
            "aapl_stock_position",
            "hedge_trade_spy",
            "hedge_trade_aapl",
            "spy_delta_pre_hedge",
            "aapl_delta_pre_hedge",
            "spy_delta_post_hedge",
            "aapl_delta_post_hedge",
        ]].head().to_string(index=False)
    )
    print()

    print("=== Last 5 Backtest Rows ===")
    print(
        frame[[
            "asof",
            "lifecycle_event",
            "gross_portfolio_value_before_costs",
            "portfolio_value",
            "pnl_options",
            "pnl_stocks",
            "transaction_costs",
            "pnl_total",
            "pnl_cumulative",
            "spy_stock_position",
            "aapl_stock_position",
            "hedge_trade_spy",
            "hedge_trade_aapl",
            "spy_delta_pre_hedge",
            "aapl_delta_pre_hedge",
            "spy_delta_post_hedge",
            "aapl_delta_post_hedge",
        ]].tail().to_string(index=False)
    )
    print()

    print("=== Exported Single-Run Artifacts ===")
    print(f"Daily CSV: {single_paths.daily_csv}")
    print(f"Summary CSV: {single_paths.summary_csv}")
    print(f"Cumulative PnL plot: {single_paths.cumulative_pnl_plot}")
    print(f"Daily PnL plot: {single_paths.daily_pnl_plot}")
    print(f"Drawdown plot: {single_paths.drawdown_plot}")
    print(f"PnL decomposition plot: {single_paths.pnl_decomposition_plot}")
    print(f"Stock positions plot: {single_paths.stock_positions_plot}")
    print(f"Transaction costs plot: {single_paths.transaction_costs_plot}")
    print(f"Hedge notionals plot: {single_paths.hedge_notionals_plot}")
    print()

    batch_backtester = DispersionBatchBacktester(single_backtester=backtester)
    batch_config = BatchBacktestConfig(
        start_date_from=date(2021, 1, 1),
        start_date_to=date(2022, 12, 31),
        frequency="monthly",
        max_entries=12,
        metric="theta_notional",
        base_qty_spy_straddle=-1.0,
        min_days=20,
        max_days=45,
        rebalance_delta_daily=True,
        stock_transaction_cost_bps=result.config.stock_transaction_cost_bps,
        stock_half_spread_bps=result.config.stock_half_spread_bps,
        liquidate_at_expiry=True,
        verbose=True,
    )
    batch_entry_dates = batch_backtester.get_entry_dates(batch_config)
    print("=== Starting Batch Backtests ===")
    print(f"Selected batch entry dates ({len(batch_entry_dates)}): {[d.isoformat() for d in batch_entry_dates]}")
    print()
    batch_result = batch_backtester.run(batch_config)
    batch_paths = exporter.export_batch(batch_result)
    batch_stats = batch_result.aggregate_statistics()
    batch_frame = batch_result.to_frame()

    print("=== Batch Entry-Date Diagnostics ===")
    if not batch_stats:
        print("No batch statistics available.")
    else:
        print(f"Number of successful runs: {int(batch_stats['num_runs'])}")
        print(f"Mean final PnL: {batch_stats['mean_final_pnl']:.6f}")
        print(f"Median final PnL: {batch_stats['median_final_pnl']:.6f}")
        print(f"Std final PnL: {batch_stats['std_final_pnl']:.6f}")
        print(f"Min final PnL: {batch_stats['min_final_pnl']:.6f}")
        print(f"Max final PnL: {batch_stats['max_final_pnl']:.6f}")
        print(f"Win rate: {batch_stats['win_rate']:.2%}")
        print(f"Mean total transaction costs: {batch_stats['mean_total_transaction_costs']:.6f}")
        print(f"Mean turnover notional: {batch_stats['mean_turnover_notional']:.6f}")
        print(f"Mean max drawdown: {batch_stats['mean_max_drawdown']:.6f}")
        print(f"Worst max drawdown: {batch_stats['worst_max_drawdown']:.6f}")
    print()

    print("=== First 5 Batch Rows ===")
    display_cols = [
        "start_date",
        "expiry",
        "num_revaluation_dates",
        "final_cumulative_pnl",
        "options_pnl_contribution",
        "stock_pnl_contribution",
        "total_transaction_costs",
        "max_drawdown",
        "status",
        "error_message",
    ]
    print(batch_frame[display_cols].head().to_string(index=False))
    print()
    print("=== Exported Batch Artifacts ===")
    print(f"Batch CSV: {batch_paths.summary_csv}")
    print(f"Batch aggregate stats CSV: {batch_paths.aggregate_stats_csv}")
    print(f"Final PnL by entry plot: {batch_paths.pnl_by_entry_plot}")
    print(f"Final PnL histogram: {batch_paths.pnl_histogram_plot}")


def main() -> None:
    run_snapshot()
    run_backtest()


if __name__ == "__main__":
    main()
