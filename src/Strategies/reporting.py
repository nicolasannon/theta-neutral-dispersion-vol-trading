from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from .backtest import DispersionBacktestResult, DispersionBatchBacktestResult


@dataclass(frozen=True)
class BacktestArtifactPaths:
    output_dir: Path
    daily_csv: Path
    summary_csv: Path
    cumulative_pnl_plot: Path
    daily_pnl_plot: Path
    drawdown_plot: Path
    pnl_decomposition_plot: Path
    stock_positions_plot: Path
    transaction_costs_plot: Path
    hedge_notionals_plot: Path


@dataclass(frozen=True)
class BatchArtifactPaths:
    output_dir: Path
    summary_csv: Path
    pnl_by_entry_plot: Path
    pnl_histogram_plot: Path
    aggregate_stats_csv: Path


@dataclass
class BacktestArtifactExporter:
    output_root: Path | str

    def export_single(self, result: DispersionBacktestResult, subdir_name: str = "single_backtest") -> BacktestArtifactPaths:
        output_dir = Path(self.output_root) / subdir_name
        output_dir.mkdir(parents=True, exist_ok=True)

        frame = result.to_frame().copy()
        summary = result.summary()
        if frame.empty or summary is None:
            raise ValueError("Cannot export artifacts for an empty backtest result")

        frame["asof"] = pd.to_datetime(frame["asof"])
        frame["pnl_options_cumulative"] = frame["pnl_options"].cumsum()
        frame["pnl_stocks_cumulative"] = frame["pnl_stocks"].cumsum()
        frame["transaction_costs_cumulative_from_flows"] = frame["transaction_costs"].cumsum()
        frame["running_max_pnl"] = frame["pnl_cumulative"].cummax()
        frame["drawdown"] = frame["pnl_cumulative"] - frame["running_max_pnl"]

        daily_csv = output_dir / "backtest_daily.csv"
        summary_csv = output_dir / "backtest_summary.csv"
        cumulative_pnl_plot = output_dir / "cumulative_pnl.png"
        daily_pnl_plot = output_dir / "daily_pnl.png"
        drawdown_plot = output_dir / "drawdown.png"
        pnl_decomposition_plot = output_dir / "pnl_decomposition.png"
        stock_positions_plot = output_dir / "stock_positions.png"
        transaction_costs_plot = output_dir / "transaction_costs.png"
        hedge_notionals_plot = output_dir / "hedge_notionals.png"

        frame.to_csv(daily_csv, index=False)
        pd.DataFrame([summary.__dict__]).to_csv(summary_csv, index=False)

        self._line_plot(frame, "asof", ["pnl_cumulative"], cumulative_pnl_plot, "Cumulative PnL")
        self._line_plot(frame, "asof", ["pnl_total"], daily_pnl_plot, "Daily PnL")
        self._line_plot(frame, "asof", ["drawdown"], drawdown_plot, "Drawdown")
        self._line_plot(
            frame,
            "asof",
            ["pnl_options_cumulative", "pnl_stocks_cumulative", "transaction_costs_cumulative_from_flows", "pnl_cumulative"],
            pnl_decomposition_plot,
            "PnL decomposition",
        )
        self._line_plot(frame, "asof", ["spy_stock_position", "aapl_stock_position"], stock_positions_plot, "Daily hedge stock positions")
        self._line_plot(frame, "asof", ["transaction_costs_cumulative"], transaction_costs_plot, "Cumulative transaction costs")
        self._line_plot(frame, "asof", ["hedge_notional_spy", "hedge_notional_aapl"], hedge_notionals_plot, "Daily hedge notionals")

        return BacktestArtifactPaths(
            output_dir=output_dir,
            daily_csv=daily_csv,
            summary_csv=summary_csv,
            cumulative_pnl_plot=cumulative_pnl_plot,
            daily_pnl_plot=daily_pnl_plot,
            drawdown_plot=drawdown_plot,
            pnl_decomposition_plot=pnl_decomposition_plot,
            stock_positions_plot=stock_positions_plot,
            transaction_costs_plot=transaction_costs_plot,
            hedge_notionals_plot=hedge_notionals_plot,
        )

    def export_batch(self, result: DispersionBatchBacktestResult, subdir_name: str = "batch_backtest") -> BatchArtifactPaths:
        output_dir = Path(self.output_root) / subdir_name
        output_dir.mkdir(parents=True, exist_ok=True)

        frame = result.to_frame().copy()
        if frame.empty:
            raise ValueError("Cannot export artifacts for an empty batch result")

        good = frame[frame["status"] == "ok"].copy()
        good["start_date"] = pd.to_datetime(good["start_date"])

        summary_csv = output_dir / "batch_backtest_summary.csv"
        aggregate_stats_csv = output_dir / "batch_backtest_aggregate_stats.csv"
        pnl_by_entry_plot = output_dir / "batch_final_pnl_by_entry.png"
        pnl_histogram_plot = output_dir / "batch_final_pnl_histogram.png"

        frame.to_csv(summary_csv, index=False)
        pd.DataFrame([result.aggregate_statistics()]).to_csv(aggregate_stats_csv, index=False)

        if not good.empty:
            plt.figure(figsize=(10, 6))
            plt.plot(good["start_date"], good["final_cumulative_pnl"])
            plt.title("Final PnL by entry date")
            plt.xlabel("Entry date")
            plt.ylabel("Final cumulative PnL")
            plt.grid(True, alpha=0.3)
            ax = plt.gca()
            locator = mdates.AutoDateLocator()
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
            plt.tight_layout()
            plt.savefig(pnl_by_entry_plot, dpi=150)
            plt.close()

            plt.figure(figsize=(10, 6))
            plt.hist(good["final_cumulative_pnl"].dropna(), bins=min(12, max(5, len(good))))
            plt.title("Distribution of final PnL")
            plt.xlabel("Final cumulative PnL")
            plt.ylabel("Frequency")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(pnl_histogram_plot, dpi=150)
            plt.close()
        else:
            for path in [pnl_by_entry_plot, pnl_histogram_plot]:
                path.write_text("No successful batch runs to plot.\n", encoding="utf-8")

        return BatchArtifactPaths(
            output_dir=output_dir,
            summary_csv=summary_csv,
            pnl_by_entry_plot=pnl_by_entry_plot,
            pnl_histogram_plot=pnl_histogram_plot,
            aggregate_stats_csv=aggregate_stats_csv,
        )

    @staticmethod
    def _line_plot(frame: pd.DataFrame, x_col: str, y_cols: list[str], output_path: Path, title: str) -> None:
        plt.figure(figsize=(10, 6))
        for y_col in y_cols:
            if y_col in frame.columns:
                plt.plot(frame[x_col], frame[y_col], label=y_col)
        plt.title(title)
        plt.xlabel(x_col)
        plt.ylabel("value")
        plt.grid(True, alpha=0.3)
        ax = plt.gca()
        if pd.api.types.is_datetime64_any_dtype(frame[x_col]):
            locator = mdates.AutoDateLocator()
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        if len(y_cols) > 1:
            plt.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
