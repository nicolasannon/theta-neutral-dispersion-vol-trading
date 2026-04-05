from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from typing import Literal

from .instruments import OptionContract, OptionType, Stock
from .market_data import OptionMarketDataProvider, RateProvider
from .pricing import BlackScholesInputs, BlackScholesPricer
from .strategies import DispersionSizer, MatchMetric


@dataclass(frozen=True)
class Straddle:
    call: OptionContract
    put: OptionContract


@dataclass(frozen=True)
class DispersionTradeSetup:
    asof: date
    expiry: date
    metric: MatchMetric
    spy_straddle: Straddle
    aapl_straddle: Straddle
    qty_spy_straddle: float
    qty_aapl_straddle: float
    spot_by_symbol: dict[str, float]
    dividend_yield_cc_by_symbol: dict[str, float]
    rate_cc: float
    vol_by_option: dict[OptionContract, float]
    option_mark_by_option: dict[OptionContract, float]
    spy_metric_value: float
    aapl_metric_value: float


@dataclass
class DispersionTradeBuilder:
    pricer: BlackScholesPricer
    sizer: DispersionSizer
    spy_symbol: str = "SPY"
    component_symbol: str = "AAPL"

    def build_theta_neutral_trade(
        self,
        asof: date,
        market_data: OptionMarketDataProvider,
        rates: RateProvider,
        metric: MatchMetric,
        base_qty_spy_straddle: float,
        min_days: int,
        max_days: int,
    ) -> DispersionTradeSetup:
        if hasattr(market_data, "set_asof"):
            market_data.set_asof(asof)

        spot_spy = market_data.get_spot(self.spy_symbol)
        spot_component = market_data.get_spot(self.component_symbol)

        q_spy_cc = self.pricer.to_continuous_rate(market_data.get_dividend_yield(self.spy_symbol))
        q_component_cc = self.pricer.to_continuous_rate(market_data.get_dividend_yield(self.component_symbol))
        r_cc = self.pricer.to_continuous_rate(rates.get_annual_rate(asof))

        exp_spy = market_data.list_expirations(self.spy_symbol)
        exp_component = market_data.list_expirations(self.component_symbol)
        expiry = pick_common_expiry(exp_spy, exp_component, asof, min_days, max_days)

        k_spy = pick_atm_strike(self.spy_symbol, expiry, market_data, spot_spy)
        k_component = pick_atm_strike(self.component_symbol, expiry, market_data, spot_component)

        spy_straddle = build_straddle(self.spy_symbol, expiry, k_spy)
        component_straddle = build_straddle(self.component_symbol, expiry, k_component)

        q_spy_call = market_data.get_option_quote(spy_straddle.call)
        q_spy_put = market_data.get_option_quote(spy_straddle.put)
        q_component_call = market_data.get_option_quote(component_straddle.call)
        q_component_put = market_data.get_option_quote(component_straddle.put)

        quotes = [q_spy_call, q_spy_put, q_component_call, q_component_put]
        if any(q.implied_vol is None for q in quotes):
            raise ValueError("Missing implied volatility in option chain.")

        spy_metric_1 = straddle_metric_per_contract(
            metric=metric,
            asof=asof,
            straddle=spy_straddle,
            spot=spot_spy,
            r_cc=r_cc,
            q_cc=q_spy_cc,
            iv_call=float(q_spy_call.implied_vol),
            iv_put=float(q_spy_put.implied_vol),
            pricer=self.pricer,
        )

        component_metric_1 = straddle_metric_per_contract(
            metric=metric,
            asof=asof,
            straddle=component_straddle,
            spot=spot_component,
            r_cc=r_cc,
            q_cc=q_component_cc,
            iv_call=float(q_component_call.implied_vol),
            iv_put=float(q_component_put.implied_vol),
            pricer=self.pricer,
        )

        sizing = self.sizer.match(
            metric=metric,
            qty_spy_straddle=base_qty_spy_straddle,
            spy_metric_per_straddle=spy_metric_1,
            aapl_metric_per_straddle=component_metric_1,
        )

        option_mark_by_option = {
            spy_straddle.call: must_get_price(market_data.get_option_mark(spy_straddle.call), spy_straddle.call.pretty_symbol()),
            spy_straddle.put: must_get_price(market_data.get_option_mark(spy_straddle.put), spy_straddle.put.pretty_symbol()),
            component_straddle.call: must_get_price(market_data.get_option_mark(component_straddle.call), component_straddle.call.pretty_symbol()),
            component_straddle.put: must_get_price(market_data.get_option_mark(component_straddle.put), component_straddle.put.pretty_symbol()),
        }

        vol_by_option = {
            spy_straddle.call: float(q_spy_call.implied_vol),
            spy_straddle.put: float(q_spy_put.implied_vol),
            component_straddle.call: float(q_component_call.implied_vol),
            component_straddle.put: float(q_component_put.implied_vol),
        }

        return DispersionTradeSetup(
            asof=asof,
            expiry=expiry,
            metric=metric,
            spy_straddle=spy_straddle,
            aapl_straddle=component_straddle,
            qty_spy_straddle=sizing.qty_spy_straddle,
            qty_aapl_straddle=sizing.qty_aapl_straddle,
            spot_by_symbol={
                self.spy_symbol: float(spot_spy),
                self.component_symbol: float(spot_component),
            },
            dividend_yield_cc_by_symbol={
                self.spy_symbol: float(q_spy_cc),
                self.component_symbol: float(q_component_cc),
            },
            rate_cc=float(r_cc),
            vol_by_option=vol_by_option,
            option_mark_by_option=option_mark_by_option,
            spy_metric_value=sizing.spy_metric_value,
            aapl_metric_value=sizing.aapl_metric_value,
        )


def pick_common_expiry(
    exp1: list[date],
    exp2: list[date],
    asof: date,
    min_days: int,
    max_days: int,
) -> date:
    common = sorted(set(exp1).intersection(set(exp2)))

    if not common:
        raise ValueError("No common expiry found between the two legs.")

    in_window = [d for d in common if min_days <= (d - asof).days <= max_days]
    if in_window:
        target_days = 0.5 * (min_days + max_days)
        return min(in_window, key=lambda d: abs((d - asof).days - target_days))

    return min(common, key=lambda d: abs((d - asof).days - 0.5 * (min_days + max_days)))


def pick_atm_strike(symbol: str, expiry: date, market_data: OptionMarketDataProvider, spot: float) -> float:
    stock = Stock(symbol)
    tmp_contract = OptionContract(
        underlying=stock,
        option_type=OptionType.CALL,
        strike=0.0,
        expiry=expiry,
    )

    chain = market_data.get_option_chain(tmp_contract)
    strikes = sorted(float(k) for k in chain["strike"].tolist())
    return min(strikes, key=lambda k: abs(k - spot))


def build_straddle(symbol: str, expiry: date, strike: float) -> Straddle:
    stock = Stock(symbol)
    return Straddle(
        call=OptionContract(stock, OptionType.CALL, strike, expiry),
        put=OptionContract(stock, OptionType.PUT, strike, expiry),
    )


def straddle_metric_per_contract(
    metric: MatchMetric,
    asof: date,
    straddle: Straddle,
    spot: float,
    r_cc: float,
    q_cc: float,
    iv_call: float,
    iv_put: float,
    pricer: BlackScholesPricer,
    multiplier: int = 100,
) -> float:
    out = 0.0

    for option_contract, implied_vol in [(straddle.call, iv_call), (straddle.put, iv_put)]:
        time_to_expiry = pricer.year_fraction(
            asof,
            option_contract.expiry,
            day_count=pricer.conventions.day_count.days_in_year,
        )

        inputs = BlackScholesInputs(
            spot=spot,
            strike=option_contract.strike,
            time_to_expiry_years=time_to_expiry,
            rate_cc=r_cc,
            dividend_yield_cc=q_cc,
            vol=implied_vol,
        )

        greeks = pricer.greeks_per_share(option_contract, inputs)

        if metric == "theta_notional":
            out += greeks.theta_per_day * multiplier
        elif metric == "vega_notional":
            out += greeks.vega * multiplier
        else:
            raise ValueError("Unknown metric.")

    return out


def must_get_price(value: float | None, label: str) -> float:
    if value is None:
        raise ValueError(f"Missing market price for {label}")
    return float(value)
