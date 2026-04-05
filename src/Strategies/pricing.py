from __future__ import annotations
import math
from dataclasses import dataclass
from datetime import date

from .config import MarketConventions
from .instruments import OptionContract, OptionType
from .math_utils import norm_cdf, norm_pdf, clamp


@dataclass(frozen=True)
class BlackScholesInputs:
    spot: float
    strike: float
    time_to_expiry_years: float
    rate_cc: float
    dividend_yield_cc: float
    vol: float


@dataclass(frozen=True)
class OptionGreeksPerShare:
    price: float
    delta: float
    gamma: float
    vega: float
    theta_per_day: float


@dataclass
class BlackScholesPricer:
    conventions: MarketConventions = MarketConventions()

    def _d1_d2(self, x: BlackScholesInputs) -> tuple[float, float]:
        T = max(x.time_to_expiry_years, 1e-10)
        vol = max(x.vol, 1e-10)

        num = math.log(x.spot / x.strike) + (
            x.rate_cc - x.dividend_yield_cc + 0.5 * vol * vol
        ) * T
        den = vol * math.sqrt(T)

        d1 = num / den
        d2 = d1 - vol * math.sqrt(T)
        return d1, d2

    def price(self, contract: OptionContract, x: BlackScholesInputs) -> float:
        d1, d2 = self._d1_d2(x)
        S, K, T = x.spot, x.strike, x.time_to_expiry_years
        r, q = x.rate_cc, x.dividend_yield_cc

        disc_r = math.exp(-r * T)
        disc_q = math.exp(-q * T)

        if contract.option_type == OptionType.CALL:
            return disc_q * S * norm_cdf(d1) - disc_r * K * norm_cdf(d2)
        else:
            return disc_r * K * norm_cdf(-d2) - disc_q * S * norm_cdf(-d1)

    def theta_per_year(self, contract: OptionContract, x: BlackScholesInputs) -> float:
        d1, d2 = self._d1_d2(x)
        S, K, T = x.spot, x.strike, x.time_to_expiry_years
        r, q, vol = x.rate_cc, x.dividend_yield_cc, x.vol

        disc_r = math.exp(-r * T)
        disc_q = math.exp(-q * T)

        first_term = -(disc_q * S * norm_pdf(d1) * vol) / (
            2.0 * math.sqrt(max(T, 1e-10))
        )

        if contract.option_type == OptionType.CALL:
            return first_term + q * disc_q * S * norm_cdf(d1) - r * disc_r * K * norm_cdf(d2)

        call_theta = first_term + q * disc_q * S * norm_cdf(d1) - r * disc_r * K * norm_cdf(d2)
        return call_theta + r * disc_r * K - q * disc_q * S

    def theta_per_day(self, contract: OptionContract, x: BlackScholesInputs) -> float:
        theta_y = self.theta_per_year(contract, x)
        return theta_y / self.conventions.day_count.days_in_year

    def delta(self, contract: OptionContract, x: BlackScholesInputs) -> float:
        d1, _ = self._d1_d2(x)
        disc_q = math.exp(-x.dividend_yield_cc * x.time_to_expiry_years)

        if contract.option_type == OptionType.CALL:
            return disc_q * norm_cdf(d1)
        else:
            return disc_q * (norm_cdf(d1) - 1.0)

    def gamma(self, contract: OptionContract, x: BlackScholesInputs) -> float:
        d1, _ = self._d1_d2(x)
        S = x.spot
        T = max(x.time_to_expiry_years, 1e-10)
        vol = max(x.vol, 1e-10)

        disc_q = math.exp(-x.dividend_yield_cc * T)
        return (disc_q * norm_pdf(d1)) / (S * vol * math.sqrt(T))

    def vega(self, contract: OptionContract, x: BlackScholesInputs) -> float:
        d1, _ = self._d1_d2(x)
        S = x.spot
        T = max(x.time_to_expiry_years, 1e-10)

        disc_q = math.exp(-x.dividend_yield_cc * T)
        return disc_q * S * norm_pdf(d1) * math.sqrt(T)

    def greeks_per_share(self, contract: OptionContract, x: BlackScholesInputs) -> OptionGreeksPerShare:
        return OptionGreeksPerShare(
            price=self.price(contract, x),
            delta=self.delta(contract, x),
            gamma=self.gamma(contract, x),
            vega=self.vega(contract, x),
            theta_per_day=self.theta_per_day(contract, x),
        )

    @staticmethod
    def year_fraction(asof: date, expiry: date, day_count: float = 365.0) -> float:
        days = (expiry - asof).days
        return max(days, 0) / day_count

    @staticmethod
    def to_continuous_rate(simple_annual_rate: float) -> float:
        r = clamp(simple_annual_rate, -0.99, 10.0)
        return math.log(1.0 + r)