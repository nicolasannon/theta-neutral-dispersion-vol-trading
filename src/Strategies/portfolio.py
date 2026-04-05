from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date

from .config import MarketConventions
from .instruments import OptionContract, OptionType, Stock
from .pricing import BlackScholesInputs, BlackScholesPricer


@dataclass(frozen=True)
class Position:
    """
    Quantity convention:
    - Options: number of contracts
    - Stock: number of shares
    """
    instrument: Stock | OptionContract
    quantity: float


@dataclass
class Portfolio:
    positions: list[Position] = field(default_factory=list)
    conventions: MarketConventions = MarketConventions()
    cash: float = 0.0

    def add(self, position: Position) -> None:
        self._add_quantity(position.instrument, position.quantity)

    def _add_quantity(self, instrument: Stock | OptionContract, quantity_delta: float) -> None:
        if abs(quantity_delta) < 1e-12:
            return

        for idx, position in enumerate(self.positions):
            if position.instrument == instrument:
                new_quantity = position.quantity + quantity_delta
                if abs(new_quantity) < 1e-12:
                    self.positions.pop(idx)
                else:
                    self.positions[idx] = Position(instrument, new_quantity)
                return

        self.positions.append(Position(instrument, quantity_delta))

    def add_option_trade(
        self,
        contract: OptionContract,
        quantity_contracts: float,
        price_per_share: float,
    ) -> None:
        self._add_quantity(contract, quantity_contracts)
        self.cash -= quantity_contracts * self.conventions.contract_multiplier * price_per_share

    def apply_stock_trade(
        self,
        symbol: str,
        shares: float,
        price_per_share: float,
        transaction_cost: float = 0.0,
    ) -> None:
        self._add_quantity(Stock(symbol), shares)
        self.cash -= shares * price_per_share
        self.cash -= transaction_cost

    def get_stock_quantity(self, symbol: str) -> float:
        out = 0.0
        for position in self.positions:
            if isinstance(position.instrument, Stock) and position.instrument.symbol == symbol:
                out += position.quantity
        return out

    def option_market_value(self, option_mark_by_option: dict[OptionContract, float]) -> float:
        total = 0.0
        for position in self.positions:
            if isinstance(position.instrument, Stock):
                continue
            total += (
                position.quantity
                * self.conventions.contract_multiplier
                * option_mark_by_option[position.instrument]
            )
        return total

    def option_payoff_value(self, spot_by_symbol: dict[str, float]) -> float:
        total = 0.0
        for position in self.positions:
            if isinstance(position.instrument, Stock):
                continue

            contract = position.instrument
            spot = spot_by_symbol[contract.underlying.symbol]
            if contract.option_type == OptionType.CALL:
                intrinsic = max(spot - contract.strike, 0.0)
            else:
                intrinsic = max(contract.strike - spot, 0.0)

            total += position.quantity * self.conventions.contract_multiplier * intrinsic

        return total

    def settle_options_at_expiry(self, spot_by_symbol: dict[str, float]) -> float:
        settlement_value = self.option_payoff_value(spot_by_symbol)
        if abs(settlement_value) > 0.0:
            self.cash += settlement_value

        remaining_positions: list[Position] = []
        for position in self.positions:
            if isinstance(position.instrument, Stock):
                remaining_positions.append(position)
        self.positions = remaining_positions
        return settlement_value

    def stock_market_value(self, spot_by_symbol: dict[str, float]) -> float:
        total = 0.0
        for position in self.positions:
            if isinstance(position.instrument, Stock):
                total += position.quantity * spot_by_symbol[position.instrument.symbol]
        return total

    def total_market_value(
        self,
        spot_by_symbol: dict[str, float],
        option_mark_by_option: dict[OptionContract, float],
    ) -> float:
        return self.cash + self.option_market_value(option_mark_by_option) + self.stock_market_value(spot_by_symbol)

    def theta_total_per_day(
        self,
        asof: date,
        spot_by_symbol: dict[str, float],
        rate_cc: float,
        dividend_yield_cc_by_symbol: dict[str, float],
        vol_by_option: dict[OptionContract, float],
        pricer: BlackScholesPricer,
    ) -> float:
        total = 0.0

        for position in self.positions:
            if isinstance(position.instrument, Stock):
                continue

            option_contract = position.instrument
            spot = spot_by_symbol[option_contract.underlying.symbol]
            dividend_yield = dividend_yield_cc_by_symbol.get(option_contract.underlying.symbol, 0.0)
            time_to_expiry = pricer.year_fraction(asof, option_contract.expiry)
            vol = vol_by_option[option_contract]

            inputs = BlackScholesInputs(
                spot=spot,
                strike=option_contract.strike,
                time_to_expiry_years=time_to_expiry,
                rate_cc=rate_cc,
                dividend_yield_cc=dividend_yield,
                vol=vol,
            )

            theta_per_share = pricer.theta_per_day(option_contract, inputs)
            theta_per_contract = theta_per_share * self.conventions.contract_multiplier
            total += theta_per_contract * position.quantity

        return total

    def delta_by_symbol(
        self,
        asof: date,
        spot_by_symbol: dict[str, float],
        rate_cc: float,
        dividend_yield_cc_by_symbol: dict[str, float],
        vol_by_option: dict[OptionContract, float],
        pricer: BlackScholesPricer,
    ) -> dict[str, float]:
        """
        Return delta exposure in number of shares per underlying.
        Used for delta hedging each leg separately (SPY / AAPL).
        """
        delta_map: dict[str, float] = {}

        for position in self.positions:
            if isinstance(position.instrument, Stock):
                symbol = position.instrument.symbol
                delta_map[symbol] = delta_map.get(symbol, 0.0) + position.quantity
                continue

            option_contract = position.instrument
            symbol = option_contract.underlying.symbol
            spot = spot_by_symbol[symbol]
            dividend_yield = dividend_yield_cc_by_symbol.get(symbol, 0.0)
            time_to_expiry = pricer.year_fraction(asof, option_contract.expiry)
            vol = vol_by_option[option_contract]

            inputs = BlackScholesInputs(
                spot=spot,
                strike=option_contract.strike,
                time_to_expiry_years=time_to_expiry,
                rate_cc=rate_cc,
                dividend_yield_cc=dividend_yield,
                vol=vol,
            )

            delta_per_share = pricer.delta(option_contract, inputs)
            delta_per_contract = delta_per_share * self.conventions.contract_multiplier
            delta_map[symbol] = delta_map.get(symbol, 0.0) + delta_per_contract * position.quantity

        return delta_map
