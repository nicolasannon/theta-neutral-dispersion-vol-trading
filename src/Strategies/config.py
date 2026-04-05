from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class DayCountConvention:
    """Conventions to convert annual quantities into per-day quantities."""
    days_in_year: float = 365.0  # Many desks quote theta per calendar day


@dataclass(frozen=True)
class MarketConventions:
    """Global conventions used in the project."""
    contract_multiplier: int = 100  # US equity options: 1 contract = 100 shares
    day_count: DayCountConvention = DayCountConvention()