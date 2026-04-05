from __future__ import annotations
import math
SQRT_2PI = math.sqrt(2.0 * math.pi)
SQRT2 = math.sqrt(2.0)
def norm_pdf(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / SQRT_2PI

def norm_cdf(x: float) -> float:
    """Standard normal CDF using erf (no SciPy dependency)."""
    return 0.5 * (1.0 + math.erf(x / SQRT2))

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))