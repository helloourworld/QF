import math
import numpy as np
import datetime as dt
from datetime import datetime, timezone, date
from typing import Iterable, List, Union


def truncateStrikes(optionStrikes: Iterable[Union[str, float]], stockPrice: float) -> List[float]:
    """Return strikes within +/-5% of the nearest strike to stockPrice (sorted)."""
    strikesToFloat = list(map(float, optionStrikes))
    os_arr = np.asarray(strikesToFloat)
    idx = int(np.argmin(np.abs(os_arr - stockPrice)))

    center = float(os_arr[idx])
    left_bound = center - (center * 0.05)
    right_bound = center + (center * 0.05)

    desired = [s for s in strikesToFloat if left_bound <= s <= right_bound]
    desired.sort()
    return desired


def calculateDelta(option_price: float, stock_price: float) -> float:
    """
    Crude fallback estimator for option delta in [0.0, 1.0].
    NOTE: This is a very rough approximation (option_price / stock_price) and
    should be replaced by one of:
      - using the 'delta' field from the option chain if available, or
      - computing delta from Black-Scholes using implied volatility, option type and time-to-expiry.

    Raises ValueError if stock_price <= 0.
    """
    if stock_price <= 0:
        raise ValueError("stock_price must be > 0 to estimate delta")
    est = option_price / stock_price
    return max(0.0, min(1.0, est))


def formatExpiryURL(expDate: Union[str, date, datetime]) -> int:
    """
    Convert an expiry (string 'YYYY-MM-DD' or date/datetime) to a UNIX timestamp (UTC seconds).
    This is suitable for Yahoo query ?date=<epoch_seconds>.
    """
    if isinstance(expDate, str):
        dt_obj = datetime.strptime(expDate, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    elif isinstance(expDate, datetime):
        dt_obj = expDate.astimezone(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    elif isinstance(expDate, date):
        dt_obj = datetime(expDate.year, expDate.month, expDate.day, tzinfo=timezone.utc)
    else:
        raise TypeError("expDate must be str 'YYYY-MM-DD', datetime.date or datetime.datetime")

    return int(dt_obj.timestamp())
