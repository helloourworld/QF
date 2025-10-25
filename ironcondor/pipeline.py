from dataclasses import dataclass
import json
from urllib.request import urlopen
import yfinance as yf
import urllib.error
import utils as ut
from typing import List, Tuple, Any
import datetime


def strtobool(val: Any) -> bool:
    """Convert string/int/bool representation to boolean.
    Accepts booleans, ints (0/1), and common truthy/falsey strings.
    Raises ValueError for unrecognized values.
    """
    if isinstance(val, bool):
        return val
    if isinstance(val, int):
        return bool(val)
    if val is None:
        raise ValueError("invalid truth value: None")
    v = str(val).strip().lower()
    if v in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    if v in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    raise ValueError(f"invalid truth value {val!r}")


@dataclass
class Call:
    currentStrike: float
    currentPrice: float
    currentProbOTM: float
    currentIV: float
    currentITM: bool


@dataclass
class Put:
    currentStrike: float
    currentPrice: float
    currentProbOTM: float
    currentIV: float
    currentITM: bool


def parseOptionsChain(ticker: str, expDate) -> Tuple[dict, List[float], float]:
	print("Fetching options chain data using yfinance...")
	try:
		tk = yf.Ticker(ticker)

		# get current stock price (try info, fall back to recent close)
		currentStockPrice = None
		try:
			info_price = tk.info.get('regularMarketPrice')
			if info_price is not None:
				currentStockPrice = float(info_price)
		except Exception:
			currentStockPrice = None

		if currentStockPrice is None:
			hist = tk.history(period="2d")
			if not hist.empty:
				currentStockPrice = float(hist['Close'].iat[-1])
			else:
				raise RuntimeError(f"unable to determine current price for {ticker}")

		# normalize expiry to 'YYYY-MM-DD' format expected by yfinance
		expiry_str = None
		if isinstance(expDate, (int, float)):
			expiry_str = datetime.datetime.fromtimestamp(
			    int(expDate)).date().isoformat()
		else:
			s = str(expDate).strip()
			if s.isdigit():
				expiry_str = datetime.datetime.fromtimestamp(int(s)).date().isoformat()
			else:
				# try ISO date
				try:
					expiry_str = datetime.date.fromisoformat(s).isoformat()
				except Exception:
					# try common m/d/Y format
					parts = s.split('/')
					if len(parts) == 3:
						m, d, y = parts
						expiry_str = datetime.date(int(y), int(m), int(d)).isoformat()
					else:
						# last resort: ask ut to format then convert
						try:
							ts = int(ut.formatExpiryURL(expDate))
							expiry_str = datetime.datetime.fromtimestamp(ts).date().isoformat()
						except Exception:
							expiry_str = None

		# ensure expiry exists in ticker options
		available = list(getattr(tk, "options", []) or [])
		if expiry_str is None or expiry_str not in available:
			# try to pick closest expiry if exact not found
			if not available:
				raise RuntimeError(f"no option expirations available for {ticker}")
			# prefer exact match on date formats, otherwise pick first matching date or closest
			if expiry_str in available:
				chosen = expiry_str
			else:
				# try matching year-month-day substrings
				matched = next(
				    (a for a in available if a.startswith(str(expiry_str or ""))), None)
				chosen = matched or available[0]
		else:
			chosen = expiry_str

		# fetch option chain for chosen expiry
		opt = tk.option_chain(chosen)
		calls_df = opt.calls
		puts_df = opt.puts

		calls = calls_df.to_dict('records') if not calls_df.empty else []
		puts = puts_df.to_dict('records') if not puts_df.empty else []

		# build a minimal optionsJson with same shape the rest of the code expects
		strikes = sorted({float(r.get('strike', 0.0)) for r in (calls + puts)})
		optionsJson = {
			'options': [{
				'calls': calls,
				'puts': puts
			}],
			'quote': {'regularMarketPrice': currentStockPrice},
			'strikes': strikes
		}

		truncatedStrikes = ut.truncateStrikes(strikes, currentStockPrice)
		return (optionsJson, truncatedStrikes, currentStockPrice)
	except Exception as e:
		raise RuntimeError(
		    f"failed to fetch options for {ticker} via yfinance: {e}") from e


def formatOptionChain(optionsJson: dict, truncatedStrikes: List[float], currentStockPrice: float) -> Tuple[List[Call], List[Put]]:
    chainCalls = optionsJson.get('options', [{}])[0].get('calls', [])
    chainPuts = optionsJson.get('options', [{}])[0].get('puts', [])
    outputCalls: List[Call] = []
    outputPuts: List[Put] = []

    # ensure strike membership uses consistent numeric types
    truncated_set = set(float(s) for s in truncatedStrikes)

    for item in chainCalls:
        currentStrike = float(item.get('strike', 0.0))
        if currentStrike in truncated_set:
            currentPrice = float(item.get('lastPrice') or 0.0)
            currentIV = float(item.get('impliedVolatility') or 0.0)
            currentITM = bool(item.get('inTheMoney', False))
            # Use strike when calculating delta (more likely correct than using option lastPrice)
            try:
                currentProbOTM = 1.0 - \
                    ut.calculateDelta(currentStrike, currentStockPrice)
            except Exception:
                # fallback if ut.calculateDelta expects different args
                currentProbOTM = 1.0 - \
                    ut.calculateDelta(currentPrice, currentStockPrice)

            outputCalls.append(
                Call(currentStrike, currentPrice, currentProbOTM, currentIV, currentITM))

    for item in chainPuts:
        currentStrike = float(item.get('strike', 0.0))
        if currentStrike in truncated_set:
            currentPrice = float(item.get('lastPrice') or 0.0)
            currentIV = float(item.get('impliedVolatility') or 0.0)
            currentITM = bool(item.get('inTheMoney', False))
            try:
                currentProbOTM = 1.0 - \
                    ut.calculateDelta(currentStrike, currentStockPrice)
            except Exception:
                currentProbOTM = 1.0 - \
                    ut.calculateDelta(currentPrice, currentStockPrice)

            outputPuts.append(Put(currentStrike, currentPrice,
                              currentProbOTM, currentIV, currentITM))

    return (outputCalls, outputPuts)
