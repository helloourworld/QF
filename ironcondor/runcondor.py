import utils as ut
import pipeline as pl
import ironcondor as ic
from datetime import datetime
from typing import Tuple


def input_condor() -> Tuple[str, str]:
    # ticker = input("Ticker (e.g. GOOG, NVDA): ").strip().upper()
    ticker = "AAPL"  # for testing purposes
    
    # date_str = input("Expiration date (YYYY-MM-DD) or (YYYY M D): ").strip()
    date_str = "2026-09-18"  # for testing purposes

    # normalize simple formats to YYYY-MM-DD
    try:
        if '-' in date_str:
            exp = datetime.strptime(date_str, "%Y-%m-%d")
        else:
            parts = date_str.replace('/', ' ').split()
            exp = datetime(int(parts[0]), int(parts[1]), int(parts[2]))
    except Exception:
        raise ValueError("Invalid date format. Use YYYY-MM-DD or 'YYYY M D'")

    return ticker, exp.strftime("%Y-%m-%d")


def main() -> None:
    try:
        ticker, exp_date = input_condor()
        options_json, truncated, current_price = pl.parseOptionsChain(ticker, exp_date)
        calls, puts = pl.formatOptionChain(options_json, truncated, current_price)

        sCalls, lCalls = ic.formatCalls(calls)
        sPuts, lPuts = ic.formatPuts(puts)

        spreads = ic.generate_spreads([sCalls, lCalls, sPuts, lPuts])
        iron_condors = ic.generate_ironcondor(spreads)

        if not iron_condors:
            print("No valid iron condors found for the given inputs.")
        else:
            ic.output_condors(iron_condors, ticker, exp_date)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
