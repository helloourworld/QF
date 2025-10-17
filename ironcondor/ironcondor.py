import utils as ut
import pipeline as pl
from typing import List, Tuple, Any


def formatCalls(inputCalls: List[pl.Call]) -> Tuple[List[pl.Call], List[pl.Call]]:

    sCall: List[pl.Call] = []
    lCall: List[pl.Call] = []

    for c in inputCalls:
        p = c.currentProbOTM
        sCallDeltaValid = 0.55 <= p <= 0.80
        lCallDeltaValid = 0.80 < p <= 0.99

        if sCallDeltaValid:
            sCall.append(c)
        elif lCallDeltaValid:
            lCall.append(c)

    return (sCall, lCall)


def formatPuts(inputPuts: List[pl.Put]) -> Tuple[List[pl.Put], List[pl.Put]]:

    sPut: List[pl.Put] = []
    lPut: List[pl.Put] = []

    for p in inputPuts:
        prob = p.currentProbOTM
        sPutDeltaValid = 0.55 <= prob <= 0.80
        lPutDeltaValid = 0.80 < prob <= 0.99

        if sPutDeltaValid:
            sPut.append(p)
        if lPutDeltaValid:
            lPut.append(p)

    sPut.reverse()
    lPut.reverse()

    return (sPut, lPut)


def generate_spreads(OTM_Strikes: List[List[Any]]) -> Tuple[List[Tuple[Any, Any]], List[Tuple[Any, Any]]]:
    """
    OTM_Strikes is expected to be [sCall, lCall, sPut, lPut]
    Returns (bearCallSpreads, bullPutSpreads) where each spread is (short_option, long_option)
    """
    bearCallSpreads: List[Tuple[Any, Any]] = []
    bullPutSpreads: List[Tuple[Any, Any]] = []

    # Bear call spreads: short from sCall, long from lCall where long strike > short strike
    for short in OTM_Strikes[0]:
        for long in OTM_Strikes[1]:
            if long.currentStrike > short.currentStrike:
                bearCallSpreads.append((short, long))

    # Bull put spreads: short from sPut, long from lPut where long strike > short strike
    for short in OTM_Strikes[2]:
        for long in OTM_Strikes[3]:
            if long.currentStrike > short.currentStrike:
                bullPutSpreads.append((short, long))

    return (bearCallSpreads, bullPutSpreads)


def generate_ironcondor(spreads: Tuple[List[Tuple[Any, Any]], List[Tuple[Any, Any]]]) -> List[Tuple[Any, Any, Any, Any]]:
    """
    Combine call and put spreads to form iron condors.
    Returns list of tuples: (shortPut, longPut, shortCall, longCall)
    """
    bearCallSpreads, bullPutSpreads = spreads
    ic_list: List[Tuple[Any, Any, Any, Any]] = []

    for call_short, call_long in bearCallSpreads:
        for put_short, put_long in bullPutSpreads:
            # Ensure put strikes are below call strikes (non-overlapping)
            if (call_short.currentStrike > put_short.currentStrike) and (call_long.currentStrike > put_long.currentStrike):
                ic_list.append((put_short, put_long, call_short, call_long))

    return ic_list


def output_condors(iron_condors: List[Tuple[Any, Any, Any, Any]], ticker: str, exp_date: str) -> None:

    print(f'Top Iron Condors for {ticker} at: {exp_date}')
    for idx, ic in enumerate(iron_condors, start=1):
        put_short, put_long, call_short, call_long = ic
        print(f"{idx}. PUTS  short@{put_short.currentStrike} ({put_short.currentPrice})  long@{put_long.currentStrike} ({put_long.currentPrice})"
              f"  |  CALLS short@{call_short.currentStrike} ({call_short.currentPrice})  long@{call_long.currentStrike} ({call_long.currentPrice})")

    # If you want to return structured data, return ic_list instead of None
    return None
