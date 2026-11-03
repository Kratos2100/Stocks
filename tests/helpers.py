from datetime import date, datetime, timedelta

from options_mas.models import Candle, DeltaSource, OptionContract, OptionType

D0 = date(2026, 6, 1)
D1 = date(2026, 6, 2)


def ts(d: date, i: int) -> datetime:
    """Bar start of the i-th 5-minute bar of session d."""
    return datetime(d.year, d.month, d.day, 9, 15) + timedelta(minutes=5 * i)


def bar(d, i, o, h, l, c, v=1000.0, buy=None, sell=None, src=DeltaSource.NONE):
    return Candle(ts(d, i), o, h, l, c, v, buy, sell, src if buy is not None else DeltaSource.NONE)


def contract(ot=OptionType.CE, strike=24000, und="NIFTY", lot=65):
    return OptionContract(und, date(2026, 6, 9), float(strike), ot, lot_size=lot)
