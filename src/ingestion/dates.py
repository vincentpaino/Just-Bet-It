"""
Date helpers shared by all ingestion code.

WHY THIS EXISTS
CFBD gives every kickoff as a UTC timestamp. Calling .date() on a UTC timestamp
pushes every evening kickoff onto the next calendar day (a Saturday night game
gets stored as Sunday), which breaks rest-day math and game ordering.

We store each game on the calendar day it is played in the U.S. To get that,
the kickoff is converted to Honolulu time first. Honolulu is the westernmost
U.S. zone and has no daylight saving, so even late Hawaii kickoffs (which can
cross midnight Eastern) land on the correct day.

If the kickoff time is still TBD, CFBD's timestamp is only a placeholder, so
its UTC date is used as given. The next ingestion run corrects it once a real
time is announced.
"""

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd

GAME_DAY_TZ = ZoneInfo("Pacific/Honolulu")


def _is_plain_date(value) -> bool:
    """True for a date with no time of day (datetime is a subclass of date)."""
    return isinstance(value, date) and not isinstance(value, datetime)


def as_utc_datetime(value):
    """
    Convert a datetime / pandas Timestamp / ISO string into a timezone-aware
    UTC datetime. Naive values are assumed to already be UTC.
    Returns None for missing values and for plain dates (no time of day).
    """
    if value is None or _is_plain_date(value):
        return None

    if pd.isna(value):
        return None

    ts = pd.Timestamp(value)

    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")

    return ts.to_pydatetime()


def to_game_date(value, time_tbd: bool = False):
    """The calendar day the game is played in the U.S. (see module docstring)."""
    if _is_plain_date(value):
        return value

    dt = as_utc_datetime(value)

    if dt is None:
        return None

    if time_tbd:
        return dt.date()

    return dt.astimezone(GAME_DAY_TZ).date()


def utc_date(value):
    """
    The UTC calendar date. This is how game dates were stored BEFORE the fix,
    so it is used to find and migrate old rows.
    """
    if _is_plain_date(value):
        return value

    dt = as_utc_datetime(value)

    return dt.date() if dt is not None else None
