# 08_build_calendar.py
# Builds the calendar table used as our stand-in for traffic data
# (no open, time-aligned Lahore traffic dataset exists).
#
# Needs:   pip install holidays hijridate
# Output:  data/processed/calendar_daily.csv      (one row per day)
# Run:     python data_scripts/08_build_calendar.py      (works from any folder)
#
# Decisions (state these in the report):
#  - Weekend = Saturday and Sunday.
#  - Public holidays come from the `holidays` package (country PK).
#  - Islamic dates (Eid, Ramadan) are CALCULATED, so they can be 1 day off from the
#    day Pakistan actually announced after moon sighting.
#  - School terms are not included (no reliable source found).

from pathlib import Path
import pandas as pd
import holidays
from hijridate import Gregorian

START, END = "2019-05-01", "2026-12-31"

days = pd.date_range(START, END, freq="D")
cal = pd.DataFrame({"date": days})

# 1. weekday and weekend
cal["day_of_week"] = cal["date"].dt.dayofweek            # Monday = 0 ... Sunday = 6
cal["is_weekend"] = cal["day_of_week"].isin([5, 6]).astype(int)

# 2. public holidays (names as given by the package)
pk = holidays.country_holidays("PK", years=range(2019, 2027))
cal["holiday_name"] = cal["date"].dt.date.map(lambda d: pk.get(d, ""))
cal["is_public_holiday"] = (cal["holiday_name"] != "").astype(int)

# 3. Eid periods (Eid-ul-Fitr and Eid-ul-Adha, as listed by the package)
cal["is_eid_period"] = cal["holiday_name"].str.contains("Eid-ul", regex=False).astype(int)

# 4. Ramadan = ninth month of the Islamic calendar
def hijri_month(ts):
    return Gregorian(ts.year, ts.month, ts.day).to_hijri().month

cal["is_ramadan"] = (cal["date"].map(hijri_month) == 9).astype(int)

# 5. save (date as plain text so it joins easily with the other tables)
cal["date"] = cal["date"].dt.strftime("%Y-%m-%d")
# Paths are relative to this script (not to the folder you run it from):
# <project>/data_scripts/08_build_calendar.py  ->  <project>/data/processed/calendar_daily.csv
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data" if (ROOT / "data").is_dir() else ROOT
OUT = DATA / "processed" / "calendar_daily.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)
cal.to_csv(OUT, index=False)

print("Saved:", OUT)
print("Rows:", len(cal), "| from", cal["date"].iloc[0], "to", cal["date"].iloc[-1])
print("Weekend days:", int(cal["is_weekend"].sum()),
      "| public holidays:", int(cal["is_public_holiday"].sum()),
      "| Ramadan days:", int(cal["is_ramadan"].sum()),
      "| Eid days:", int(cal["is_eid_period"].sum()))
