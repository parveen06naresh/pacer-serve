"""Turn the raw CSVs into the compact data file the web app loads.

Output: web/data.js, which defines window.CR_DATA with one entry per index:
  start  first trading date (YYYY-MM-DD)
  gaps   calendar-day gap to each next trading date, one digit per day
  close  daily closes
  rf     cash yield on that day, annual % (3-month T-bill)
  dy     dividend yield on that day, annual % (S&P 500 only, Shiller data)
"""
import csv
import json
import pathlib
from datetime import date

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"


def read_csv(name, value_col):
    out = {}
    with open(RAW / name) as f:
        for row in csv.DictReader(f):
            v = row[value_col]
            if v in ("", "."):
                continue
            out[row[next(iter(row))]] = float(v)
    return out


def monthly_lookup(series):
    """date string -> value of the latest month at or before that date."""
    months = sorted(series)

    def at(d):
        key = d[:7] + "-01"
        lo, hi = 0, len(months) - 1
        best = months[0]
        while lo <= hi:
            mid = (lo + hi) // 2
            if months[mid] <= key:
                best = months[mid]
                lo = mid + 1
            else:
                hi = mid - 1
        return series[best]
    return at


def build(index_file, start, with_dividends):
    closes = read_csv(index_file, "close")
    days = sorted(d for d in closes if d >= start)
    tb3ms = monthly_lookup(read_csv("TB3MS.csv", "TB3MS"))
    dgs3mo = read_csv("DGS3MO.csv", "DGS3MO")

    shiller = {}
    last_div = None
    with open(RAW / "SHILLER.csv") as f:
        for row in csv.DictReader(f):
            div, px = float(row["Dividend"]), float(row["SP500"])
            if div > 0:
                last_div = div
            if last_div and px > 0:
                shiller[row["Date"]] = 100 * last_div / px
    dy_at = monthly_lookup(shiller)

    rf, dy, gaps = [], [], []
    last_rf = tb3ms(days[0])
    prev = None
    for d in days:
        if d in dgs3mo:
            last_rf = dgs3mo[d]
        elif d < "1981-09-01":
            last_rf = tb3ms(d)
        rf.append(round(last_rf, 2))
        dy.append(round(dy_at(d), 2) if with_dividends else 0)
        cur = date.fromisoformat(d)
        if prev is not None:
            g = (cur - prev).days
            assert 1 <= g <= 9, (d, g)
            gaps.append(str(g))
        prev = cur
    return {
        "start": days[0],
        "gaps": "".join(gaps),
        "close": [round(closes[d], 2) for d in days],
        "rf": rf,
        "dy": dy,
    }


if __name__ == "__main__":
    data = {
        "spx": build("SPX.csv", "1950-01-03", True),
        "ndq": build("NDQ.csv", "1971-02-05", False),
    }
    for k, v in data.items():
        print(k, v["start"], len(v["close"]), "days")
    out = ROOT / "web" / "data.js"
    out.write_text("window.CR_DATA=" + json.dumps(data, separators=(",", ":")) + ";\n")
    print(out, out.stat().st_size // 1024, "KB")
