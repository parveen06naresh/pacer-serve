"""Download the raw market data Crash Replay is built from.

Index prices come from Yahoo Finance's public chart endpoint (with Stooq as a
fallback); rates and recession dates come from FRED. Writes one CSV per
series to data/raw/. Standard library only.
"""
import json
import pathlib
import time
import urllib.request
from datetime import datetime, timezone

FRED = ["DGS1MO", "DGS3MO", "DGS6MO", "DGS1", "DGS2", "DGS3", "DGS5",
        "DGS7", "DGS10", "DGS20", "DGS30", "GS10", "TB3MS", "USREC",
        "SP500"]
# symbol on Yahoo -> (file name, symbol on Stooq)
INDEXES = {"^GSPC": ("SPX", "^spx"), "^IXIC": ("NDQ", "^ndq"),
           "^VIX": ("VIX", None), "^DJI": ("DJI", "^dji")}

OUT = pathlib.Path(__file__).resolve().parent.parent / "data" / "raw"
UA = {"User-Agent": "Mozilla/5.0 (crash-replay data fetch)"}


def get(url: str) -> bytes:
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception as e:  # network hiccup: back off and retry
            print(f"  attempt {attempt + 1} failed for {url}: {e}")
            time.sleep(2 ** attempt)
    raise RuntimeError(url)


def fred(series: str) -> None:
    body = get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}")
    (OUT / f"{series}.csv").write_bytes(body)
    print(f"FRED {series}: {body.count(b'\n') - 1} rows")


def yahoo(symbol: str) -> str:
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/"
           f"{urllib.request.quote(symbol)}?period1=-1325635200"
           f"&period2={int(time.time())}&interval=1d&events=div,split")
    res = json.loads(get(url))["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    rows = ["date,open,high,low,close,volume"]
    for i, ts in enumerate(res["timestamp"]):
        c = q["close"][i]
        if c is None:
            continue
        d = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
        vals = [q[k][i] for k in ("open", "high", "low")]
        vals = [c if v is None else v for v in vals]
        rows.append(f"{d},{vals[0]:.4f},{vals[1]:.4f},{vals[2]:.4f},{c:.4f},"
                    f"{q['volume'][i] or 0}")
    return "\n".join(rows) + "\n"


def stooq(symbol: str) -> str:
    body = get(f"https://stooq.com/q/d/l/?s={symbol}&i=d").decode()
    if not body.lower().startswith("date"):
        raise RuntimeError(f"stooq returned no data for {symbol}")
    return body.replace("\r", "").lower()


def index(symbol: str, name: str, stooq_symbol) -> None:
    for source, fn, arg in (("yahoo", yahoo, symbol),
                            ("stooq", stooq, stooq_symbol)):
        if arg is None:
            continue
        try:
            text = fn(arg)
            (OUT / f"{name}.csv").write_text(text)
            print(f"{source} {name}: {text.count(chr(10)) - 1} rows")
            return
        except Exception as e:
            print(f"{source} {name} failed: {e}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for sym, (name, st) in INDEXES.items():
        index(sym, name, st)
    for s in FRED:
        try:
            fred(s)
        except Exception as e:
            print(f"FRED {s} failed: {e}")
