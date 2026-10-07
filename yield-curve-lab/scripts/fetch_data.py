"""Download the raw FRED series Yield Curve Lab is built from.

Writes one CSV per series to data/raw/. Needs only the standard library,
so it runs anywhere (GitHub Actions, a laptop, Colab).
"""
import pathlib
import time
import urllib.request

DAILY = ["DGS1MO", "DGS3MO", "DGS6MO", "DGS1", "DGS2", "DGS3",
         "DGS5", "DGS7", "DGS10", "DGS20", "DGS30"]
MONTHLY = ["GS10", "TB3MS", "USREC"]

OUT = pathlib.Path(__file__).resolve().parent.parent / "data" / "raw"


def fetch(series: str) -> None:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                body = r.read()
            break
        except Exception as e:  # network hiccup: back off and retry
            print(f"{series}: attempt {attempt + 1} failed ({e})")
            time.sleep(2 ** attempt)
    else:
        raise SystemExit(f"could not download {series}")
    (OUT / f"{series}.csv").write_bytes(body)
    rows = body.count(b"\n") - 1
    print(f"{series}: {rows} rows")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for s in DAILY + MONTHLY:
        fetch(s)
