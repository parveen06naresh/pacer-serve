"""Inline data.js and engine.js into one self-contained HTML file.

  python scripts/bundle.py                      -> dist/crash-replay.html (page body only)
  python scripts/bundle.py --standalone URL     -> dist/standalone/index.html, a full
                                                   document with link-preview tags, hosted at URL
"""
import pathlib
import re
import sys

WEB = pathlib.Path(__file__).resolve().parent.parent / "web"
DIST = WEB.parent / "dist"

html = (WEB / "index.html").read_text()
inline = "".join(f"<script>\n{(WEB / n).read_text()}</script>\n" for n in ("data.js", "engine.js"))
html = re.sub(r"<!--DATA-->.*?<!--/DATA-->\n", lambda m: inline, html, flags=re.S)

if len(sys.argv) > 2 and sys.argv[1] == "--standalone":
    url = sys.argv[2].rstrip("/")
    html = re.sub(r'const SHARE_URL = "[^"]*";', f'const SHARE_URL = "{url}";', html)
    desc = ("Relive the biggest stock market crashes with $10,000 and real prices. "
            "Sell, hold or go all in, then see if you beat the market.")
    head = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta property="og:type" content="website">
<meta property="og:title" content="Crash Replay: would you have sold in 2008?">
<meta property="og:description" content="{desc}">
<meta property="og:url" content="{url}">
<meta property="og:image" content="{url}/og.png">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="Crash Replay: would you have sold in 2008?">
<meta name="twitter:description" content="{desc}">
<meta name="twitter:image" content="{url}/og.png">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 28 28'%3E%3Crect width='28' height='28' rx='8' fill='%230b1220'/%3E%3Cpath d='M5 9 L10 12 L13 10 L17 19 L20 16 L23 18' fill='none' stroke='white' stroke-width='2.4' stroke-linecap='round' stroke-linejoin='round'/%3E%3Ccircle cx='23' cy='18' r='2.2' fill='%23e0353a'/%3E%3C/svg%3E">
<style>body{{margin:0}}</style>
</head>
<body>
"""
    out = DIST / "standalone" / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(head + html + "</body>\n</html>\n")
else:
    out = DIST / "crash-replay.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(html)
print(out, out.stat().st_size // 1024, "KB")
