"""Inline data.js and engine.js into one self-contained HTML file (dist/)."""
import pathlib
import re

WEB = pathlib.Path(__file__).resolve().parent.parent / "web"
html = (WEB / "index.html").read_text()
inline = "".join(f"<script>\n{(WEB / n).read_text()}</script>\n" for n in ("data.js", "engine.js"))
html = re.sub(r"<!--DATA-->.*?<!--/DATA-->\n", lambda m: inline, html, flags=re.S)
out = WEB.parent / "dist" / "crash-replay.html"
out.parent.mkdir(exist_ok=True)
out.write_text(html)
print(out, out.stat().st_size // 1024, "KB")
