"""Bundle a run's artifacts into a single self-contained ``report.html``.

Walks a directory and embeds:
 - manifest.json + co-evolve summary (rendered as a header block)
 - all ``plots/*.png`` (and any image at any depth) inline (base64)
 - all ``*.mp4`` inline (base64) so the report is one shareable file
 - ``metrics.csv`` (head) and ``tournament.csv`` if present

Usage:

    python -m train.tools.report --dir artifacts/<run> --out report.html
"""
from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import os
from typing import List, Tuple


def _b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def _walk(root: str, exts: Tuple[str, ...]) -> List[str]:
    out = []
    for dp, _dn, fn in os.walk(root):
        for f in fn:
            if f.lower().endswith(exts):
                out.append(os.path.join(dp, f))
    return sorted(out)


def _csv_table(path: str, max_rows: int = 12) -> str:
    rows: List[List[str]] = []
    with open(path, "r") as f:
        r = csv.reader(f)
        for i, row in enumerate(r):
            rows.append(row)
            if i >= max_rows:
                break
    if not rows:
        return ""
    head, *body = rows
    th = "".join(f"<th>{html.escape(h)}</th>" for h in head)
    trs = []
    for row in body:
        trs.append("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in row) + "</tr>")
    return f"<table><thead><tr>{th}</tr></thead><tbody>{''.join(trs)}</tbody></table>"


def _json_block(path: str) -> str:
    try:
        data = json.load(open(path))
        return f"<pre>{html.escape(json.dumps(data, indent=2))}</pre>"
    except Exception as e:
        return f"<p>could not read {html.escape(path)}: {html.escape(str(e))}</p>"


def build_report(run_dir: str, out_path: str, title: str | None = None) -> str:
    run_dir = os.path.abspath(run_dir)
    title = title or f"digital-evo-rl report — {os.path.basename(run_dir)}"
    sections: List[str] = []

    # Header / summaries.
    for fname in ("manifest.json", "coevolve_summary.json", "tournament_summary.json", "evolve_summary.json", "eval_summary.json"):
        path = os.path.join(run_dir, fname)
        if os.path.exists(path):
            sections.append(f"<section><h2>{html.escape(fname)}</h2>{_json_block(path)}</section>")

    # CSVs.
    for fname in ("metrics.csv", "tournament.csv", "eval.csv", "sweep_summary.csv"):
        path = os.path.join(run_dir, fname)
        if os.path.exists(path):
            sections.append(f"<section><h2>{html.escape(fname)} (first 12 rows)</h2>{_csv_table(path)}</section>")

    # Images.
    images = _walk(run_dir, (".png", ".jpg", ".jpeg"))
    if images:
        cards = []
        for img in images:
            rel = os.path.relpath(img, run_dir)
            cards.append(
                f"<figure><img src='data:image/png;base64,{_b64(img)}' alt='{html.escape(rel)}'/>"
                f"<figcaption>{html.escape(rel)}</figcaption></figure>"
            )
        sections.append(f"<section><h2>Plots ({len(images)})</h2><div class='grid'>{''.join(cards)}</div></section>")

    # Videos.
    videos = _walk(run_dir, (".mp4",))
    if videos:
        cards = []
        for v in videos:
            rel = os.path.relpath(v, run_dir)
            cards.append(
                f"<figure><video controls width='400' src='data:video/mp4;base64,{_b64(v)}'></video>"
                f"<figcaption>{html.escape(rel)}</figcaption></figure>"
            )
        sections.append(f"<section><h2>Videos ({len(videos)})</h2><div class='grid'>{''.join(cards)}</div></section>")

    html_doc = f"""<!doctype html>
<html lang='en'><head><meta charset='utf-8'><title>{html.escape(title)}</title>
<style>
 body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 1200px; margin: 2em auto; padding: 0 1em; color: #222; }}
 h1 {{ border-bottom: 2px solid #333; padding-bottom: 0.3em; }}
 h2 {{ margin-top: 2em; color: #1a4480; }}
 pre {{ background: #f5f5f5; padding: 1em; overflow-x: auto; font-size: 0.85em; }}
 table {{ border-collapse: collapse; font-size: 0.9em; }}
 th, td {{ border: 1px solid #ccc; padding: 4px 8px; }}
 th {{ background: #eef; }}
 .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(380px, 1fr)); gap: 1em; }}
 figure {{ margin: 0; border: 1px solid #ddd; padding: 0.5em; background: white; }}
 figure img, figure video {{ max-width: 100%; height: auto; display: block; }}
 figcaption {{ font-size: 0.8em; color: #555; margin-top: 0.3em; word-break: break-all; }}
 section {{ margin-bottom: 2em; }}
</style></head><body>
<h1>{html.escape(title)}</h1>
<p>Source: <code>{html.escape(run_dir)}</code></p>
{''.join(sections)}
</body></html>"""

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write(html_doc)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()
    path = build_report(args.dir, args.out, args.title)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
