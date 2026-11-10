#!/usr/bin/env python3
"""Build a Cloudflare Bulk Redirects CSV for GSC-hit retired teach uni shells.

Free plan: ≤10,000 URL redirects. We only include retired slugs that appear in a
GSC Pages CSV (and/or the removals list), expanded to hub + preview/assignment/
exam/sync — typically ~2–3k rows, not the full ~19k farm.

Usage:
  python3 .scripts/generate_teach_bulk_redirects.py \\
    --gsc-pages /path/to/Pages.csv

  # then import .scripts/teach-uni-bulk-redirects.csv in CF → Bulk redirects
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

try:
    import yaml
except ImportError as e:  # pragma: no cover
    raise SystemExit("PyYAML required: pip install pyyaml") from e

ROOT = Path(__file__).resolve().parents[1]
SITE = "https://zlu.me"
TARGET = f"{SITE}/teach/universities/"
SERVICES = ("preview", "assignment", "exam", "sync")
OUT_CSV = ROOT / ".scripts" / "teach-uni-bulk-redirects.csv"
OUT_META = ROOT / ".scripts" / "teach-uni-bulk-redirects.meta.txt"
REMOVALS = ROOT / ".scripts" / "gsc_removals_teach_dead_urls.txt"
LIMIT = 10000

UNI_RE = re.compile(
    r"^/teach/universities/([^/]+)(?:/(preview|assignment|exam|sync))?/?$"
)


def load_retired() -> set[str]:
    raw = yaml.safe_load((ROOT / "_data" / "teach_retired_slugs.yml").read_text())
    retired = {str(s).strip() for s in (raw or {}).get("slugs", []) if str(s).strip()}
    live = {p.stem for p in (ROOT / "_data" / "courses").glob("*.yml")}
    return retired - live


def path_of(url: str) -> str:
    return re.sub(r"^https?://[^/]+", "", url).rstrip("/") + "/"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--gsc-pages",
        type=Path,
        help="GSC Performance → Pages CSV export (Top pages, Impressions)",
    )
    ap.add_argument("--limit", type=int, default=LIMIT)
    args = ap.parse_args()

    retired = load_retired()
    slug_impr: dict[str, float] = {}

    if args.gsc_pages and args.gsc_pages.is_file():
        with args.gsc_pages.open(newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                url = row.get("Top pages") or row.get("Page") or row.get("Pages") or ""
                path = path_of(url)
                m = UNI_RE.match(path.rstrip("/"))
                if not m:
                    # path_of added trailing slash; match without relying on that
                    m = UNI_RE.match(path)
                if not m:
                    continue
                slug = m.group(1)
                if slug not in retired:
                    continue
                try:
                    impr = float(str(row.get("Impressions") or 0).replace(",", ""))
                except ValueError:
                    impr = 0.0
                slug_impr[slug] = max(slug_impr.get(slug, 0.0), impr)

    if REMOVALS.is_file():
        for line in REMOVALS.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line.startswith("http"):
                continue
            path = path_of(line)
            m = UNI_RE.match(path)
            if not m:
                continue
            slug = m.group(1)
            if slug in retired:
                slug_impr.setdefault(slug, 0.0)

    if not slug_impr:
        raise SystemExit(
            "No retired GSC/removal URLs found. Pass --gsc-pages Pages.csv "
            "and/or keep .scripts/gsc_removals_teach_dead_urls.txt."
        )

    ranked = sorted(slug_impr.items(), key=lambda kv: (-kv[1], kv[0]))
    paths: list[str] = []
    seen: set[str] = set()
    for slug, _impr in ranked:
        for p in [f"/teach/universities/{slug}/"] + [
            f"/teach/universities/{slug}/{s}/" for s in SERVICES
        ]:
            if p in seen:
                continue
            if len(paths) >= args.limit:
                break
            seen.add(p)
            paths.append(p)
        if len(paths) >= args.limit:
            break

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        for p in paths:
            f.write(f"{SITE}{p},{TARGET},301,,,,\n")

    with OUT_META.open("w", encoding="utf-8") as f:
        f.write(f"impressed_retired_slugs={len(slug_impr)}\n")
        f.write(f"redirect_rows={len(paths)}\n")
        f.write(f"target={TARGET}\n")
        f.write(f"limit={args.limit}\n")
        f.write("top_slugs:\n")
        for slug, impr in ranked[:30]:
            f.write(f"  {int(impr):5d}  {slug}\n")

    print(f"wrote {OUT_CSV.relative_to(ROOT)} ({len(paths)} rows from {len(slug_impr)} slugs)")
    print(f"meta  {OUT_META.relative_to(ROOT)}")
    print("Import CSV in Cloudflare dashboard → Bulk redirects (Free ≤10,000).")


if __name__ == "__main__":
    main()
