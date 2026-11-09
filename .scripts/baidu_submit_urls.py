#!/usr/bin/env python3
"""Submit site URLs to Baidu's active push API, within daily quota.

Baidu limits:
  - max 2000 URLs per request
  - daily quota varies by site (response field "remain")
  - re-pushing old URLs wastes quota and can lower your limit

With ~10/day, never-submitted URLs are ranked before push:
  0 home /teach  → 1 region hubs  → 2 faq/uni index
  → 3 uni hubs (more courses first)  → 4 course pages
  → 5 Chinese posts  → 6 other posts  → 7 glossaries  → 8 rest

Usage:
  export BAIDU_TOKEN=your_token   # or put it in .scripts/baidu.env
  python3 .scripts/baidu_submit_urls.py
  python3 .scripts/baidu_submit_urls.py --dry-run
  python3 .scripts/baidu_submit_urls.py --limit 50
  python3 .scripts/baidu_submit_urls.py --install-cron
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

SITE = "https://zlu.me"
API_BASE = "http://data.zz.baidu.com/urls"
SITEMAP_URL = f"{SITE}/sitemap.xml"
MAX_PER_REQUEST = 2000
# Baidu adjusts daily quota per site; zlu.me currently gets ~10/day (remain→0).
DEFAULT_DAILY_LIMIT = 10

# Regions Chinese parents search most often on Baidu (submit before other regions).
PRIORITY_REGIONS = frozenset(
    {
        "china",
        "hong-kong",
        "singapore",
        "australia",
        "uk",
        "usa",
        "malaysia",
    }
)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
ENV_FILE = SCRIPT_DIR / "baidu.env"
STATE_FILE = SCRIPT_DIR / "baidu_submit_state.json"
LOG_FILE = SCRIPT_DIR / "baidu_submit.log"


def load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_state() -> dict:
    if not STATE_FILE.is_file():
        return {"submitted": {}}
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"submitted": {}}
    data.setdefault("submitted", {})
    return data


def save_state(state: dict) -> None:
    STATE_FILE.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def fetch_sitemap_urls(sitemap_url: str) -> list[tuple[str, str]]:
    """Return list of (url, lastmod) from sitemap (or sitemap index)."""
    # Cloudflare rejects Python's default User-Agent with 403.
    req = urllib.request.Request(
        sitemap_url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (compatible; zlu-baidu-submit/1.1; +https://zlu.me)"
            ),
            "Accept": "application/xml,text/xml,*/*",
        },
    )
    xml_bytes = urllib.request.urlopen(req, timeout=60).read()
    root = ET.fromstring(xml_bytes)
    # Handle default xmlns by ignoring namespaces
    tag = lambda el: el.tag.split("}")[-1] if "}" in el.tag else el.tag

    if tag(root) == "sitemapindex":
        urls: list[tuple[str, str]] = []
        for sm in root:
            if tag(sm) != "sitemap":
                continue
            loc = next((c.text.strip() for c in sm if tag(c) == "loc" and c.text), None)
            if loc:
                urls.extend(fetch_sitemap_urls(loc))
        return urls

    results: list[tuple[str, str]] = []
    for url_el in root:
        if tag(url_el) != "url":
            continue
        loc = None
        lastmod = ""
        for child in url_el:
            name = tag(child)
            if name == "loc" and child.text:
                loc = child.text.strip()
            elif name == "lastmod" and child.text:
                lastmod = child.text.strip()
        if loc and loc.startswith(SITE):
            results.append((loc, lastmod))
    return results


def path_of(url: str) -> str:
    return urllib.parse.urlparse(url).path or "/"


def course_counts_from_sitemap(sitemap: list[tuple[str, str]]) -> dict[str, int]:
    """Map university slug → course page count from /teach/courses/{slug}/…"""
    counts: dict[str, int] = {}
    for url, _ in sitemap:
        m = re.match(r"^/teach/courses/([^/]+)/", path_of(url))
        if m:
            slug = m.group(1)
            counts[slug] = counts.get(slug, 0) + 1
    return counts


def lastmod_ord(lastmod: str) -> float:
    if not lastmod:
        return 0.0
    text = lastmod.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return 0.0


def url_priority(url: str, course_counts: dict[str, int]) -> tuple:
    """Lower tuple = submit sooner. Spend the ~10/day quota on Baidu money pages."""
    path = path_of(url).rstrip("/") or "/"

    if path in ("/", "/teach"):
        return (0, 0, path)

    # /teach/{region} hubs (not universities/courses/faq indexes)
    m = re.fullmatch(r"/teach/([^/]+)", path)
    if m and m.group(1) not in ("universities", "courses", "faq"):
        region = m.group(1)
        return (1, 0 if region in PRIORITY_REGIONS else 1, region)

    if path in ("/teach/universities", "/teach/faq"):
        return (2, 0, path)

    # University hubs: denser catalogs first (more course spokes = more intent).
    m = re.fullmatch(r"/teach/universities/([^/]+)", path)
    if m:
        slug = m.group(1)
        return (3, -course_counts.get(slug, 0), slug)

    # Course pages under the same denser universities first.
    m = re.match(r"^/teach/courses/([^/]+)/", path)
    if m:
        slug = m.group(1)
        return (4, -course_counts.get(slug, 0), path)

    # Chinese-language notes (Baidu can read these; English blog is lower).
    if "/zh/" in path or path.endswith("-zh.html") or "chinese" in path.lower():
        return (5, 0, path)

    if re.match(r"^/\d{4}/\d{2}/\d{2}/", path):
        return (6, 0, path)

    if path.startswith(("/ai-glossary", "/cs-glossary", "/med-glossary", "/algos")):
        return (7, 0, path)

    return (8, 0, path)


def pick_urls(
    sitemap: list[tuple[str, str]],
    state: dict,
    limit: int,
    resubmit_after_days: int,
) -> list[str]:
    submitted: dict = state["submitted"]
    now = datetime.now(timezone.utc)
    course_counts = course_counts_from_sitemap(sitemap)

    fresh: list[tuple[str, str]] = []
    stale: list[tuple[str, str]] = []

    for url, lastmod in sitemap:
        info = submitted.get(url)
        if not info:
            fresh.append((url, lastmod))
            continue
        try:
            last = datetime.strptime(info["at"], "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=timezone.utc
            )
        except (KeyError, ValueError):
            fresh.append((url, lastmod))
            continue
        age_days = (now - last).total_seconds() / 86400
        if age_days >= resubmit_after_days:
            stale.append((url, lastmod))

    # Never-submitted first; within each bucket: teach money pages, then newer lastmod.
    def sort_key(item: tuple[str, str]) -> tuple:
        url, lastmod = item
        return (url_priority(url, course_counts), -lastmod_ord(lastmod), url)

    fresh.sort(key=sort_key)
    stale.sort(key=sort_key)
    chosen = [u for u, _ in fresh + stale]
    return chosen[:limit]


def post_urls(api_url: str, urls: list[str]) -> dict:
    body = "\n".join(urls).encode("utf-8")
    req = urllib.request.Request(
        api_url,
        data=body,
        method="POST",
        headers={"Content-Type": "text/plain"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return json.loads(raw)
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"error": e.code, "message": raw or str(e)}
    except urllib.error.URLError as e:
        return {"error": "network", "message": str(e.reason)}


def install_cron(hour: int, minute: int) -> None:
    python = sys.executable
    script = SCRIPT_DIR / "baidu_submit_urls.py"
    # Load env file inside the job; redirect output to log
    # PYTHONUNBUFFERED so cron log gets POST lines even if the job is killed.
    job = (
        f"{minute} {hour} * * * "
        f"cd {REPO_ROOT} && "
        f"PYTHONUNBUFFERED=1 {python} {script} >> {LOG_FILE} 2>&1"
    )
    try:
        current = subprocess.check_output(["crontab", "-l"], text=True, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        current = ""

    marker = "baidu_submit_urls.py"
    lines = [ln for ln in current.splitlines() if marker not in ln]
    # Drop trailing empty lines then append
    while lines and not lines[-1].strip():
        lines.pop()
    lines.append(job)
    new_crontab = "\n".join(lines) + "\n"
    proc = subprocess.run(["crontab", "-"], input=new_crontab, text=True, capture_output=True)
    if proc.returncode != 0:
        print(proc.stderr or "Failed to install crontab", file=sys.stderr)
        sys.exit(1)
    print(f"Installed daily cron at {hour:02d}:{minute:02d} local time:")
    print(f"  {job}")
    print(f"Ensure {ENV_FILE} contains BAIDU_TOKEN=...")
    if not ENV_FILE.is_file():
        ENV_FILE.write_text(
            "# Baidu Ziyuan active-push token (do not commit)\n"
            "BAIDU_TOKEN=\n"
            f"BAIDU_SITE={SITE}\n",
            encoding="utf-8",
        )
        print(f"Created {ENV_FILE} — fill in BAIDU_TOKEN then re-run is not needed.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Submit zlu.me URLs to Baidu")
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_DAILY_LIMIT,
        help=f"Max URLs to submit this run (default {DEFAULT_DAILY_LIMIT})",
    )
    parser.add_argument(
        "--resubmit-after-days",
        type=int,
        default=30,
        help="Re-queue already-submitted URLs after N days (default 30)",
    )
    parser.add_argument(
        "--sitemap",
        default=SITEMAP_URL,
        help=f"Sitemap URL (default {SITEMAP_URL})",
    )
    parser.add_argument("--dry-run", action="store_true", help="List URLs only")
    parser.add_argument(
        "--plain",
        action="store_true",
        help="With --dry-run: print bare URLs (for Baidu 手动提交 paste)",
    )
    parser.add_argument(
        "--mark-submitted",
        action="store_true",
        help="Record the ranked batch (or --urls-file) as submitted without calling API "
        "(use after Baidu 手动提交)",
    )
    parser.add_argument(
        "--urls-file",
        help="URL list file (one per line). Used with --mark-submitted",
    )
    parser.add_argument(
        "--install-cron",
        action="store_true",
        help="Install a daily launchd/crontab job and exit",
    )
    parser.add_argument("--cron-hour", type=int, default=9, help="Cron hour (local)")
    parser.add_argument("--cron-minute", type=int, default=0, help="Cron minute")
    parser.add_argument("--reset-state", action="store_true", help="Clear submit history")
    args = parser.parse_args()

    load_env_file(ENV_FILE)

    if args.install_cron:
        install_cron(args.cron_hour, args.cron_minute)
        return 0

    if args.reset_state and STATE_FILE.is_file():
        STATE_FILE.unlink()
        print(f"Cleared {STATE_FILE}")

    token = os.environ.get("BAIDU_TOKEN", "").strip()
    site = os.environ.get("BAIDU_SITE", SITE).strip() or SITE
    if not args.dry_run and not args.mark_submitted and not token:
        print(
            f"Missing BAIDU_TOKEN. Set env or add it to {ENV_FILE}",
            file=sys.stderr,
        )
        return 1

    state = load_state()
    limit = max(1, min(args.limit, MAX_PER_REQUEST))

    if args.urls_file and args.mark_submitted:
        path = Path(args.urls_file)
        batch = [
            ln.strip()
            for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ][:limit]
        unique: list[tuple[str, str]] = []
    else:
        status = sys.stderr if args.plain else sys.stdout
        print(f"[{utc_now()}] Fetching sitemap: {args.sitemap}", file=status)
        try:
            sitemap = fetch_sitemap_urls(args.sitemap)
        except Exception as e:
            print(f"Failed to fetch sitemap: {e}", file=sys.stderr)
            return 1

        # Dedupe preserving order
        seen = set()
        unique = []
        for url, lastmod in sitemap:
            if url not in seen:
                seen.add(url)
                unique.append((url, lastmod))

        batch = pick_urls(unique, state, limit, args.resubmit_after_days)
        never = sum(1 for u, _ in unique if u not in state["submitted"])
        print(
            f"Sitemap URLs: {len(unique)} | never submitted: {never} | "
            f"this batch: {len(batch)}",
            file=status,
        )

    if not batch:
        print("Nothing to submit.")
        return 0

    if args.dry_run:
        if args.plain or not unique:
            for u in batch:
                print(u)
        else:
            course_counts = course_counts_from_sitemap(unique)
            for u in batch:
                tier = url_priority(u, course_counts)[0]
                print(f"[{tier}] {u}")
        return 0

    if args.mark_submitted:
        at = utc_now()
        for url in batch:
            state["submitted"][url] = {"at": at, "via": "manual"}
        state["last_run"] = {
            "at": at,
            "requested": len(batch),
            "response": {"manual": True, "marked": len(batch)},
        }
        save_state(state)
        print(f"Marked {len(batch)} URL(s) as manually submitted in {STATE_FILE.name}")
        for u in batch:
            print(u)
        return 0

    api = f"{API_BASE}?site={site}&token={token}"
    # Mask token in logs
    print(f"POSTing {len(batch)} URL(s) to Baidu…")
    result = post_urls(api, batch)
    print(json.dumps(result, ensure_ascii=False))

    if result.get("message") == "over quota" or result.get("error") == 400:
        print(
            "Baidu API over quota. Use 手动提交 in the console, then:\n"
            f"  python3 .scripts/baidu_submit_urls.py --mark-submitted --limit {len(batch)}",
            file=sys.stderr,
        )
        return 2

    success = int(result.get("success", 0) or 0)
    if success > 0 or result.get("remain") is not None:
        at = utc_now()
        # Mark only as many as reported success; if success missing but no error, mark all
        mark_count = success if success else (0 if result.get("error") else len(batch))
        for url in batch[:mark_count]:
            state["submitted"][url] = {"at": at}
        state["last_run"] = {
            "at": at,
            "requested": len(batch),
            "response": result,
        }
        save_state(state)
        print(f"Recorded {mark_count} URL(s) in {STATE_FILE.name}")
        if result.get("remain") is not None:
            print(f"Daily remain after push: {result['remain']}")
        return 0

    print("Submission failed; state not updated.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
