#!/usr/bin/env python3
"""Submit site URLs to IndexNow (Bing, Yandex, Seznam, Naver, etc.).

Ownership is proven by hosting {KEY}.txt at the site root containing the key.
See https://www.indexnow.org/documentation

Usage:
  python3 .scripts/indexnow_submit_urls.py
  python3 .scripts/indexnow_submit_urls.py --dry-run
  python3 .scripts/indexnow_submit_urls.py --limit 100
  python3 .scripts/indexnow_submit_urls.py --url https://zlu.me/path/
  python3 .scripts/indexnow_submit_urls.py --urls-file urls.txt
  python3 .scripts/indexnow_submit_urls.py --ensure-key-file
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

SITE = "https://zlu.me"
HOST = "zlu.me"
SITEMAP_URL = f"{SITE}/sitemap.xml"
ENDPOINT = "https://api.indexnow.org/indexnow"
MAX_PER_REQUEST = 10000
# Bing prefers streaming changed URLs over large daily sitemap batches.
DEFAULT_LIMIT = 20

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
ENV_FILE = SCRIPT_DIR / "indexnow.env"
STATE_FILE = SCRIPT_DIR / "indexnow_submit_state.json"
LOG_FILE = SCRIPT_DIR / "indexnow_submit.log"
USER_AGENT = "zlu-indexnow/1.0 (+https://zlu.me)"


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


def discover_key_from_repo() -> str | None:
    """Find a root {key}.txt whose contents match the filename stem."""
    for path in sorted(REPO_ROOT.glob("*.txt")):
        if path.name in {"robots.txt", "CNAME"}:
            continue
        stem = path.stem
        if not (8 <= len(stem) <= 128):
            continue
        try:
            body = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if body == stem:
            return stem
    return None


def resolve_key() -> str:
    key = os.environ.get("INDEXNOW_KEY", "").strip()
    if key:
        return key
    found = discover_key_from_repo()
    if found:
        return found
    raise SystemExit(
        f"Missing INDEXNOW_KEY. Set env, add it to {ENV_FILE}, "
        f"or place {{key}}.txt at the site root."
    )


def ensure_key_file(key: str) -> Path:
    path = REPO_ROOT / f"{key}.txt"
    path.write_text(key + "\n", encoding="utf-8")
    return path


def parse_sitemap_xml(xml_bytes: bytes, allow_localhost: bool = False) -> list[tuple[str, str]]:
    root = ET.fromstring(xml_bytes)
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
        if not loc:
            continue
        # Rewrite local build URLs to production before submit.
        if "localhost" in loc or loc.startswith("http://127."):
            if not allow_localhost:
                loc = re.sub(r"^https?://[^/]+", SITE, loc)
            else:
                continue
        if loc.startswith(SITE) or loc.startswith(f"https://www.{HOST}"):
            results.append((loc, lastmod))
    return results


def fetch_sitemap_urls(sitemap_url: str) -> list[tuple[str, str]]:
    # Cloudflare (and similar) reject Python's default User-Agent with 403.
    req = urllib.request.Request(
        sitemap_url,
        headers={"User-Agent": USER_AGENT},
    )
    xml_bytes = urllib.request.urlopen(req, timeout=60).read()
    return parse_sitemap_xml(xml_bytes)


def load_sitemap_file(path: Path) -> list[tuple[str, str]]:
    return parse_sitemap_xml(path.read_bytes())


def pick_urls(
    sitemap: list[tuple[str, str]],
    state: dict,
    limit: int,
    resubmit_after_days: int,
) -> list[str]:
    submitted: dict = state["submitted"]
    now = datetime.now(timezone.utc)

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

    def sort_key(item: tuple[str, str]) -> tuple:
        url, lastmod = item
        return (0 if lastmod else 1, lastmod, url)

    fresh.sort(key=sort_key, reverse=True)
    stale.sort(key=sort_key, reverse=True)
    return [u for u, _ in fresh + stale][:limit]


def normalize_url(url: str) -> str:
    url = url.strip()
    if not url:
        return url
    if url.startswith("/"):
        return SITE.rstrip("/") + url
    return url


def post_indexnow(host: str, key: str, urls: list[str]) -> tuple[int, str]:
    key_location = f"https://{host}/{key}.txt"
    payload = {
        "host": host,
        "key": key,
        "keyLocation": key_location,
        "urlList": urls,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        ENDPOINT,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return resp.status, raw
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        return e.code, raw or str(e)
    except urllib.error.URLError as e:
        return 0, str(e.reason)


def verify_key_live(host: str, key: str) -> tuple[bool, str]:
    url = f"https://{host}/{key}.txt"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8", errors="replace").strip()
            if body == key:
                return True, f"OK {url}"
            return False, f"Key file body mismatch at {url}"
    except Exception as e:
        return False, f"Key file not reachable at {url}: {e}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Submit zlu.me URLs to IndexNow")
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Max URLs from sitemap this run (default {DEFAULT_LIMIT})",
    )
    parser.add_argument(
        "--resubmit-after-days",
        type=int,
        default=30,
        help="Re-queue already-submitted URLs after N days (default 30)",
    )
    parser.add_argument("--sitemap", default=SITEMAP_URL, help="Sitemap URL")
    parser.add_argument(
        "--sitemap-file",
        help="Local sitemap.xml path (e.g. _site/sitemap.xml). Rewrites localhost → https://zlu.me.",
    )
    parser.add_argument(
        "--url",
        action="append",
        default=[],
        help="Submit this URL (repeatable). Skips sitemap selection.",
    )
    parser.add_argument(
        "--urls-file",
        help="File with one URL per line. Skips sitemap selection.",
    )
    parser.add_argument("--dry-run", action="store_true", help="List URLs only")
    parser.add_argument(
        "--ensure-key-file",
        action="store_true",
        help="Write/overwrite root {key}.txt from INDEXNOW_KEY and exit",
    )
    parser.add_argument(
        "--skip-key-check",
        action="store_true",
        help="Do not verify the live key file before submitting",
    )
    parser.add_argument("--reset-state", action="store_true", help="Clear submit history")
    args = parser.parse_args()

    load_env_file(ENV_FILE)
    host = os.environ.get("INDEXNOW_HOST", HOST).strip() or HOST

    if args.ensure_key_file:
        key = os.environ.get("INDEXNOW_KEY", "").strip() or discover_key_from_repo()
        if not key:
            print("Need INDEXNOW_KEY to write the key file.", file=sys.stderr)
            return 1
        path = ensure_key_file(key)
        print(f"Wrote {path} — deploy so {SITE}/{key}.txt is public.")
        return 0

    try:
        key = resolve_key()
    except SystemExit as e:
        print(e, file=sys.stderr)
        return 1

    key_path = REPO_ROOT / f"{key}.txt"
    if not key_path.is_file():
        ensure_key_file(key)
        print(f"Created missing key file: {key_path}")

    if args.reset_state and STATE_FILE.is_file():
        STATE_FILE.unlink()
        print(f"Cleared {STATE_FILE}")

    batch: list[str] = []
    if args.url or args.urls_file:
        for u in args.url:
            nu = normalize_url(u)
            if nu:
                batch.append(nu)
        if args.urls_file:
            for line in Path(args.urls_file).read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    batch.append(normalize_url(line))
        seen_urls: set[str] = set()
        deduped: list[str] = []
        for u in batch:
            if u not in seen_urls:
                seen_urls.add(u)
                deduped.append(u)
        batch = deduped
    else:
        try:
            if args.sitemap_file:
                path = Path(args.sitemap_file)
                print(f"[{utc_now()}] Loading sitemap file: {path}")
                sitemap = load_sitemap_file(path)
            else:
                print(f"[{utc_now()}] Fetching sitemap: {args.sitemap}")
                sitemap = fetch_sitemap_urls(args.sitemap)
        except Exception as e:
            print(f"Failed to load sitemap: {e}", file=sys.stderr)
            return 1

        seen = set()
        unique: list[tuple[str, str]] = []
        for url, lastmod in sitemap:
            if url not in seen:
                seen.add(url)
                unique.append((url, lastmod))

        state = load_state()
        limit = max(1, min(args.limit, MAX_PER_REQUEST))
        batch = pick_urls(unique, state, limit, args.resubmit_after_days)
        never = sum(1 for u, _ in unique if u not in state["submitted"])
        print(
            f"Sitemap URLs: {len(unique)} | never submitted: {never} | "
            f"this batch: {len(batch)}"
        )

    if not batch:
        print("Nothing to submit.")
        return 0

    if args.dry_run:
        for u in batch:
            print(u)
        return 0

    if not args.skip_key_check:
        ok, msg = verify_key_live(host, key)
        print(msg)
        if not ok:
            print(
                "Deploy the key file (and site) first, then re-run. "
                "Or pass --skip-key-check to force (may return 202/403).",
                file=sys.stderr,
            )
            return 1

    print(f"POSTing {len(batch)} URL(s) to IndexNow ({ENDPOINT})…")
    status, raw = post_indexnow(host, key, batch)
    print(f"HTTP {status}")
    if raw:
        print(raw)

    # 200 = OK, 202 = accepted (key validation may still be pending)
    if status in (200, 202):
        state = load_state()
        at = utc_now()
        for url in batch:
            state["submitted"][url] = {"at": at}
        state["last_run"] = {
            "at": at,
            "requested": len(batch),
            "status": status,
            "response": raw,
        }
        save_state(state)
        print(f"Recorded {len(batch)} URL(s) in {STATE_FILE.name}")
        return 0

    print("Submission failed; state not updated.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
