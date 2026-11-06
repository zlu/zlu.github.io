#!/usr/bin/env python3
"""Stream changed site URLs to IndexNow (Bing, Yandex, Seznam, Naver, etc.).

Bing flags sitemap-style daily dumps as "batch mode". Prefer streaming:
submit only URLs that were added/updated/deleted, one request at a time,
when the change happens (e.g. after publish / deploy).

Ownership is proven by hosting {KEY}.txt at the site root containing the key.
See https://www.indexnow.org/documentation

Usage:
  # Stream URLs changed in the last commit (default)
  python3 .scripts/indexnow_submit_urls.py
  python3 .scripts/indexnow_submit_urls.py --dry-run

  # Explicit URLs (best after publishing a post)
  python3 .scripts/indexnow_submit_urls.py --url https://zlu.me/2026/10/29/slug.html
  # After a data-only "sync course map" commit sitting on top of a post:
  python3 .scripts/indexnow_submit_urls.py --from-git HEAD~3..HEAD

  # Rare: sitemap-driven (still streams one URL per request)
  python3 .scripts/indexnow_submit_urls.py --from-sitemap --limit 5

  python3 .scripts/indexnow_submit_urls.py --ensure-key-file

Notes:
  - `_data/courses/{slug}.yml` maps to `/teach/universities/{slug}/` plus linked
    course pages from `_data/teach_catalog.yml` (run sync-teach-catalog first).
  - Catalog/report-only commits still yield 0 URLs; widen `--from-git` or pass `--url`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

SITE = "https://zlu.me"
HOST = "zlu.me"
SITEMAP_URL = f"{SITE}/sitemap.xml"
ENDPOINT = "https://api.indexnow.org/indexnow"
# Streaming default: one URL per request. Keep sitemap pulls tiny if used.
DEFAULT_LIMIT = 5
STREAM_DELAY_SEC = 0.35

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
ENV_FILE = SCRIPT_DIR / "indexnow.env"
STATE_FILE = SCRIPT_DIR / "indexnow_submit_state.json"
LOG_FILE = SCRIPT_DIR / "indexnow_submit.log"
USER_AGENT = "zlu-indexnow/1.0 (+https://zlu.me)"

POST_RE = re.compile(
    r"^_posts/(\d{4})-(\d{2})-(\d{2})-(.+)\.(md|markdown|html)$"
)
COLLECTION_DIRS = {
    "_ai-glossary": "ai-glossary",
    "_med-glossary": "med-glossary",
    "_algos": "algos",
    "_albums": "photos",
    "_archives": "archive",
}
# Shared / infra paths fan out to many pages — leave those to the sitemap crawl.
SKIP_PREFIXES = (
    ".github/",
    ".scripts/",
    ".cursor/",
    "admin/",
    "vendor/",
    "_includes/",
    "_layouts/",
    "_sass/",
    "_plugins/",
    "_data/",
    "assets/",
    "redirects/",
    "_drafts/",
    "_photos/",  # photos collection output: false
)


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


def parse_lastmod(raw: str) -> datetime | None:
    if not raw:
        return None
    raw = raw.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+0000"
    elif re.search(r"[+-]\d{2}:\d{2}$", raw):
        raw = raw[:-3] + raw[-2:]  # +00:00 → +0000
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M%z", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(raw, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            continue
    return None


def _sitemap_priority(url: str) -> int:
    """Lower = more important when lastmods collide after a full rebuild."""
    path = urllib.parse.urlparse(url).path or "/"
    if re.match(r"^/\d{4}/\d{2}/\d{2}/", path):
        return 0  # blog posts
    if path.startswith("/teach/courses/") or path.startswith("/teach/universities/"):
        return 1
    if path.startswith("/ai-glossary/") or path.startswith("/algos/"):
        return 2
    if path.startswith("/teach/"):
        return 3
    return 4


def pick_changed_sitemap_urls(
    sitemap: list[tuple[str, str]],
    state: dict,
    limit: int,
    since: datetime | None,
    *,
    only_new: bool = True,
) -> list[str]:
    """Pick sitemap URLs worth streaming.

    Default: **never-submitted** URLs (diff vs indexnow_submit_state.json).
    Jekyll often stamps the same lastmod on half the site after deploy, so
    lastmod-alone + --limit is a lottery and can miss the new post.
    """
    submitted: dict = state["submitted"]
    candidates: list[tuple[str, datetime | None, bool]] = []

    for url, lastmod in sitemap:
        lm = parse_lastmod(lastmod)
        is_new = url not in submitted
        if only_new:
            if not is_new:
                continue
            # New sitemap URLs always qualify; lastmod is only a sort signal.
        else:
            if since is not None:
                if lm is None or lm < since:
                    continue
            elif not is_new:
                continue
        candidates.append((url, lm, is_new))

    def sort_key(item: tuple[str, datetime | None, bool]) -> tuple:
        url, lm, is_new = item
        recent = 0
        if since is not None and lm is not None and lm >= since:
            recent = 0
        elif lm is not None:
            recent = 1
        else:
            recent = 2
        lm_ord = lm.timestamp() if lm else 0.0
        return (0 if is_new else 1, recent, -lm_ord, _sitemap_priority(url), url)

    candidates.sort(key=sort_key)
    return [u for u, *_ in candidates][:limit]


def normalize_url(url: str) -> str:
    url = url.strip()
    if not url:
        return url
    if url.startswith("/"):
        return SITE.rstrip("/") + url
    return url


def read_front_matter(path: Path) -> dict[str, str]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end < 0:
        return {}
    meta: dict[str, str] = {}
    for line in text[3:end].splitlines():
        if ":" not in line or line.startswith(" ") or line.startswith("-"):
            continue
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip().strip("'\"")
    return meta


def is_indexnow_excluded(meta: dict[str, str]) -> bool:
    if meta.get("sitemap", "").lower() == "false":
        return True
    robots = meta.get("robots", "").lower()
    return "noindex" in robots


COURSE_YML_RE = re.compile(r"^_data/courses/([^/]+)\.ya?ml$")
LINKED_HOW = frozenset(
    {"material field", "material alias", "code match", "title fuzzy"}
)


def ascii_slug(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return re.sub(r"^-+|-+$", "", text)


def course_page_slug(code: str, title: str) -> str:
    """Match teach_pages.rb course_slug (code preferred, else title)."""
    base = (code or "").strip() or (title or "").strip()
    slug = ascii_slug(base)
    if slug:
        return slug
    import hashlib

    digest = hashlib.md5(base.encode("utf-8")).hexdigest()[:8]
    return f"course-{digest}"


def load_teach_catalog_by_slug() -> dict[str, dict]:
    path = REPO_ROOT / "_data" / "teach_catalog.yml"
    if not path.is_file():
        return {}
    try:
        import yaml  # type: ignore
    except ImportError:
        return {}
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    unis = []
    if isinstance(raw, dict):
        unis = list(raw.get("universities") or [])
    elif isinstance(raw, list):
        unis = raw
    out: dict[str, dict] = {}
    for entry in unis:
        if not isinstance(entry, dict):
            continue
        slug = ascii_slug(str(entry.get("slug") or ""))
        if slug:
            out[slug] = entry
    return out


def teach_urls_for_course_yml(rel: str, catalog: dict[str, dict]) -> list[str]:
    """Expand `_data/courses/{slug}.yml` → indexable teach URLs."""
    m = COURSE_YML_RE.match(rel)
    if not m:
        return []
    slug = ascii_slug(m.group(1))
    if not slug:
        return []
    entry = catalog.get(slug)
    urls = [f"{SITE}/teach/universities/{slug}/"]
    if not entry:
        # Uni hub may still exist from admin YAML even before catalog refresh.
        return urls
    linked_n = int(entry.get("linked_course_count") or 0)
    if linked_n <= 0:
        # Uni hub is noindex when nothing is linked — skip spam.
        return []
    for course in entry.get("courses") or []:
        if not isinstance(course, dict):
            continue
        how = str(course.get("link") or "")
        if how not in LINKED_HOW or int(course.get("depth_score") or 0) <= 0:
            continue
        cslug = course_page_slug(
            str(course.get("code") or ""), str(course.get("title") or "")
        )
        if cslug:
            urls.append(f"{SITE}/teach/courses/{slug}/{cslug}/")
    return urls


def path_to_urls(rel: str, catalog: dict[str, dict] | None = None) -> list[str]:
    """Map a repo-relative content path to public URL(s), or [] to skip."""
    rel = rel.replace("\\", "/").lstrip("./")
    if not rel or rel.endswith("/"):
        return []

    # Course admin YAML → generated teach pages (exception to SKIP_PREFIXES).
    if COURSE_YML_RE.match(rel):
        return teach_urls_for_course_yml(rel, catalog or load_teach_catalog_by_slug())

    if any(rel.startswith(p) for p in SKIP_PREFIXES):
        return []
    if rel in {".gitignore", "CNAME", "robots.txt", "Gemfile", "Gemfile.lock", "README.md"}:
        return []
    if not rel.endswith((".md", ".markdown", ".html", ".htm")):
        return []

    src = REPO_ROOT / rel
    meta = read_front_matter(src) if src.is_file() else {}
    if is_indexnow_excluded(meta):
        return []

    m = POST_RE.match(rel)
    if m:
        y, mo, d, slug, _ext = m.groups()
        return [f"{SITE}/{y}/{mo}/{d}/{slug}.html"]

    for prefix, collection in COLLECTION_DIRS.items():
        if rel.startswith(prefix + "/"):
            name = Path(rel).stem
            if name.startswith("_"):
                return []
            return [f"{SITE}/{collection}/{name}/"]

    if rel.startswith("_tags/"):
        # Thin tag indexes: sitemap: false — don't IndexNow-spam them.
        return []

    if "/" not in rel:
        permalink = meta.get("permalink")
        if permalink:
            if not permalink.startswith("/"):
                permalink = "/" + permalink
            return [SITE.rstrip("/") + permalink]
        stem = Path(rel).stem
        if stem in {"index", "home"}:
            return [f"{SITE}/"]
        return [f"{SITE}/{stem}/"]

    return []


def path_to_url(rel: str) -> str | None:
    """Back-compat: first URL for a path, or None."""
    urls = path_to_urls(rel)
    return urls[0] if urls else None


def git_changed_paths(revision_range: str) -> list[str]:
    cmd = [
        "git",
        "-C",
        str(REPO_ROOT),
        "-c",
        "core.quotepath=false",
        "diff",
        "-z",
        "--name-only",
        "--diff-filter=ACMR",
        revision_range,
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode("utf-8", errors="replace").strip() if e.stderr else str(e)
        raise SystemExit(f"git diff failed for {revision_range!r}: {err}") from e
    return [p for p in out.decode("utf-8", errors="replace").split("\0") if p]


def urls_from_git(revision_range: str) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()
    skipped: list[str] = []
    catalog = load_teach_catalog_by_slug()
    for path in git_changed_paths(revision_range):
        mapped = path_to_urls(path, catalog)
        if not mapped:
            skipped.append(path)
            continue
        for url in mapped:
            if url not in seen:
                seen.add(url)
                urls.append(url)
    if skipped:
        print(f"Skipped {len(skipped)} non-page path(s) (shared/infra/assets).")
    if not urls and skipped:
        data_only = all(
            p.startswith("_data/") or p in {"README.md"} for p in skipped
        )
        if data_only:
            print(
                "Hint: last commit looks data/report-only. "
                "Widen the range, e.g. --from-git HEAD~3..HEAD, "
                "or pass --url for the post."
            )
        course_ymls = [p for p in skipped if COURSE_YML_RE.match(p)]
        if course_ymls:
            print(
                "Hint: course YAML changed but no linked materials in teach_catalog.yml. "
                "Run: python3 .scripts/sync-teach-catalog.py && retry."
            )
    return urls


def stream_indexnow(host: str, key: str, url: str) -> tuple[int, str]:
    """Submit one URL via GET (IndexNow streaming)."""
    key_location = f"https://{host}/{key}.txt"
    qs = urllib.parse.urlencode(
        {
            "url": url,
            "key": key,
            "keyLocation": key_location,
        }
    )
    req = urllib.request.Request(
        f"{ENDPOINT}?{qs}",
        method="GET",
        headers={"User-Agent": USER_AGENT},
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


def dedupe(urls: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for u in urls:
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stream changed zlu.me URLs to IndexNow (avoid sitemap batch dumps)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Max URLs this run (default {DEFAULT_LIMIT})",
    )
    parser.add_argument(
        "--from-git",
        nargs="?",
        const="HEAD~1..HEAD",
        metavar="RANGE",
        help="Map git path changes to URLs (default RANGE: HEAD~1..HEAD). "
        "This is the default mode when no --url/--urls-file/--from-sitemap is given.",
    )
    parser.add_argument(
        "--from-sitemap",
        action="store_true",
        help="Pick never-submitted sitemap URLs (vs indexnow_submit_state.json), "
        "preferring posts/teach. Still streamed one-by-one. "
        "Also used automatically when --from-git finds 0 page URLs.",
    )
    parser.add_argument(
        "--changed-since-days",
        type=int,
        default=14,
        help="With --from-sitemap: prefer URLs whose lastmod is within N days "
        "(default 14). Never-submitted URLs are still eligible.",
    )
    parser.add_argument(
        "--include-resubmit",
        action="store_true",
        help="With --from-sitemap: also consider already-submitted URLs "
        "(lastmod filter only). Default is new-to-IndexNow only.",
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
        help="Submit this URL (repeatable). Preferred for streaming.",
    )
    parser.add_argument(
        "--urls-file",
        help="File with one URL per line.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=STREAM_DELAY_SEC,
        help=f"Seconds between streamed requests (default {STREAM_DELAY_SEC})",
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

    explicit = bool(args.url or args.urls_file)
    use_sitemap = bool(args.from_sitemap or args.sitemap_file)
    use_git = args.from_git is not None or (not explicit and not use_sitemap)

    def collect_sitemap_urls(*, label: str) -> list[str]:
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
            return []

        seen: set[str] = set()
        unique: list[tuple[str, str]] = []
        for url, lastmod in sitemap:
            if url not in seen:
                seen.add(url)
                unique.append((url, lastmod))

        since = datetime.now(timezone.utc).timestamp() - max(0, args.changed_since_days) * 86400
        since_dt = datetime.fromtimestamp(since, tz=timezone.utc)
        state = load_state()
        limit = max(1, args.limit)
        only_new = not args.include_resubmit
        picked = pick_changed_sitemap_urls(
            unique, state, limit, since_dt, only_new=only_new
        )
        mode = "new-to-IndexNow" if only_new else "lastmod window (incl. resubmit)"
        print(
            f"{label}: sitemap={len(unique)} | {mode} | "
            f"lastmod≥{since_dt.date()} | streaming={len(picked)} (limit {limit})"
        )
        if len(picked) > 20:
            print(
                "Warning: >20 URLs looks like IndexNow batch mode to Bing. "
                "Lower --limit or submit with --url after publish.",
                file=sys.stderr,
            )
        return picked

    urls: list[str] = []
    if explicit:
        for u in args.url:
            nu = normalize_url(u)
            if nu:
                urls.append(nu)
        if args.urls_file:
            for line in Path(args.urls_file).read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    urls.append(normalize_url(line))
        urls = dedupe(urls)
    elif use_sitemap:
        urls = collect_sitemap_urls(label="Sitemap mode")
    else:
        revision = args.from_git or "HEAD~1..HEAD"
        print(f"[{utc_now()}] Mapping git changes: {revision}")
        urls = urls_from_git(revision)
        urls = urls[: max(1, args.limit)] if urls else []
        print(f"Changed page URL(s): {len(urls)}")
        if not urls:
            # Data-only tip commit (teach catalog sync) often sits on top of a
            # published post. Fall back to never-submitted sitemap URLs.
            print("Git mapped 0 page URLs — falling back to sitemap new-URL diff.")
            urls = collect_sitemap_urls(label="Sitemap fallback")

    if not urls:
        print("Nothing to submit.")
        return 0

    if args.dry_run:
        for u in urls:
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

    print(f"Streaming {len(urls)} URL(s) to IndexNow ({ENDPOINT})…")
    state = load_state()
    at = utc_now()
    ok_count = 0
    fail_count = 0
    last_status = 0
    last_raw = ""

    for i, url in enumerate(urls):
        status, raw = stream_indexnow(host, key, url)
        last_status, last_raw = status, raw
        # 200 = OK, 202 = accepted (key validation may still be pending)
        if status in (200, 202):
            state["submitted"][url] = {"at": at}
            ok_count += 1
            print(f"  [{i + 1}/{len(urls)}] HTTP {status} {url}")
        else:
            fail_count += 1
            print(f"  [{i + 1}/{len(urls)}] HTTP {status} {url}", file=sys.stderr)
            if raw:
                print(f"    {raw}", file=sys.stderr)
            if status == 429:
                print("Rate limited; stopping.", file=sys.stderr)
                break
        if i + 1 < len(urls) and args.delay > 0:
            time.sleep(args.delay)

    state["last_run"] = {
        "at": at,
        "mode": "stream",
        "requested": len(urls),
        "ok": ok_count,
        "failed": fail_count,
        "status": last_status,
        "response": last_raw,
    }
    save_state(state)
    print(f"Recorded {ok_count} URL(s) in {STATE_FILE.name}")
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
