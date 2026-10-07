#!/usr/bin/env python3
"""Rebuild _data/teach_universities.yml from Wikipedia lists (AU/UK/US/CA/HK/SG)."""
from __future__ import annotations

import re
import time
import urllib.request
from collections import Counter
from html import unescape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "_data" / "teach_universities.yml"
UA = {"User-Agent": "Mozilla/5.0 (compatible; zlu.me-teach-bot/1.0; +https://zlu.me)"}

SOURCES = {
    "usa": [
        "https://en.wikipedia.org/wiki/Index_of_colleges_and_universities_in_the_United_States",
    ],
    "uk": [
        "https://en.wikipedia.org/wiki/List_of_universities_in_the_United_Kingdom",
        "https://en.wikipedia.org/wiki/List_of_universities_in_Scotland",
        "https://en.wikipedia.org/wiki/List_of_universities_in_Wales",
        "https://en.wikipedia.org/wiki/List_of_universities_in_Northern_Ireland",
    ],
    "australia": [
        "https://en.wikipedia.org/wiki/List_of_universities_in_Australia",
    ],
    "canada": [
        "https://en.wikipedia.org/wiki/List_of_universities_in_Canada",
        "https://en.wikipedia.org/wiki/List_of_colleges_in_Canada",
    ],
    "hong-kong": [
        "https://en.wikipedia.org/wiki/List_of_universities_and_colleges_in_Hong_Kong",
        "https://en.wikipedia.org/wiki/List_of_higher_education_institutions_in_Hong_Kong",
    ],
    "singapore": [
        "https://en.wikipedia.org/wiki/List_of_universities_and_colleges_in_Singapore",
    ],
}

LINK_RE = re.compile(
    r'<a[^>]+href="(?:https://en\.wikipedia\.org)?/wiki/([^"#:]+)"[^>]*title="([^"]+)"'
    r'|<a[^>]+title="([^"]+)"[^>]+href="(?:https://en\.wikipedia\.org)?/wiki/([^"#:]+)"',
    re.I,
)


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=90) as resp:
        return resp.read().decode("utf-8", "replace")


def extract_names(html: str) -> list[str]:
    names = []
    for match in LINK_RE.finditer(html):
        if match.group(1) is not None:
            href, title = match.group(1), unescape(match.group(2))
        else:
            title, href = unescape(match.group(3)), match.group(4)
        if href.startswith(("List_", "Category:", "File:", "Template:")):
            continue
        if "disambiguation" in title.lower():
            continue
        title = title.strip()
        if 4 <= len(title) <= 120:
            names.append(title)
    return names


def is_uni_name(name: str) -> bool:
    lowered = name.lower()
    bad = (
        "list of",
        "category:",
        "template:",
        "file:",
        "university system",
        "board of",
        "association of",
        "consortium",
        "history of",
        "rankings",
        "admission",
        "education in",
        "higher education",
        "college football",
        "college basketball",
        "athletic",
        "stadium",
        "alumni",
        "press",
        "publishing",
        "hospital",
        "museum",
        "library system",
    )
    if any(token in lowered for token in bad):
        return False
    keys = (
        "university",
        "college",
        "institute of technology",
        "polytechnic",
        "institute of",
        "school of mines",
        "conservatoire",
        "conservatory",
    )
    return any(token in lowered for token in keys)


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.casefold())
    return slug.strip("-")[:80]


def main() -> None:
    by_region: dict[str, dict[str, str]] = {region: {} for region in SOURCES}
    for region, urls in SOURCES.items():
        for url in urls:
            html = fetch(url)
            print(f"OK {region} {url.rsplit('/', 1)[-1][:60]}")
            for name in extract_names(html):
                if not is_uni_name(name):
                    continue
                key = name.casefold()
                if key not in by_region[region] or len(name) > len(by_region[region][key]):
                    by_region[region][key] = name
            time.sleep(0.2)
        print(f"== {region}: {len(by_region[region])}")

    items = []
    seen_slugs: set[str] = set()
    seen_names: set[str] = set()
    for region in ("australia", "uk", "canada", "hong-kong", "singapore", "usa"):
        for name in sorted(by_region[region].values(), key=lambda value: value.casefold()):
            name_key = name.casefold()
            if name_key in seen_names:
                continue
            slug = slugify(name)
            if not slug:
                continue
            if slug in seen_slugs:
                index = 2
                while f"{slug}-{index}" in seen_slugs:
                    index += 1
                slug = f"{slug}-{index}"
            seen_slugs.add(slug)
            seen_names.add(name_key)
            items.append({"name": name, "region": region, "slug": slug})

    lines = [
        "# Auto-generated university directory for teach SEO pages",
        "# Regions: australia, uk, usa, canada, hong-kong, singapore",
        f"# Count: {len(items)}",
        "",
    ]
    for item in items:
        escaped = item["name"].replace("\\", "\\\\").replace('"', '\\"')
        lines.append(f'- name: "{escaped}"')
        lines.append(f"  region: {item['region']}")
        lines.append(f"  slug: {item['slug']}")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT} count={len(items)}")
    print(dict(Counter(item["region"] for item in items)))


if __name__ == "__main__":
    main()
