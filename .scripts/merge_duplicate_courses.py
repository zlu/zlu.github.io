#!/usr/bin/env python3
"""Merge duplicate _data/courses/*.yml into canonical slugs and emit redirect stubs.

Run from repo root:
  python3 .scripts/merge_duplicate_courses.py
  python3 .scripts/merge_duplicate_courses.py --dry-run
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COURSES = ROOT / "_data" / "courses"
REDIRECTS = ROOT / "redirects"

# old_slug -> canonical_slug (keep the target file; delete the source)
MERGES: list[tuple[str, str]] = [
    ("monash-university", "monash"),
    ("imperial-college", "imperial-college-london"),
    ("edinburgh", "university-of-edinburgh"),
    ("university-of-edingburgh", "university-of-edinburgh"),
    ("glasgow", "university-of-glasgow"),
    ("southampton", "university-of-southampton"),
    ("national-university-of-singapore", "nus"),
    ("western-australia", "university-of-western-australia"),
    ("universifty-of-south-australia", "south-australia"),
    ("northeastern", "northeastern-university"),
    ("hongkong-baptist", "hong-kong-baptist-university"),
    ("hongkong-metropolitan", "hong-kong-metropolitan-university"),
    ("hong-kong-polytechnic-university", "polyu"),
    ("hong-kong-university-of-science-and-technology", "hkust"),
    ("macau-university-of-science-and-technology", "must"),
    ("national-malaysia", "national-university-of-malaysia"),
]

# Empty shells that never produced pages — delete
DELETE_EMPTY = [
    "university-of-liverpool",
    "university-of-stirling",
    "william-mary",
]

SERVICES = ("sync", "assignment", "exam", "preview")


def strip_front(text: str) -> str:
    text = text.strip()
    if text.startswith("---"):
        text = text[3:].lstrip("\n")
    if "\n---" in text:
        text = text.split("\n---", 1)[0]
    return text.rstrip() + "\n"


def parse_doc(path: Path) -> dict:
    raw = strip_front(path.read_text(encoding="utf-8"))
    uni = ""
    folder = ""
    courses: list[dict] = []
    cur: dict | None = None
    in_courses = False

    for line in raw.splitlines():
        if re.match(r"^university:\s*", line):
            uni = line.split(":", 1)[1].strip().strip('"').strip("'")
            continue
        if re.match(r"^teaching_folder:\s*", line):
            folder = line.split(":", 1)[1].strip().strip('"').strip("'")
            continue
        if re.match(r"^courses:\s*", line):
            in_courses = True
            rest = line.split(":", 1)[1].strip()
            if rest in ("[]", "~", "null", ""):
                continue
            continue
        if not in_courses:
            continue
        if re.match(r"^\s*-\s+", line):
            if cur:
                courses.append(cur)
            cur = {}
            item = re.sub(r"^\s*-\s*", "", line)
            if ":" in item:
                k, _, v = item.partition(":")
                cur[k.strip()] = v.strip().strip('"').strip("'")
            continue
        if cur is not None and re.match(r"^\s+\w+:", line):
            k, _, v = line.strip().partition(":")
            cur[k.strip()] = v.strip().strip('"').strip("'")
    if cur:
        courses.append(cur)

    return {"university": uni, "teaching_folder": folder, "courses": courses}


def course_key(c: dict) -> str:
    code = re.sub(r"[^a-z0-9]+", "", (c.get("code") or "").lower())
    if code:
        return f"code:{code}"
    title = re.sub(r"[^a-z0-9]+", "", (c.get("title") or "").lower())
    return f"title:{title}"


def ascii_slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower())
    return re.sub(r"^-|-$", "", re.sub(r"-+", "-", s))


def course_slug(c: dict) -> str:
    base = (c.get("code") or "").strip() or (c.get("title") or "").strip()
    slug = ascii_slug(base)
    return slug or "course"


def prefer_course(a: dict, b: dict) -> dict:
    """Prefer the entry with material / richer fields."""
    score = lambda c: (
        1 if (c.get("material") or "").strip() else 0,
        1 if (c.get("code") or "").strip() else 0,
        len((c.get("title") or "").strip()),
        len((c.get("department") or "").strip()),
    )
    return a if score(a) >= score(b) else b


def merge_courses(keep: list[dict], extra: list[dict]) -> list[dict]:
    by: dict[str, dict] = {}
    order: list[str] = []
    for c in keep + extra:
        if not (c.get("code") or c.get("title")):
            continue
        k = course_key(c)
        if k not in by:
            by[k] = dict(c)
            order.append(k)
        else:
            by[k] = prefer_course(by[k], c)
    return [by[k] for k in order]


def dump_doc(university: str, folder: str, courses: list[dict]) -> str:
    lines = ["---", f"university: {university}"]
    if folder:
        lines.append(f"teaching_folder: {folder}")
    lines.append("courses:")
    for c in courses:
        # first field on the dash line
        if c.get("code"):
            lines.append(f"- code: {c['code']}")
            if c.get("title"):
                lines.append(f"  title: {c['title']}")
        else:
            lines.append(f"- title: {c.get('title', '')}")
        if c.get("department"):
            lines.append(f"  department: {c['department']}")
        if c.get("material"):
            lines.append(f"  material: {c['material']}")
        if c.get("theme"):
            lines.append(f"  theme: {c['theme']}")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def write_redirect(old_path: str, new_path: str, dry: bool) -> Path:
    # /teach/universities/foo/ -> teach-universities-foo.md
    stem = old_path.strip("/").replace("/", "-")
    path = REDIRECTS / f"{stem}.md"
    body = (
        "---\n"
        "layout: redirect\n"
        f"permalink: {old_path}\n"
        f"redirect_to: {new_path}\n"
        "sitemap: false\n"
        "---\n"
    )
    if not dry:
        path.write_text(body, encoding="utf-8")
    return path


def redirect_pair(old_slug: str, new_slug: str, old_courses: list[dict], dry: bool) -> list[Path]:
    written: list[Path] = []
    written.append(
        write_redirect(
            f"/teach/universities/{old_slug}/",
            f"/teach/universities/{new_slug}/",
            dry,
        )
    )
    for svc in SERVICES:
        written.append(
            write_redirect(
                f"/teach/universities/{old_slug}/{svc}/",
                f"/teach/universities/{new_slug}/{svc}/",
                dry,
            )
        )
    for c in old_courses:
        if not (c.get("code") or c.get("title")):
            continue
        cs = course_slug(c)
        written.append(
            write_redirect(
                f"/teach/courses/{old_slug}/{cs}/",
                f"/teach/courses/{new_slug}/{cs}/",
                dry,
            )
        )
    return written


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    dry = args.dry_run
    REDIRECTS.mkdir(exist_ok=True)

    for old, new in MERGES:
        old_path = COURSES / f"{old}.yml"
        new_path = COURSES / f"{new}.yml"
        if not old_path.is_file():
            print(f"skip missing old: {old}")
            continue
        if not new_path.is_file():
            print(f"ERROR missing canonical: {new}")
            return 1

        old_doc = parse_doc(old_path)
        new_doc = parse_doc(new_path)
        merged = merge_courses(new_doc["courses"], old_doc["courses"])
        university = new_doc["university"] or old_doc["university"]
        # Prefer canonical English names
        if old == "university-of-edingburgh":
            university = "University of Edinburgh"
        if old == "macau-university-of-science-and-technology":
            university = "Macau University of Science and Technology"
        folder = new_doc["teaching_folder"] or old_doc["teaching_folder"]

        print(
            f"merge {old} ({len(old_doc['courses'])}) → {new} "
            f"({len(new_doc['courses'])} → {len(merged)}) folder={folder!r}"
        )
        if not dry:
            new_path.write_text(dump_doc(university, folder, merged), encoding="utf-8")
            old_path.unlink()

        for p in redirect_pair(old, new, old_doc["courses"], dry):
            print(f"  redirect {p.name}")

    # must.yml may have been empty; ensure courses from macau merge stick
    for empty in DELETE_EMPTY:
        path = COURSES / f"{empty}.yml"
        if path.is_file():
            print(f"delete empty shell {empty}")
            if not dry:
                path.unlink()

    # national-malaysia may already be deleted via MERGES
    must = COURSES / "must.yml"
    if must.is_file():
        doc = parse_doc(must)
        if not doc["courses"]:
            print("warn: must.yml still has no courses after merge")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
