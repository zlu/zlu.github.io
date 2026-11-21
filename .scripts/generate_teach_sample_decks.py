#!/usr/bin/env python3
"""Generate 4-slide tutoring intro decks for courses with enough private materials.

For each unique teaching/<folder>/<material> with enough content files:
  1. Build a short Marp tutoring deck (cover / topics / stuck / tutoring)
  2. Export PDF via marp/zlu-marp
  3. Render 4 WebP sample pages under assets/teach/<folder>/<course-slug>/sample/
  4. Wire sample_slides into every admin course that points at that material

Never claims university lecture notes. Sample framing is always "our tutoring deck".

Usage:
  python3 .scripts/generate_teach_sample_decks.py            # dry-run
  python3 .scripts/generate_teach_sample_decks.py --write    # decks + yaml
  python3 .scripts/generate_teach_sample_decks.py --write --min-files 5
  python3 .scripts/generate_teach_sample_decks.py --write --limit 10
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".scripts"))

from enrich_teach_courses import (  # noqa: E402
    ASSN_RE,
    CONTENT_EXT,
    LECT_HINT_RE,
    SKIP_DIR,
    TEACHING,
    TOPIC_LEXICON,
    WEEK_RE,
    clean_token,
    load_themes,
    normalize_theme,
    resolve_material_path,
    scan_files,
)


def gather_evidence_light(material_path: Path) -> dict:
    """Like gather_evidence, but filenames only — no PDF text extraction."""
    files = scan_files(material_path, limit=400)
    rels = [str(p.relative_to(material_path)) for p in files]
    weeks = sorted({
        int(m.group(1))
        for rel in rels
        for m in [WEEK_RE.search(rel)]
        if m and 0 <= int(m.group(1)) <= 20
    })
    assn_files = [r for r in rels if ASSN_RE.search(r)]
    lect_files = [r for r in rels if LECT_HINT_RE.search(r) or WEEK_RE.search(r)]
    tops = sorted({r.split("/")[0] for r in rels if "/" in r})[:16]
    bases = [clean_token(Path(r).name) for r in rels]
    hay = " ".join(rels + bases + tops).lower()
    topic_hits = []
    seen = set()
    for rx, en, cn, hint in TOPIC_LEXICON:
        if re.search(rx, hay, re.I) and en not in seen:
            topic_hits.append({"en": en, "cn": cn, "hint": hint})
            seen.add(en)
        if len(topic_hits) >= 8:
            break
    week_labels = []
    for rel in lect_files + rels:
        m = WEEK_RE.search(rel)
        if not m:
            continue
        label = clean_token(Path(rel).name)
        label = re.sub(r"(?i)week\s*\d+\s*", "", label).strip(" -_:")
        # Drop raw "lecture notes" filename noise from public topics.
        if len(label) >= 4 and not re.search(r"(?i)^lecture\s*notes?", label):
            week_labels.append((int(m.group(1)), label))
    week_labels = sorted(set(week_labels))[:10]
    assn_labels = []
    for rel in assn_files[:20]:
        label = clean_token(Path(rel).name)
        if len(label) >= 3:
            assn_labels.append(label)
    return {
        "n_files": len(files),
        "weeks": weeks,
        "week_span": [min(weeks), max(weeks)] if weeks else None,
        "tops": tops,
        "assn_files": assn_files[:15],
        "assn_labels": assn_labels[:10],
        "lect_files": lect_files[:15],
        "week_labels": week_labels,
        "topic_hits": topic_hits,
        "snippets": [],
        "hay": hay[:4000],
        "has_notebooks": any(p.suffix.lower() == ".ipynb" for p in files),
        "has_code": any(p.suffix.lower() in {".py", ".java", ".c", ".cpp", ".r", ".scala"} for p in files),
        "has_pdf": any(p.suffix.lower() == ".pdf" for p in files),
    }

COURSES_DIR = ROOT / "_data" / "courses"
MARP = ROOT / "marp" / "zlu-marp"
ASSETS = ROOT / "assets" / "teach"

MIN_FILES_DEFAULT = 5


def ascii_slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower())
    return re.sub(r"-+", "-", s).strip("-")


def load_admin(path: Path) -> dict:
    docs = list(yaml.safe_load_all(path.read_text()))
    return next(d for d in docs if isinstance(d, dict))


def dump_admin(doc: dict) -> str:
    body = yaml.safe_dump(doc, allow_unicode=True, sort_keys=False, width=100).rstrip() + "\n"
    return f"---\n{body}---\n"


def yaml_escape(s: str) -> str:
    return s.replace('"', '\\"')


def topic_lines(ev: dict, themes: dict, theme: str, limit: int = 4) -> list[tuple[str, str]]:
    hits = ev.get("topic_hits") or []
    out = [(h["en"], h["cn"]) for h in hits[:limit]]
    if len(out) >= 2:
        return out
    # week labels
    for _w, label in (ev.get("week_labels") or [])[:limit]:
        if len(label) >= 4:
            out.append((label[:60], label[:60]))
        if len(out) >= limit:
            break
    if len(out) >= 2:
        return out
    # theme fallback
    meta = themes.get(theme) or {}
    for t in (meta.get("topics") or [])[:limit]:
        if isinstance(t, dict) and t.get("en"):
            out.append((t["en"], t.get("cn") or t["en"]))
    return out[:limit]


def stuck_lines(ev: dict, themes: dict, theme: str, limit: int = 3) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if ev.get("assn_labels"):
        out.append(
            (
                f"Assignment briefs such as {ev['assn_labels'][0][:48]}",
                f"作业 brief（如 {ev['assn_labels'][0][:40]}）读不透或对不上评分",
            )
        )
    if ev.get("has_notebooks"):
        out.append(
            (
                "Notebook runs once, but is not re-runnable / marker-ready",
                "Notebook 碰巧能跑，但复跑不了、交不出去",
            )
        )
    if ev.get("has_code"):
        out.append(
            (
                "Code compiles locally but fails autograder / edge cases",
                "本地能跑，autograder / 边界情况挂掉",
            )
        )
    meta = themes.get(theme) or {}
    for t in meta.get("stuck") or []:
        if isinstance(t, dict) and t.get("en"):
            out.append((t["en"], t.get("cn") or t["en"]))
        if len(out) >= limit:
            break
    if not out:
        out = [
            ("Lecture speed + dense vocabulary", "课速快、术语密，笔记跟丢"),
            ("Class demo does not replay alone", "课上演示自己复不出来"),
            ("Brief asks for a method never written down", "brief 要的方法从没写清"),
        ]
    # dedupe
    seen = set()
    uniq = []
    for en, cn in out:
        if en in seen:
            continue
        seen.add(en)
        uniq.append((en, cn))
    return uniq[:limit]


def cover_parts(topics: list[tuple[str, str]]) -> str:
    bits = []
    for en, _ in topics[:4]:
        token = re.split(r"[/,·—-]", en)[0].strip()
        token = re.sub(r"\s+", " ", token)[:28].upper()
        if token:
            bits.append(token)
    return " · ".join(bits) if bits else "LECTURE · ASSIGNMENT · EXAM"


def build_deck_md(
    code: str,
    title: str,
    uni_label: str,
    topics: list[tuple[str, str]],
    stuck: list[tuple[str, str]],
) -> str:
    handle = code or title
    short_title = title if len(title) <= 48 else title[:45].rsplit(" ", 1)[0] + "…"
    parts = cover_parts(topics)
    topic_rows = ""
    for i, (en, _cn) in enumerate(topics[:4]):
        badge = chr(ord("A") + i)
        topic_rows += f"""  <div class="roadmap-row">
    <div class="badge">{badge}</div>
    <div>
      <strong>{en}</strong>
      <span class="desc">tutoring focus</span>
    </div>
    <div class="meta">~15 min</div>
  </div>
"""
    stuck_lis = "\n".join(f"- {en}" for en, _ in stuck[:3])

    return f"""---
marp: true
theme: zlu
title: {handle} · Tutoring Intro
description: Tutoring intro deck for {handle} ({uni_label}).
paginate: true
size: 16:9
math: mathjax
html: true
footer: zlu · {handle}
---

<!-- _class: cover -->

zlu · {handle} · {uni_label.upper()[:28]}

# {short_title}

Tutoring intro — what we cover, where people stall, how sessions run.

<div class="cover-parts">{parts}</div>

---

# What we cover

<style scoped>
.roadmap {{ gap: 0.35rem; margin-top: 0.4rem; }}
.roadmap-row {{ padding: 0.42rem 0.7rem; gap: 0.7rem; }}
.roadmap-row .badge {{ width: 1.9rem; height: 1.9rem; font-size: 0.85em; border-radius: 8px; }}
.roadmap-row strong {{ font-size: 0.85em; margin-bottom: 0; }}
.roadmap-row span.desc {{ font-size: 0.68em; }}
.roadmap-row .meta {{ font-size: 0.72em; }}
</style>

<p class="eyebrow">{handle} · TOPICS</p>

<div class="roadmap">
{topic_rows}</div>

---

# Where people get stuck

<p class="eyebrow">{handle} · STUCK</p>

{stuck_lis}

<div class="tip">
<strong>Bring to session.</strong> Your brief / rubric PDF, the failing notebook or code, and one concrete stuck point.
</div>

---

# How tutoring runs

<p class="eyebrow">{handle} · SESSION</p>

1. **Lecture sync** — catch up on this week's ideas in bilingual terms.
2. **Assignment** — you type; we debug approach, errors, and edge cases.
3. **Exam / quiz** — timed drills on recurring question shapes.

<div class="cover-parts" style="margin-top:1.1rem">CATCH-UP · DEBUG · DRILL</div>
"""


def export_pdf(md_path: Path, pdf_path: Path) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(MARP), str(md_path), "-o", str(pdf_path)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def render_webps(pdf_path: Path, out_dir: Path) -> list[str]:
    import fitz
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*"):
        old.unlink()

    doc = fitz.open(pdf_path)
    # Always publish first 4 pages of the intro deck.
    names = ["01-title", "02-topics", "03-stuck", "04-tutoring"]
    mat = fitz.Matrix(2.0, 2.0)
    stems = []
    for i, stem in enumerate(names):
        if i >= doc.page_count:
            break
        pix = doc[i].get_pixmap(matrix=mat, alpha=False)
        png = out_dir / f"{stem}.png"
        webp = out_dir / f"{stem}.webp"
        pix.save(str(png))
        Image.open(png).save(webp, "WEBP", quality=82, method=6)
        png.unlink()
        stems.append(stem)
    return stems


def sample_slides_block(
    code: str,
    title: str,
    asset_prefix: str,
    topics: list[tuple[str, str]],
    stems: list[str],
) -> dict[str, Any]:
    handle = code or title
    bullets = [{"en": en, "cn": cn} for en, cn in topics[:4]]
    captions = {
        "01-title": ("Title", "封面"),
        "02-topics": ("Topics", "讲什么"),
        "03-stuck": ("Stuck points", "常见卡点"),
        "04-tutoring": ("How tutoring runs", "怎么上"),
    }
    slides = []
    for stem in stems:
        cap_en, cap_cn = captions.get(stem, (stem, stem))
        slides.append(
            {
                "src": f"{asset_prefix}/{stem}.webp",
                "alt_en": f"{handle} tutoring {cap_en.lower()}",
                "alt_cn": f"{handle} 辅导课件 · {cap_cn}",
                "caption_en": cap_en,
                "caption_cn": cap_cn,
            }
        )
    return {
        "title_en": f"{handle} — sample slides",
        "title_cn": f"{handle} — 辅导课件样例",
        "lead_en": "Four pages from our tutoring intro deck: title, topics, stuck points, and how sessions run.",
        "lead_cn": "我们辅导入门课件的四页：封面、讲什么、常见卡点、怎么上。",
        "bullets": bullets,
        "slides": slides,
        "cta_en": "Book tutoring on WeChat",
        "cta_cn": "微信预约辅导",
    }


def fast_content_count(material_path: Path, limit: int = 600) -> int:
    """Count content files without opening PDFs."""
    from enrich_teach_courses import CONTENT_EXT, SKIP_DIR, scan_files

    return len(scan_files(material_path, limit=limit))


def collect_targets(min_files: int) -> list[dict]:
    """Group admin courses by unique material folder (fast count only)."""
    themes = load_themes()
    groups: dict[tuple[str, str], dict] = {}

    for yml in sorted(COURSES_DIR.glob("*.yml")):
        data = load_admin(yml)
        folder = (data.get("teaching_folder") or yml.stem).strip()
        uni_label = data.get("university") or yml.stem
        uni_cn = data.get("name_cn") or uni_label
        for course in data.get("courses") or []:
            if not isinstance(course, dict):
                continue
            mat = (course.get("material") or "").strip()
            if not mat:
                continue
            key = (folder, mat)
            g = groups.setdefault(
                key,
                {
                    "folder": folder,
                    "material": mat,
                    "uni_label": uni_label,
                    "uni_cn": uni_cn,
                    "refs": [],
                },
            )
            g["refs"].append(
                {
                    "yml": yml,
                    "code": (course.get("code") or "").strip(),
                    "title": (course.get("title") or "").strip(),
                    "theme": course.get("theme") or "",
                    "has_sample": bool(course.get("sample_slides")),
                }
            )

    targets = []
    for key, g in groups.items():
        folder, mat = key
        root = TEACHING / folder
        mpath = resolve_material_path(root, mat) if root.is_dir() else None
        if mpath is None:
            continue
        n_files = fast_content_count(mpath)
        if n_files < min_files:
            continue
        refs = sorted(g["refs"], key=lambda r: (0 if r["code"] else 1, r["code"]))
        primary = refs[0]
        code = primary["code"]
        title = primary["title"] or mat
        identity_hay = f"{code} {title} {mat}"
        theme = normalize_theme(str(primary["theme"] or ""), identity_hay, themes)
        course_slug = ascii_slug(code or title)
        if not course_slug:
            course_slug = ascii_slug(mat)[:40]
        # Public asset paths must be ASCII (Jekyll static file URLs).
        asset_folder = ascii_slug(folder) or ascii_slug(refs[0]["yml"].stem)
        if not asset_folder:
            asset_folder = "teach-misc"
        targets.append(
            {
                "folder": folder,
                "asset_folder": asset_folder,
                "material": mat,
                "mpath": mpath,
                "n_files": n_files,
                "uni_label": g["uni_label"],
                "uni_cn": g["uni_cn"],
                "code": code,
                "title": title,
                "theme": theme,
                "course_slug": course_slug,
                "refs": refs,
                "themes": themes,
                "already_sampled": any(r["has_sample"] for r in refs),
            }
        )
    return sorted(targets, key=lambda t: (-t["n_files"], t["folder"], t["code"]))


def process_target(t: dict, write: bool, skip_existing: bool) -> dict:
    code, title = t["code"], t["title"]
    handle = code or title
    deck_dir = t["mpath"] / "decks"
    md_path = deck_dir / "tutoring-intro.md"
    pdf_path = deck_dir / "tutoring-intro.pdf"
    asset_folder = t.get("asset_folder") or ascii_slug(t["folder"]) or "teach-misc"
    asset_dir = ASSETS / asset_folder / t["course_slug"] / "sample"
    asset_prefix = f"/assets/teach/{asset_folder}/{t['course_slug']}/sample"

    result = {
        "handle": handle,
        "n_files": t["n_files"],
        "folder": t["folder"],
        "material": t["material"],
        "skipped": False,
        "error": None,
    }

    if skip_existing and t["already_sampled"]:
        result["skipped"] = True
        return result

    # Filename-level evidence is enough for a 4-slide intro (skip slow PDF OCR/parse).
    ev = gather_evidence_light(t["mpath"])
    theme = t["theme"]
    if theme == "general" and ev.get("topic_hits"):
        theme = ev["topic_hits"][0].get("hint") or theme
        theme = normalize_theme(theme, f"{code} {title}", t["themes"])
    topics = topic_lines(ev, t["themes"], theme)
    stuck = stuck_lines(ev, t["themes"], theme)

    md = build_deck_md(code, title, t["uni_label"], topics, stuck)

    if not write:
        result["preview_topics"] = [en for en, _ in topics]
        return result

    try:
        deck_dir.mkdir(parents=True, exist_ok=True)
        md_path.write_text(md, encoding="utf-8")
        export_pdf(md_path, pdf_path)
        stems = render_webps(pdf_path, asset_dir)
        block = sample_slides_block(code, title, asset_prefix, topics, stems)

        # Wire into every admin course pointing at this material
        by_yml: dict[Path, list[str]] = defaultdict(list)
        for ref in t["refs"]:
            by_yml[ref["yml"]].append(ref["code"])

        for yml_path, codes in by_yml.items():
            data = load_admin(yml_path)
            changed = False
            code_set = {c.replace(" ", "") for c in codes if c} | set(codes)
            for course in data.get("courses") or []:
                if not isinstance(course, dict):
                    continue
                if (course.get("material") or "").strip() != t["material"]:
                    continue
                ccode = (course.get("code") or "").strip()
                if code_set and ccode and ccode.replace(" ", "") not in code_set and ccode not in code_set:
                    # same material folder may be shared — still wire all sharing this material
                    pass
                if (course.get("material") or "").strip() == t["material"]:
                    # Always attach samples; lightly refresh overview to mention samples if missing
                    course["sample_slides"] = block
                    ov = (course.get("overview_en") or "")
                    if "sample slide" not in ov.lower() and "tutoring deck" not in ov.lower():
                        course["overview_en"] = (
                            (ov + " " if ov else "")
                            + "Sample slides from our tutoring intro deck are below."
                        ).strip()
                    ov_cn = (course.get("overview_cn") or "")
                    if "样例" not in ov_cn and "辅导课件" not in ov_cn:
                        course["overview_cn"] = (
                            (ov_cn + " " if ov_cn else "")
                            + "下方有我们辅导入门课件的样例页。"
                        ).strip()
                    # FAQ item if none about samples
                    faqs = course.get("faq") or []
                    if not any("sample" in str(f.get("q_en", "")).lower() for f in faqs if isinstance(f, dict)):
                        faqs = list(faqs) + [
                            {
                                "q_en": "Are sample slides available?",
                                "q_cn": "有辅导课件样例吗？",
                                "a_en": "Yes — sample pages from our tutoring intro deck are on this page.",
                                "a_cn": "有，本页有我们辅导入门课件的样例页。",
                            }
                        ]
                        course["faq"] = faqs
                    changed = True
            if changed:
                yml_path.write_text(dump_admin(data), encoding="utf-8")

        result["stems"] = stems
        result["asset"] = str(asset_dir.relative_to(ROOT))
    except Exception as e:
        result["error"] = str(e)
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--min-files", type=int, default=MIN_FILES_DEFAULT)
    ap.add_argument("--limit", type=int, default=0, help="Process at most N folders")
    ap.add_argument(
        "--include-existing",
        action="store_true",
        help="Regenerate even if sample_slides already present",
    )
    ap.add_argument("--folder", action="append", default=[], help="Limit to teaching folder name(s)")
    args = ap.parse_args()

    if not MARP.exists():
        print(f"missing marp wrapper: {MARP}", file=sys.stderr)
        return 1

    targets = collect_targets(args.min_files)
    if args.folder:
        allow = set(args.folder)
        targets = [t for t in targets if t["folder"] in allow]
    if args.limit:
        targets = targets[: args.limit]

    skip_existing = not args.include_existing
    print(f"targets={len(targets)} min_files={args.min_files} write={args.write}")

    ok = skip = err = 0
    for i, t in enumerate(targets, 1):
        handle = t["code"] or t["title"]
        print(f"[{i}/{len(targets)}] {t['folder']}/{t['material'][:40]} · {handle} n={t['n_files']}", flush=True)
        r = process_target(t, write=args.write, skip_existing=skip_existing)
        if r.get("skipped"):
            skip += 1
            print("  skip existing sample")
        elif r.get("error"):
            err += 1
            print(f"  ERROR: {r['error']}")
        else:
            ok += 1
            if not args.write:
                print(f"  dry topics: {r.get('preview_topics')}")
            else:
                print(f"  wrote {r.get('asset')} stems={r.get('stems')}")

    mode = "WROTE" if args.write else "DRY-RUN"
    print(f"\n{mode}: ok={ok} skipped={skip} errors={err}")
    return 0 if err == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
