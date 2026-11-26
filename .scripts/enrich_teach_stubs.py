#!/usr/bin/env python3
"""Enrich stub teach courses (no lead_en) with subject-pack-specific copy.

Only the page shell is generic. Topics / lead / stuck come from
`_data/teach_subject_packs.yml` matched on code+title — never from tutoring-process
filler ("bilingual catch-up chips") or bare theme packs.

Usage:
  python3 .scripts/enrich_teach_stubs.py                 # dry-run
  python3 .scripts/enrich_teach_stubs.py --write         # apply
  python3 .scripts/enrich_teach_stubs.py --write --uni nus
  python3 .scripts/enrich_teach_stubs.py --write --force # also refresh weak theme-ish leads
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional

import yaml

ROOT = Path(__file__).resolve().parents[1]
COURSES_DIR = ROOT / "_data" / "courses"
PACKS_PATH = ROOT / "_data" / "teach_subject_packs.yml"

# Leads that are clearly theme/synth filler — eligible for --force refresh.
WEAK_LEAD_RE = re.compile(
    r"Briefs, tooling, and autograder|"
    r"Abstractions stack fast|"
    r"One-to-one tutoring: catch up on lectures|"
    r"lecture speed and dense vocabulary|"
    r"Common topics: (?:Programming foundations|Reading the brief|OS / systems programming)",
    re.I,
)
WEAK_TOPIC_MARKERS = (
    "Reading the brief and designing before typing",
    "Programming foundations and debugging",
    "Git basics when the course requires it",
    "Algorithms, ADTs, and complexity",
    "Security / privacy threat models and labs",
    "中英双语补课",
)


def load_yaml_doc(path: Path) -> dict:
    docs = list(yaml.safe_load_all(path.read_text(encoding="utf-8")))
    return next(d for d in docs if isinstance(d, dict))


def dump_admin(doc: dict) -> str:
    body = yaml.safe_dump(doc, allow_unicode=True, sort_keys=False, width=100).rstrip() + "\n"
    return f"---\n{body}---\n"


def load_packs() -> list[dict]:
    data = load_yaml_doc(PACKS_PATH)
    packs = list(data.get("packs") or [])
    compiled = []
    for p in packs:
        if not isinstance(p, dict):
            continue
        rx = (p.get("match") or "").strip()
        if not rx:
            continue
        try:
            cre = re.compile(rx, re.I)
        except re.error as e:
            raise SystemExit(f"bad pack match {p.get('id')}: {e}") from e
        compiled.append({**p, "_re": cre})
    return compiled


def match_pack(code: str, title: str, packs: list[dict]) -> dict:
    hay = f"{code} {title}".strip()
    for p in packs:
        if p["_re"].search(hay):
            return p
    return packs[-1]  # general-fallback


def bilingual_items(raw) -> list[dict]:
    out = []
    for t in raw or []:
        if isinstance(t, dict) and (t.get("en") or t.get("cn")):
            out.append({"en": (t.get("en") or "").strip(), "cn": (t.get("cn") or t.get("en") or "").strip()})
    return out


def title_fallback_topics(title: str) -> list[dict]:
    """When pack has empty topics (general fallback), derive chips from the title itself."""
    t = re.sub(r"\s+", " ", (title or "").strip())
    if not t:
        return [
            {"en": "Core lecture concepts for this unit", "cn": "本课核心讲座概念"},
            {"en": "Assignment / lab methods under deadline", "cn": "DDL 下的作业 / Lab 方法"},
            {"en": "Exam question shapes for this syllabus", "cn": "本课大纲的常考题型"},
        ]
    # Split on common separators; keep meaningful phrases.
    parts = re.split(r"\s+[:/\-–—]\s+|\s+and\s+|\s+&\s+|\s+with\s+", t, flags=re.I)
    parts = [p.strip(" .") for p in parts if len(p.strip(" .")) >= 4]
    topics = []
    for p in parts[:5]:
        topics.append({"en": p, "cn": p})
    if len(topics) < 3:
        topics.append({"en": f"Core ideas in {t}", "cn": f"{t} 的核心概念"})
        topics.append({"en": f"Assignments / labs for {t}", "cn": f"{t} 的作业 / Lab"})
        topics.append({"en": f"Exam drills aligned to {t}", "cn": f"对准 {t} 的考前练习"})
    return topics[:6]


def build_copy(uni_label: str, uni_cn: str, code: str, title: str, pack: dict) -> dict:
    theme = (pack.get("theme") or "general").strip() or "general"
    topics = bilingual_items(pack.get("topics"))
    if not topics:
        topics = title_fallback_topics(title)
    stuck = bilingual_items(pack.get("stuck"))
    sessions = bilingual_items(pack.get("session_angles"))
    hard_en = (pack.get("hard_en") or "").strip()
    hard_cn = (pack.get("hard_cn") or "").strip()
    if hard_en and hard_en[0].islower():
        hard_en = hard_en[0].upper() + hard_en[1:]

    label = " ".join(x for x in (code, title) if x).strip() or title or code
    cn_handle = code or title
    topic_ens = [t["en"] for t in topics[:5] if t.get("en")]
    topic_cns = [t["cn"] for t in topics[:4] if t.get("cn")]
    arc = " → ".join(topic_ens[:4])

    lead_en = (
        f"{label} at {uni_label}: {arc}"
        f"{'…' if len(topic_ens) > 4 else ''}. "
        f"{hard_en}. Tutoring rebuilds that syllabus arc, then debugs the lab or assignment where it breaks."
    )
    lead_cn = (
        f"在{uni_cn}上 {cn_handle}，主线是{' → '.join(topic_cns[:4])}"
        f"{'…' if len(topic_cns) > 4 else ''}。"
        f"{hard_cn}。辅导先重建这条大纲主线，再对着 Lab / 作业里断开的地方 debug。"
    )

    overview_en = (
        f"One-to-one tutoring for {label} at {uni_label}. "
        f"Covered topics: {'; '.join(topic_ens)}. "
        f"Bring your brief, failing test, lecture slide screenshot, or the topic name you are stuck on."
    )
    overview_cn = (
        f"{cn_handle}（{uni_cn}）一对一辅导。"
        f"覆盖主题包括{'、'.join(topic_cns)}。"
        f"预约时请带上 brief、失败测例、卡住的课件截图，或主题名。"
    )

    if not sessions:
        sessions = [
            {"en": "Restate this week’s claim before touching the homework", "cn": "碰作业前先复述本周命题"},
            {"en": "Map the brief to a short checklist; you drive the fix", "cn": "把 brief 映射成短清单；你来主导修改"},
            {"en": "Drill the exam shape for the weak topic you named", "cn": "对准你点名的薄弱主题做考题形练习"},
        ]

    tutoring = [
        {
            "en": f"Lecture sync for {cn_handle}: bilingual walkthrough of {arc or 'this week’s ideas'} so tutorial language matches.",
            "cn": f"按 {cn_handle} 同步跟课：用中英双语把「{(' → '.join(topic_cns[:3])) or '本周概念'}」讲透，对齐 tutorial 用语。",
        },
        {
            "en": "Assignments / labs: approach review and live debug (you at the keyboard).",
            "cn": "作业 / Lab：拆 brief、梳思路，现场 debug（你来操作键盘）。",
        },
        {
            "en": f"Exam prep: timed drills on {(topic_ens[0] if topic_ens else 'recurring')} question shapes for this course.",
            "cn": f"考前突击：按本课「{(topic_cns[0] if topic_cns else '常考')}」等题型做限时练习。",
        },
    ]

    faq = [
        {
            "q_en": f"What does {cn_handle} cover?",
            "q_cn": f"{cn_handle} 学什么？",
            "a_en": f"{'; '.join(topic_ens[:4])}"
            f"{'…' if len(topic_ens) > 4 else ''}. Say which week or assessment you are on when booking.",
            "a_cn": f"{'、'.join(topic_cns[:4])}"
            f"{'…' if len(topic_cns) > 4 else ''}。预约时说明你卡在哪一周或哪次评估。",
        },
        {
            "q_en": "What should I bring to the first session?",
            "q_cn": "第一次课前准备什么？",
            "a_en": "School, course code, deadline, plus your brief/PDF, code, or a screenshot of the lecture slide you are stuck on.",
            "a_cn": "学校、课号、截止日期，再加上 brief/PDF、代码，或卡住的那页课件截图。",
        },
    ]

    hook = topic_ens[0] if topic_ens else "lecture speed and dense vocabulary"
    st = title if len(title) <= 36 else (title[:33].rsplit(" ", 1)[0] + "…")
    head = f"{code} tutoring ({st})" if code else f"{st} tutoring"
    meta = f"{head}. Stuck on {hook}? Bilingual Chinese/English help."
    if len(meta) > 155:
        meta = meta[:152] + "…"

    return {
        "theme": theme,
        "meta_description_en": meta,
        "lead_en": lead_en,
        "lead_cn": lead_cn,
        "overview_en": overview_en,
        "overview_cn": overview_cn,
        "topics": topics[:8],
        "stuck": stuck[:5],
        "session_angles": sessions[:3],
        "tutoring": tutoring,
        "faq": faq,
        "pack_id": pack.get("id") or "unknown",
    }


def needs_enrichment(course: dict, force: bool) -> bool:
    lead = (course.get("lead_en") or "").strip()
    if not lead:
        return True
    if not force:
        return False
    if WEAK_LEAD_RE.search(lead):
        return True
    tops = " | ".join(
        (t.get("en") or "") for t in (course.get("topics") or []) if isinstance(t, dict)
    )
    return any(m in tops for m in WEAK_TOPIC_MARKERS)


def apply_copy(course: dict, copy: dict) -> None:
    course["theme"] = copy["theme"]
    course["meta_description_en"] = copy["meta_description_en"]
    course["lead_en"] = copy["lead_en"]
    course["lead_cn"] = copy["lead_cn"]
    course["overview_en"] = copy["overview_en"]
    course["overview_cn"] = copy["overview_cn"]
    course["topics"] = copy["topics"]
    course["stuck"] = copy["stuck"]
    course["session_angles"] = copy["session_angles"]
    course["tutoring"] = copy["tutoring"]
    # Preserve hand-written FAQ extras; replace only if none or only sample-slide FAQ.
    existing_faq = course.get("faq") or []
    if not existing_faq:
        course["faq"] = copy["faq"]
    else:
        # Prepend subject FAQ if not already present.
        has_cover = any(
            isinstance(f, dict) and "cover" in (f.get("q_en") or "").lower()
            for f in existing_faq
        )
        if not has_cover:
            course["faq"] = list(copy["faq"]) + list(existing_faq)


def process_file(path: Path, packs: list[dict], write: bool, force: bool, only_codes: Optional[set[str]]) -> dict:
    data = load_yaml_doc(path)
    uni_label = data.get("university") or path.stem
    uni_cn = data.get("name_cn") or uni_label
    stats = {"uni": path.stem, "enriched": 0, "skipped": 0, "packs": {}}
    changed = False
    for course in data.get("courses") or []:
        if not isinstance(course, dict):
            continue
        code = (course.get("code") or "").strip()
        title = (course.get("title") or "").strip()
        if not code and not title:
            continue
        if only_codes and code not in only_codes and code.replace(" ", "") not in only_codes:
            continue
        if not needs_enrichment(course, force=force):
            stats["skipped"] += 1
            continue
        pack = match_pack(code, title, packs)
        copy = build_copy(uni_label, uni_cn, code, title, pack)
        if write:
            apply_copy(course, copy)
            changed = True
        stats["enriched"] += 1
        pid = copy["pack_id"]
        stats["packs"][pid] = stats["packs"].get(pid, 0) + 1
    if write and changed:
        path.write_text(dump_admin(data), encoding="utf-8")
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--force", action="store_true", help="Also refresh weak theme-ish curated leads")
    ap.add_argument("--uni", action="append", default=[], help="Limit to admin slug(s)")
    ap.add_argument("--code", action="append", default=[], help="Limit to course code(s)")
    args = ap.parse_args()

    packs = load_packs()
    only = set(args.uni) if args.uni else None
    only_codes = {c.replace(" ", "") for c in args.code} | set(args.code) if args.code else None

    paths = sorted(COURSES_DIR.glob("*.yml"))
    if only:
        paths = [p for p in paths if p.stem in only]

    total = 0
    pack_tot: dict[str, int] = {}
    for path in paths:
        st = process_file(path, packs, write=args.write, force=args.force, only_codes=only_codes)
        if st["enriched"]:
            print(f"== {st['uni']} enriched={st['enriched']} skipped={st['skipped']} ==")
            for pid, n in sorted(st["packs"].items(), key=lambda x: -x[1]):
                print(f"  {pid}: {n}")
                pack_tot[pid] = pack_tot.get(pid, 0) + n
            total += st["enriched"]

    mode = "WROTE" if args.write else "DRY-RUN"
    print(f"\n{mode}: enriched {total} courses")
    print("pack totals:")
    for pid, n in sorted(pack_tot.items(), key=lambda x: -x[1]):
        print(f"  {pid}: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
