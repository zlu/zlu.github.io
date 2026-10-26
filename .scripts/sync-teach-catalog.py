#!/usr/bin/env python3
"""Link Decap admin course YAML ↔ private ~/baidu-sync/teaching materials.

Admin (_data/courses/*.yml) is the source of truth for public SEO pages.
Teaching folder is private class material and is never published.

This script:
  - Scans teaching/ for filename/count metadata only (no PDF/assignment bodies)
  - Links admin unis/courses via teaching_folder + material (or conventions)
  - Writes _data/teach_catalog.yml for page enrichment
  - Prints a green / yellow / red link report
  - NEVER creates, updates, or deletes _data/courses/*.yml

Convention:
  _data/courses/<slug>.yml
    teaching_folder: <optional; defaults to slug or known alias>
    courses:
      - code / title / department
        material: <optional subfolder under teaching/<teaching_folder>/>

Run:
  python3 .scripts/sync-teach-catalog.py
  python3 .scripts/sync-teach-catalog.py --suggest      # materials with no admin course
  python3 .scripts/sync-teach-catalog.py --write-links  # fill empty material:/teaching_folder: in admin YAML
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("PyYAML required", file=sys.stderr)
    sys.exit(1)

ROOT = Path(__file__).resolve().parents[1]
TEACHING = Path.home() / "baidu-sync" / "teaching"
COURSES_DIR = ROOT / "_data" / "courses"
EVIDENCE_JSON = ROOT / "_data" / "teach_evidence.json"
CATALOG_YML = ROOT / "_data" / "teach_catalog.yml"
REPORT_MD = ROOT / "_data" / "teach_link_report.md"

SKIP_TOP = {
    "Policy-Commons-study",
    "deutsch",
    "python",
    "python in finanace",
    "rasberry-pi",
    "mock-exams",
    "statistical modelling，machine learning and artificial intelligence",
}
IGNORE_IN_COUNT = {
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    ".idea",
    "out",
    "dist",
    "build",
    ".vscode",
    "target",
    "bin",
    "obj",
}
CONTENT_EXTS = {
    ".pdf",
    ".pptx",
    ".ppt",
    ".md",
    ".ipynb",
    ".py",
    ".java",
    ".cpp",
    ".c",
    ".h",
    ".hpp",
    ".sql",
    ".r",
    ".html",
    ".tex",
    ".docx",
    ".doc",
    ".xlsx",
    ".csv",
    ".txt",
    ".js",
    ".ts",
    ".scala",
    ".go",
    ".rs",
}

TOPIC_PATTERNS = [
    (r"\ba[\-\s]?star\b|\ba\*", "A* search", "A* 搜索"),
    (r"\bsql\b|relational|normali[sz]", "SQL / databases", "SQL / 数据库"),
    (r"\bheap|priority.?queue", "heaps / priority queues", "堆 / 优先队列"),
    (r"\bbst|binary.?search.?tree|avl|red.?black", "trees / BST", "树 / BST"),
    (r"\bsort|quicksort|mergesort", "sorting", "排序"),
    (r"\blinked.?list|stack|queue", "lists / stacks / queues", "链表 / 栈 / 队列"),
    (r"\bgraph|bfs|dfs|dijkstra", "graphs", "图算法"),
    (r"\bcomplexity|big.?o|runtime", "complexity", "复杂度"),
    (r"\bpointer|memory|smart.?pointer|malloc", "memory / pointers", "内存 / 指针"),
    (r"\boop|inherit|polymorph|class(es)?\b", "OOP", "面向对象"),
    (r"\bstl|vector|map\b", "C++ STL", "C++ STL"),
    (r"\bml\b|machine.?learn|neural|backprop|gradient", "machine learning", "机器学习"),
    (r"\bllm|transformer|attention|rag\b", "LLMs / NLP", "大模型 / NLP"),
    (r"\bpython\b", "Python", "Python"),
    (r"\bjava\b", "Java", "Java"),
    (r"\bc\+\+|cpp\b", "C++", "C++"),
    (r"\bjavascript|typescript|react|node\b", "web / JS", "Web / JS"),
    (r"\bhtml|css|dom\b", "web front-end", "前端"),
    (r"\bapi|rest|http\b", "APIs / HTTP", "API / HTTP"),
    (r"\bsecurity|crypto|auth", "security", "安全"),
    (r"\bos\b|operating.?system|thread|concurren|deadlock", "operating systems", "操作系统"),
    (r"\bnetwork|tcp|udp|socket", "networking", "计算机网络"),
    (r"\bstat|regression|bayes|probability", "stats / probability", "统计 / 概率"),
    (r"\blinear.?algebra|matrix|vector.?space", "linear algebra", "线性代数"),
    (r"\bexam|midterm|final|mock", "exam prep", "考前突击"),
    (r"\bassignment|coursework|homework|lab\b", "assignments / labs", "作业 / Lab"),
    (r"\bdata.?struct", "data structures", "数据结构"),
    (r"\balgorithm", "algorithms", "算法"),
    (r"\bdatabase|dbms|query.?opt", "databases", "数据库"),
    (r"\barchitect|cpu|pipeline|cache", "computer architecture", "计算机体系结构"),
    (r"\bsoftwar.?engin|agile|uml|design.?pattern", "software engineering", "软件工程"),
    (r"\bdeep.?learn|cnn|rnn|pytorch|tensorflow", "deep learning", "深度学习"),
    (r"\bdistribut", "distributed systems", "分布式系统"),
]

# teaching-folder name → preferred admin slug (when they differ)
FOLDER_TO_SLUG = {
    "algonguin-college": "algonquin-college",
    "baruch": "baruch-college",
    "beijing-normal-hk-baptist-u": "beijing-normal-hongkong-baptist-university",
    "bristol-u": "bristol",
    "cardiff": "cardiff",
    "cmu": "carnegie-mellon-university",
    "concordia": "concordia-university",
    "cornell": "cornell-university",
    "cuhk-sz": "cuhk-shenzhen",
    "duke-kunshan-u": "duke-kunshan-university",
    "hk-baptist": "hong-kong-baptist-university",
    "hk-polytech": "hong-kong-polytechnic-university",
    "hkmu": "hong-kong-metropolitan-university",
    "imperial-college-london": "imperial-college",
    "james-cook-singapore": "jcu",
    "johnhopkins": "john-hopkins",
    "leeds-university": "leeds",
    "lmu-munich大学": "lmu-munich",
    "macquarie-university": "macquarie",
    "monash": "monash",
    "new-castle": "newcastle",
    "north-eastern": "northeastern",
    "oregon-state": "oregon-state-university",
    "queensland": "queensland",
    "sjsu": "san-jose-state-university",
    "tum": "tum",
    "u-birmingham": "birmingham",
    "u-bristol": "bristol",
    "u-edinburgh": "edinburgh",
    "u-glasgow": "glasgow",
    "u-manchester": "university-of-manchester",
    "u-melborne": "melbourne",
    "u-nottingham": "nottingham",
    "u-pittsburg": "pittsburgh",
    "u-queensland": "queensland",
    "u-south-australia": "universifty-of-south-australia",
    "u-southampton": "university-of-southampton",
    "u-sydney": "university-of-sydney",
    "u-warwick": "university-of-warwick",
    "u-western-australia": "university-of-western-australia",
    "uc-berkeley": "berkeley",
    "university-college-london": "ucl",
    "unsw-sydney": "unsw",
    "北师大": "beijing-normal-university",
    "北方工业大": "north-china-university-of-technology",
    "南阳理工": "nanyang-institute-of-technology",
    "慕尼黑工业大学": "tum",
    "澳门城市大学": "city-university-of-macau",
    "澳门理工大学": "macao-polytechnic-university",
    "澳门科技大学": "must",
}

CODE_RE = re.compile(
    r"\b((?:CS|COMP|COMPSCI|DATA|ELEC|INFO|INFS|CITS|CSC|CSE|EECS|ISYS|CART|"
    r"ESC|FIT|ENGR|ENG|MATH|MANG|MA|STAT|IE|EE|CP|IWBS|SOFT|SDSC|COMS|"
    r"INHN|CSSE|DAT|CM|CMT|LC|LI|LM)\s*[-_]?\s*\d{2,4}[A-Z]?)\b",
    re.I,
)


def ascii_slug(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return re.sub(r"^-+|-+$", "", text)


def code_key(code: str) -> str:
    return re.sub(r"\s+", "", code or "").upper()


def topic_hits(text: str):
    t = text.lower()
    hits = []
    for pat, en, cn in TOPIC_PATTERNS:
        if re.search(pat, t, re.I):
            hits.append({"en": en, "cn": cn})
    return hits


def extract_codes(*texts):
    codes = []
    for text in texts:
        for m in CODE_RE.findall(text or ""):
            codes.append(re.sub(r"\s+", " ", m).upper().replace("_", " "))
        m2 = re.match(
            r"^((?:CS|DATA|COMP|FIT|INFO|CART|ESC|MATH|MA|CM|CMT|CP|IE|EE)\d{2,4}[A-Z]?)",
            (text or "").strip(),
            re.I,
        )
        if m2:
            raw = m2.group(1).upper()
            codes.append(re.sub(r"^([A-Z]+)(\d)", r"\1 \2", raw))
    seen = set()
    out = []
    for c in codes:
        k = code_key(c)
        if k and k not in seen and len(c) < 24:
            seen.add(k)
            out.append(c)
    return out[:6]


def scan_material_dir(course_dir: Path) -> dict | None:
    files = []
    for root, dirs, fnames in os.walk(course_dir):
        dirs[:] = [d for d in dirs if d not in IGNORE_IN_COUNT and not d.startswith(".")]
        for fn in fnames:
            if fn.startswith("."):
                continue
            files.append(Path(root) / fn)
    content_files = [f for f in files if f.suffix.lower() in CONTENT_EXTS]
    if len(content_files) < 1 and len(files) < 1:
        return None

    topics = []
    for f in content_files[:120]:
        topics.extend(topic_hits(f"{course_dir.name} {f.name}"))
    # dedupe topics preserving order
    topic_map = {(t["en"], t["cn"]): t for t in topics}
    top_topics = list(topic_map.values())[:8]

    ext_counts = collections.Counter(f.suffix.lower() for f in content_files)
    weeks = set()
    for f in content_files:
        for m in re.finditer(r"(?:week|w|lecture|lec)[_\s\-]*(\d{1,2})", f.name, re.I):
            weeks.add(int(m.group(1)))

    codes = extract_codes(course_dir.name, *[f.name for f in content_files[:50]])
    has_exam = any(
        any(k in f.name.lower() for k in ("exam", "mock", "midterm", "final"))
        for f in content_files
    )
    has_assign = any(
        any(k in f.name.lower() for k in ("assignment", "homework", "coursework", "lab", "project"))
        for f in content_files
    )
    return {
        "folder": course_dir.name,
        "codes": codes,
        "topics": top_topics,
        "shape": {
            "pdf": ext_counts.get(".pdf", 0),
            "slides": ext_counts.get(".pptx", 0) + ext_counts.get(".ppt", 0),
            "notebooks": ext_counts.get(".ipynb", 0),
            "code": sum(
                ext_counts.get(x, 0)
                for x in (".py", ".java", ".cpp", ".c", ".h", ".js", ".ts", ".sql", ".r")
            ),
            "notes_md": ext_counts.get(".md", 0),
            "total_content_files": len(content_files),
            "week_span": [min(weeks), max(weeks)] if weeks else None,
        },
        "signals": {
            "exam_prep_materials": has_exam,
            "assignment_lab_materials": has_assign,
            "weekly_lecture_arc": bool(weeks),
        },
        "depth_score": len(content_files)
        + 3 * len(weeks)
        + (5 if has_exam else 0)
        + (3 if has_assign else 0),
    }


def scan_teaching_tree() -> dict[str, dict]:
    """folder_name -> { path, materials: { subfolder_name -> scan } }"""
    if not TEACHING.is_dir():
        raise SystemExit(f"Teaching folder not found: {TEACHING}")

    tree: dict[str, dict] = {}
    for uni_dir in sorted(p for p in TEACHING.iterdir() if p.is_dir() and not p.name.startswith(".")):
        if uni_dir.name in SKIP_TOP:
            continue
        materials = {}
        for child in uni_dir.iterdir():
            if child.name.startswith(".") or child.name in IGNORE_IN_COUNT:
                continue
            if child.is_dir():
                scanned = scan_material_dir(child)
                if scanned:
                    materials[child.name] = scanned
            elif child.is_file() and re.search(r"alias", child.name, re.I):
                # macOS alias pointer — record as stub keyed by filename
                codes = extract_codes(child.name)
                materials[child.name] = {
                    "folder": child.name,
                    "codes": codes,
                    "topics": [],
                    "shape": {"total_content_files": 0, "week_span": None},
                    "signals": {
                        "exam_prep_materials": False,
                        "assignment_lab_materials": False,
                        "weekly_lecture_arc": False,
                    },
                    "depth_score": 1,
                    "alias": True,
                }
        tree[uni_dir.name] = {
            "path": str(uni_dir),
            "materials": materials,
            "preferred_slug": FOLDER_TO_SLUG.get(uni_dir.name, ascii_slug(uni_dir.name)),
        }
    return tree


def load_admin_courses() -> dict[str, dict]:
    out = {}
    if not COURSES_DIR.is_dir():
        return out
    for f in COURSES_DIR.glob("*.yml"):
        text = f.read_text(encoding="utf-8")
        text = re.sub(r"\n---\s*$", "\n", text)
        try:
            docs = list(yaml.safe_load_all(text))
            data = next((d for d in docs if isinstance(d, dict)), None)
        except Exception as exc:
            print(f"WARN: skip {f.name}: {exc}", file=sys.stderr)
            continue
        if isinstance(data, dict) and data.get("university"):
            out[f.stem] = data
    return out


def resolve_teaching_folder(slug: str, entry: dict, tree: dict[str, dict]) -> str | None:
    explicit = str(entry.get("teaching_folder") or "").strip()
    if explicit:
        return explicit if explicit in tree else None

    if slug in tree:
        return slug

    # reverse alias: admin slug → teaching folder
    for folder, preferred in FOLDER_TO_SLUG.items():
        if preferred == slug and folder in tree:
            return folder

    # fuzzy: any folder whose preferred slug matches
    for folder, meta in tree.items():
        if meta.get("preferred_slug") == slug:
            return folder
    return None


def folder_has_exact_code(folder_name: str, scan: dict, code: str) -> bool:
    """True only when the course code is an exact extracted token (not COMP2107⊂COMP2017)."""
    if not code:
        return False
    tokens = [code_key(c) for c in (scan.get("codes") or [])]
    tokens += [code_key(c) for c in extract_codes(folder_name)]
    if code in tokens:
        return True
    # Folder like COMP9021-foo or comp9021_25T3
    compact = re.sub(r"[^A-Z0-9]", "", folder_name.upper())
    if compact.startswith(code) and (
        len(compact) == len(code) or not compact[len(code)].isdigit()
    ):
        return True
    return False


def match_material(
    course: dict, materials: dict[str, dict]
) -> tuple[str | None, dict | None, str]:
    """Return (material_name, scan, how)."""
    explicit = str(course.get("material") or "").strip()
    if explicit:
        if explicit in materials:
            return explicit, materials[explicit], "material field"
        key = explicit.strip("/").split("/")[-1]
        if key in materials:
            return key, materials[key], "material field"
        return explicit, None, "material field missing on disk"

    code = code_key(str(course.get("code") or ""))
    if code:
        for name, scan in materials.items():
            if folder_has_exact_code(name, scan, code):
                return name, scan, "code match"

    title = ascii_slug(str(course.get("title") or ""))
    if title and len(title) >= 8:
        for name, scan in materials.items():
            nslug = ascii_slug(name)
            if nslug == title or nslug.startswith(title) or title.startswith(nslug):
                return name, scan, "title fuzzy"

    return None, None, "unlinked"


CONFIDENT_LINK_HOW = frozenset({"material field", "code match"})


def coverage_summary(n: int, signals: dict, topics: list, lang: str) -> str:
    if lang == "en":
        bits = [f"Hands-on tutoring coverage across {n} course(s) with linked private materials."]
        if signals.get("weekly_lecture_arc"):
            bits.append("Weekly lecture catch-up.")
        if signals.get("assignment_lab_materials"):
            bits.append("Assignment / lab debug sessions.")
        if signals.get("exam_prep_materials"):
            bits.append("Exam-prep drills.")
        ens = [t["en"] for t in topics[:4] if isinstance(t, dict) and t.get("en")]
        if ens:
            bits.append("Typical themes: " + ", ".join(ens) + ".")
        bits.append("No ghostwriting — you submit your own work.")
        return " ".join(bits)

    bits = [f"基于已关联的 {n} 门课私有资料整理（资料本身不公开）。"]
    if signals.get("weekly_lecture_arc"):
        bits.append("可跟周进度同步答疑。")
    if signals.get("assignment_lab_materials"):
        bits.append("作业 / Lab 拆题与 debug。")
    if signals.get("exam_prep_materials"):
        bits.append("考前突击。")
    cns = [t["cn"] for t in topics[:4] if isinstance(t, dict) and t.get("cn")]
    if cns:
        bits.append("常见主题：" + "、".join(cns) + "。")
    bits.append("不代写、不代考。")
    return "".join(bits)


def build_catalog_and_report(admin: dict, tree: dict[str, dict], suggest: bool):
    catalog_unis = []
    green, yellow, red = [], [], []
    linked_material_keys = set()  # (folder, material_name)

    for slug, entry in sorted(admin.items()):
        uni_name = entry.get("university") or slug
        folder = resolve_teaching_folder(slug, entry, tree)
        materials = tree.get(folder, {}).get("materials", {}) if folder else {}

        courses_out = []
        uni_topics = {}
        uni_signals = {
            "exam_prep_materials": False,
            "assignment_lab_materials": False,
            "weekly_lecture_arc": False,
        }
        depth = 0
        linked_n = 0

        for course in entry.get("courses") or []:
            if not isinstance(course, dict):
                continue
            code = str(course.get("code") or "").strip()
            title = str(course.get("title") or "").strip()
            if not code and not title:
                continue

            mat_name, scan, how = match_material(course, materials) if folder else (None, None, "no teaching folder")
            label = f"{slug}: {code or title}"

            if scan:
                green.append(f"{label} ↔ {folder}/{mat_name} ({how})")
                linked_material_keys.add((folder, mat_name))
                linked_n += 1
                depth += scan.get("depth_score", 0)
                for t in scan.get("topics") or []:
                    uni_topics[(t["en"], t["cn"])] = t
                for k, v in (scan.get("signals") or {}).items():
                    uni_signals[k] = uni_signals[k] or v
                courses_out.append(
                    {
                        "code": code,
                        "title": title,
                        "material": mat_name,
                        "link": how,
                        "topics": scan.get("topics") or [],
                        "shape": scan.get("shape") or {},
                        "signals": scan.get("signals") or {},
                        "depth_score": scan.get("depth_score", 0),
                    }
                )
            elif mat_name and how.endswith("missing on disk"):
                red.append(f"{label} — material '{mat_name}' not under teaching/{folder}/")
                courses_out.append(
                    {
                        "code": code,
                        "title": title,
                        "material": mat_name,
                        "link": how,
                        "topics": [],
                        "shape": {},
                        "signals": {},
                        "depth_score": 0,
                    }
                )
            else:
                red.append(f"{label} — no material link (set material: under teaching/{folder or '?'} )")
                courses_out.append(
                    {
                        "code": code,
                        "title": title,
                        "material": course.get("material") or "",
                        "link": how,
                        "topics": [],
                        "shape": {},
                        "signals": {},
                        "depth_score": 0,
                    }
                )

        if not folder:
            red.append(f"{slug} ({uni_name}) — teaching folder not found (set teaching_folder:)")

        topics = list(uni_topics.values())[:12]
        catalog_unis.append(
            {
                "slug": slug,
                "university": uni_name,
                "teaching_folder": folder or "",
                "hand_curated": True,
                "course_count": len(courses_out),
                "linked_course_count": linked_n,
                "courses": courses_out,
                "topics": topics,
                "signals": uni_signals,
                "total_depth": depth,
                "coverage_summary_en": coverage_summary(linked_n or len(courses_out), uni_signals, topics, "en"),
                "coverage_summary_cn": coverage_summary(linked_n or len(courses_out), uni_signals, topics, "cn"),
            }
        )

    # Yellow: materials on disk with no admin course pointing at them
    if suggest:
        for folder, meta in sorted(tree.items()):
            preferred = meta["preferred_slug"]
            for mat_name, scan in meta["materials"].items():
                if (folder, mat_name) in linked_material_keys:
                    continue
                codes = ", ".join(scan.get("codes") or []) or "—"
                yellow.append(
                    f"teaching/{folder}/{mat_name} (codes: {codes}) "
                    f"— add under admin slug '{preferred}' with material: {mat_name}"
                )

    catalog_unis.sort(key=lambda u: (-u["total_depth"], u["slug"]))
    return catalog_unis, green, yellow, red


def write_report(green, yellow, red) -> str:
    lines = [
        "# Teach link report",
        "",
        "Generated by `.scripts/sync-teach-catalog.py`.",
        "",
        f"- Green (linked): **{len(green)}**",
        f"- Yellow (materials with no admin course): **{len(yellow)}**",
        f"- Red (admin missing materials / bad pointers): **{len(red)}**",
        "",
        "## Green",
        "",
    ]
    lines += [f"- {g}" for g in green] or ["- _(none)_"]
    lines += ["", "## Yellow — add in /admin?", ""]
    lines += [f"- {y}" for y in yellow] or ["- _(none — run with --suggest)_"]
    lines += ["", "## Red", ""]
    lines += [f"- {r}" for r in red] or ["- _(none)_"]
    lines.append("")
    text = "\n".join(lines)
    REPORT_MD.write_text(text, encoding="utf-8")
    return text


def dump_admin_yaml(doc: dict) -> str:
    """Preserve Decap yaml-frontmatter style (--- … ---)."""
    body = yaml.safe_dump(doc, allow_unicode=True, sort_keys=False).rstrip() + "\n"
    return f"---\n{body}---\n"


def write_links_into_admin(admin: dict, tree: dict[str, dict]) -> tuple[int, int]:
    """Fill empty teaching_folder / material from confident matches only.

    Never overwrites a non-empty material: or teaching_folder: value.
    Never deletes courses. Returns (files_updated, fields_filled).
    """
    files_updated = 0
    fields_filled = 0

    for slug, entry in admin.items():
        path = COURSES_DIR / f"{slug}.yml"
        if not path.exists():
            continue

        changed = False
        folder = resolve_teaching_folder(slug, entry, tree)
        if folder and not str(entry.get("teaching_folder") or "").strip():
            entry["teaching_folder"] = folder
            changed = True
            fields_filled += 1

        materials = tree.get(folder, {}).get("materials", {}) if folder else {}
        courses = entry.get("courses") or []
        if not isinstance(courses, list):
            continue

        for course in courses:
            if not isinstance(course, dict):
                continue
            if str(course.get("material") or "").strip():
                continue
            mat_name, scan, how = match_material(course, materials)
            if scan and how in CONFIDENT_LINK_HOW and mat_name:
                course["material"] = mat_name
                changed = True
                fields_filled += 1

        # Rebuild doc with stable key order for humans/Decap
        doc: dict = {"university": entry.get("university") or slug}
        if entry.get("teaching_folder"):
            doc["teaching_folder"] = entry["teaching_folder"]
        clean_courses = []
        for course in courses:
            if not isinstance(course, dict):
                continue
            row = {}
            if course.get("code") not in (None, ""):
                row["code"] = course["code"]
            if course.get("title") not in (None, ""):
                row["title"] = course["title"]
            if course.get("department") not in (None, ""):
                row["department"] = course["department"]
            if course.get("material") not in (None, ""):
                row["material"] = course["material"]
            if course.get("theme") not in (None, ""):
                row["theme"] = course["theme"]
            if row:
                clean_courses.append(row)
        doc["courses"] = clean_courses

        if changed:
            path.write_text(dump_admin_yaml(doc), encoding="utf-8")
            files_updated += 1

    return files_updated, fields_filled


def write_teaching_index(tree: dict[str, dict]) -> None:
    """Folder names only — for humans / Decap hints, no file contents."""
    rows = []
    for folder, meta in sorted(tree.items()):
        rows.append(
            {
                "folder": folder,
                "admin_slug": meta["preferred_slug"],
                "materials": sorted(meta["materials"].keys()),
            }
        )
    path = ROOT / "_data" / "teaching_index.yml"
    path.write_text(
        yaml.safe_dump(
            {
                "note": "Private teaching/ folder index (names only). Regenerate via sync-teach-catalog.py",
                "universities": rows,
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--suggest",
        action="store_true",
        help="List teaching materials that have no admin course yet (yellow)",
    )
    parser.add_argument(
        "--write-links",
        action="store_true",
        help=(
            "Write confident links into _data/courses/*.yml "
            "(fills empty teaching_folder / material only; never overwrites)"
        ),
    )
    args = parser.parse_args()

    print(f"Admin courses: {COURSES_DIR}")
    print(f"Teaching dir:  {TEACHING}")
    admin = load_admin_courses()
    tree = scan_teaching_tree()
    print(f"Loaded {len(admin)} admin universities, {len(tree)} teaching folders")

    write_teaching_index(tree)

    if args.write_links:
        updated, filled = write_links_into_admin(admin, tree)
        print(f"--write-links: updated {updated} YAML files, filled {filled} empty fields")
        # reload after write so catalog reflects persisted pointers
        admin = load_admin_courses()

    EVIDENCE_JSON.write_text(
        json.dumps(
            {
                "source": str(TEACHING),
                "generated_note": "Filenames/counts only — no lecture/assignment body text",
                "folders": {
                    name: {
                        "preferred_slug": meta["preferred_slug"],
                        "materials": list(meta["materials"].keys()),
                    }
                    for name, meta in tree.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    catalog_unis, green, yellow, red = build_catalog_and_report(admin, tree, suggest=args.suggest)
    public_catalog = {
        "source": "Admin _data/courses + linked private teaching/ metadata",
        "policy": (
            "Class files stay private. "
            "Run with --write-links to persist empty teaching_folder/material into admin YAML."
        ),
        "university_count": len(catalog_unis),
        "course_count": sum(u["course_count"] for u in catalog_unis),
        "linked_course_count": sum(u["linked_course_count"] for u in catalog_unis),
        "universities": catalog_unis,
    }
    CATALOG_YML.write_text(
        yaml.safe_dump(public_catalog, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    write_report(green, yellow, red)

    print(
        f"Catalog: {public_catalog['university_count']} unis, "
        f"{public_catalog['course_count']} courses, "
        f"{public_catalog['linked_course_count']} linked"
    )
    print(f"Links:  green={len(green)}  yellow={len(yellow)}  red={len(red)}")
    print(f"Wrote {CATALOG_YML}")
    print(f"Wrote {REPORT_MD}")
    print(f"Wrote _data/teaching_index.yml")
    if not args.write_links:
        print("Admin course YAML: unchanged (pass --write-links to fill empty material: fields)")
    if red[:8]:
        print("\nSample red (set material: / teaching_folder: in /admin):")
        for r in red[:8]:
            print(f"  • {r}")
    if args.suggest and yellow[:8]:
        print("\nSample yellow (materials waiting for admin courses):")
        for y in yellow[:8]:
            print(f"  • {y}")


if __name__ == "__main__":
    main()
