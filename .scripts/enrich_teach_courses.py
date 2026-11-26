#!/usr/bin/env python3
"""Enrich _data/courses/*.yml stubs from private teaching/ materials.

Walks each admin course with a material: link but no hand-curated lead_en,
scans the linked folder for weekly notes / assignments, peeks at a few PDF
first pages, then writes Monash-style bilingual lead / overview / topics /
stuck / tutoring grounded in that evidence. Also corrects empty codes,
placeholder titles, typos, and invalid theme fields.

Never publishes teaching materials. Never overwrites an existing lead_en.

Usage:
  python3 .scripts/enrich_teach_courses.py            # dry-run summary
  python3 .scripts/enrich_teach_courses.py --write     # apply YAML updates
  python3 .scripts/enrich_teach_courses.py --write --uni unsw
  python3 .scripts/enrich_teach_courses.py --scrub-public-copy --write
      # drop Week-N filename topics / inventory leaks without rewriting leads

Public topics are syllabus concepts only — never "Week N: <filename>".
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import yaml

ROOT = Path(__file__).resolve().parents[1]
COURSES_DIR = ROOT / "_data" / "courses"
THEMES_DIR = ROOT / "_data" / "teach_themes"
TEACHING = Path.home() / "baidu-sync" / "teaching"

SKIP_DIR = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", ".idea",
    "checkpoint", "out", "dist", "build", ".vscode", "target", "bin", "obj",
    ".jekyll-cache",
}
CONTENT_EXT = {
    ".pdf", ".pptx", ".ppt", ".md", ".ipynb", ".py", ".docx", ".doc",
    ".txt", ".rmd", ".r", ".java", ".c", ".cpp", ".h", ".hpp", ".sql",
    ".scala", ".go", ".rs", ".m", ".mlx",
}

WEEK_RE = re.compile(r"(?:^|[^\d])(?:week|w|wk)[\s_-]*0*(\d{1,2})(?:[^\d]|$)", re.I)
ASSN_RE = re.compile(
    r"(?:assign(?:ment)?s?|assessments?|ass(?:ign)?\s*\d|homework|hw\s*\d|"
    r"coursework|cw\s*\d|project\s*\d|a\d[a-z]?|lab\s*\d|pa\s*\d)",
    re.I,
)
LECT_HINT_RE = re.compile(
    r"(?:lecture|lect|seminar|tutorial|tut(?:orial)?|workshop|applied|"
    r"notes|slide|pre-?class|seminar)",
    re.I,
)

# Explicit corrections found while walking materials / admin YAML.
TITLE_FIXES: dict[tuple[str, str], str] = {
    ("nus", "CS1010E"): "Programming Methodology",
    ("imperial-college-london", "COMP6000"): "Advanced Computer Architecture",
    ("university-of-sydney", "MKTG1011"): "Fundamentals of Marketing Analytics",
    ("lmu", ""): "Symbolische Programmiersprache",
    ("concordia-university", "CART263"): "Creative Computation (JavaScript)",
    ("university-of-southampton", "MANG2092"): "Business Analytics / Management Analytics",
    ("newcastle", "CSC8202"): "Cryptography",
    ("jcu", "CP3404"): "Network Security and Cryptography",
    ("hkust", ""): "Financial Machine Learning",
    ("university-of-edinburgh", ""): "Economics Mathematics",
    ("melbourne", ""): "Mathematics (Term 1)",
    ("baruch-college", ""): "MFE Technical Interview Prep",
    ("hku", ""): "CS Master Entrance Exam Prep",
    ("northeastern-university", ""): "PhD Application Research Proposal",
    ("bayes-business-school", ""): "Introduction to R for Business Analytics",
    ("tum", ""): "Introduction to Computer Architecture",
    ("psb-academy", ""): "Introduction to Programming",
    ("oregon-state-university", ""): "Computer Architecture",
    ("cornell-university", ""): "Database Systems",
    ("duke-kunshan-university", "LIT111"): "Literature / Writing Foundations",
    ("beijing-normal-university", ""): "German A1",
}

CODE_FIXES: dict[tuple[str, str], str] = {
    # (uni_slug, material_lower) -> code
    ("lmu", "symbolische-programmierspache"): "Symbolische Programmiersprache",
    ("university-college-dublin", "mobile computing"): "Mobile Computing",
    ("university-college-dublin", "computer graphics"): "Computer Graphics",
    ("university-college-dublin", "software methodology"): "Software Methodology",
    ("hong-kong-metropolitan-university", "comp s321f - advanced-database-and-data-warehousing"): "COMP S321F",
    ("hong-kong-metropolitan-university", "comp s460f advanced topics in data mining"): "COMP S460F",
    ("university-of-sydney", "distributed systems"): "COMP3221",
    ("university-of-sydney", "mktg1011-foundamentals of marketing analytics"): "MKTG1011",
    ("baruch-college", "mfe-tech-interview"): "MFE Interview",
    ("nottingham", "eeee4119-ai-and-intelligent-systems"): "EEEE4119",
    ("hkust", "fin-ml"): "FINA ML",
    ("unc-chapel-hill", "appl 101-exploring engineering"): "APPL 101",
    ("tum", "intro to computer architecture"): "Intro to Computer Architecture",
    ("melbourne", "math-term 1"): "MAST Term1",
    ("psb-academy", "introduction to programming"): "Programming Intro",
    ("hong-kong-baptist-university", "comps413f-application-design-and-development-for-mobile-devices"): "COMPS413F",
    ("cuhk-shenzhen", "dda3020-machine-learning"): "DDA3020",
    ("sim", "itsd010 computing and cybersecurity fundamentals"): "ITSD010",
    ("beijing-normal-university", "deutsch-a1"): "Deutsch A1",
    ("cornell-university", "database"): "Database",
    ("beijing-normal-hongkong-baptist-university", "research and application of language models"): "Language Models",
    ("duke-kunshan-university", "lit111"): "LIT111",
    ("northeastern-university", "phd-application-rp"): "PhD RP",
    ("rensselaer", "csci4850-modern-binary-exploitation"): "CSCI 4850",
    ("hku", "cs-master-mock-exam"): "CS Master Exam",
    ("oregon-state-university", "computer-architecture"): "Computer Architecture",
    ("bayes-business-school", "intro-to-r"): "Intro to R",
    ("university-of-edinburgh", "econ-math"): "Econ Math",
}

TITLE_FROM_MATERIAL: dict[tuple[str, str], str] = {
    ("university-of-sydney", "SOFT2201"): "Software Construction / Design",
    ("university-of-sydney", "ELEC1601"): "Introduction to Computer Systems",
    ("melbourne", "COMP1000"): "Foundations of Algorithms",
    ("queensland", "CSSE7100"): "Introduction to Software Engineering",
    ("queensland", "COMS3200"): "Computer Networks",
    ("macquarie", "COMP6200"): "Data Science / Analytics Foundations",
    ("hong-kong-baptist-university", "COMP7025"): "Data Analytics / Mining Topics",
    ("hkust", "COMP1023"): "Introduction to Programming / Python",
    ("concordia-university", "CART211"): "Creative Computation Studio",
    ("university-of-southampton", "MATH6161"): "Mathematical Methods",
    ("university-of-southampton", "MATH6164"): "Mathematical Methods / Statistics",
    ("birmingham", "LC Artificial Intelligence 1"): "Artificial Intelligence 1",
    ("birmingham", "LI Artificial Intelligence 2"): "Artificial Intelligence 2",
    ("birmingham", "Database And Web Programming"): "Database and Web Programming",
    ("birmingham", "Data Structures and Algorithms"): "Data Structures and Algorithms",
    ("birmingham", "LM Data Structures, Algorithms, And Databases"): "Data Structures, Algorithms, and Databases",
    ("birmingham", "LMLI Artificial Intelligence And Machine Learning"): "Artificial Intelligence and Machine Learning",
    ("birmingham", "Computer Systems"): "Computer Systems",
    ("algonquin-college", "ESC4402"): "Engineering Software / Computing",
}

THEME_ORDER = [
    "automata", "security", "machine-learning", "data-science", "databases",
    "algorithms", "systems", "web-mobile", "math-for-cs", "programming", "general",
]

THEME_HARD = {
    "programming": {
        "en": "briefs, tooling, and autograder edge cases hit harder than syntax itself",
        "cn": "真正难的往往不是语法，而是 brief 读不透、工具链不熟，以及 autograder 专砸边界",
    },
    "algorithms": {
        "en": "invariants, complexity claims, and code must stay aligned under time pressure",
        "cn": "限时下不变量、复杂度结论和代码必须说的是同一件事",
    },
    "systems": {
        "en": "abstractions stack fast and labs fail for reasons that look opaque",
        "cn": "抽象叠得快，Lab 挂掉时又常常看不出是概念错还是环境错",
    },
    "databases": {
        "en": "one wrong schema or join quietly breaks every later query",
        "cn": "Schema 或 join 偏一点，后面查询会整条链一起歪",
    },
    "data-science": {
        "en": "messy notebooks and weak method stories lose marks even when a model runs",
        "cn": "就算模型能跑，凌乱 notebook 和站不住的方法叙事也会丢分",
    },
    "machine-learning": {
        "en": "method names blur together until pipelines, metrics, and write-ups diverge",
        "cn": "方法名搅在一起，pipeline、指标和报告对不上就会崩",
    },
    "math-for-cs": {
        "en": "symbols and proof phrasing outrun calculation skill",
        "cn": "符号与证明表述往往比计算本身更劝退",
    },
    "security": {
        "en": "threat models, crypto vocabulary, and lab setup form a terminology wall",
        "cn": "威胁模型、密码学术语和 Lab 环境叠成一堵术语墙",
    },
    "web-mobile": {
        "en": "front-end and back-end failures mask each other inside one brief",
        "cn": "前后端报错互相掩盖，一份 brief 里很难定位坏在哪一层",
    },
    "automata": {
        "en": "Greek letters and state diagrams look familiar until the exam figure is new",
        "cn": "希腊字母和状态图课上眼熟，考试换一张就不会读",
    },
    "general": {
        "en": "lecture speed and dense vocabulary outrun note-taking",
        "cn": "课速一快、术语一密，笔记一断档后面就更难补",
    },
}

TOPIC_LEXICON = [
    # (regex, en, cn, theme_hint)
    (r"spark|pyspark|hadoop|kafka|flink|big\s*data", "Spark / distributed data processing", "Spark / 分布式数据处理", "data-science"),
    (r"wrangl|tidy\s*data|dplyr|data\s*clean|data\s*scien|data\s*analy|data\s*mining", "Data wrangling / analytics tables", "数据清洗 / 分析表", "data-science"),
    (r"neural|deep\s*learn|cnn|rnn|transformer|pytorch|tensorflow|keras|artificial\s*intell", "Neural nets / deep learning / AI", "神经网络 / 深度学习 / AI", "machine-learning"),
    (r"\bml\b|classif|cluster|featuri[sz]|scikit|machine\s*learn", "ML pipelines, features, and model choice", "ML 流水线、特征与模型选择", "machine-learning"),
    (r"sql|schema|normali[sz]|transaction|\bdatabase\b|\bdbms\b", "SQL, schema design, and databases", "SQL、Schema 与数据库", "databases"),
    (r"network|tcp|udp|socket|wireshark|routing|\bhttp\b|\bdns\b|\bosi\b", "Networking layers, packets, and protocols", "网络分层、报文与协议", "systems"),
    (r"operat(?:ing)?\s*system|\bos\b|concurren|schedul|systems?\s*program", "OS / systems programming", "操作系统 / 系统编程", "systems"),
    (r"architect|assembly|\bcpu\b|cache|verilog|fpga|computer\s*system", "Computer architecture and organisation", "计算机体系结构与组成", "systems"),
    (r"secur|crypto|threat|vulnerab|firewall|privacy|exploit|cyber", "Security / privacy threat models and labs", "安全 / 隐私威胁模型与实验", "security"),
    (r"algorithm|complexit|dynamic\s*program|linked\s*list|\badt\b|data\s*struct", "Algorithms, ADTs, and complexity", "算法、ADT 与复杂度", "algorithms"),
    (r"automat|dfa|nfa|regex|petri|grammar|parser|compiler|formal\s*lang", "Automata, formal languages, and proofs", "自动机、形式语言与证明", "automata"),
    (r"discrete|linear\s*algebra|calculus|probability|statistic|numer|\bmath", "Math foundations and proof habits", "数学基础与证明习惯", "math-for-cs"),
    (r"\bjava\b|\bpython\b|c\+\+|javascript|typescript|\boop\b|object.?orient|program(?:ming)?\s*method", "Programming foundations and debugging", "编程基础与调试", "programming"),
    (r"software\s*design|software\s*construct|software\s*engin", "Software design and construction", "软件设计与构造", "programming"),
    (r"react|node|android|ios|mobile|html|css|\bweb\b|frontend|backend", "Web / mobile app structure", "Web / 移动应用结构", "web-mobile"),
    (r"rmarkdown|tidyverse|\bmatlab\b", "R / MATLAB scientific workflow", "R / MATLAB 科研工作流", "data-science"),
    (r"distributed|rpc|consensus|mapreduce", "Distributed systems ideas", "分布式系统概念", "systems"),
    (r"computer\s*vision|opencv", "Computer vision pipelines", "计算机视觉流水线", "machine-learning"),
    (r"nlp|language\s*model|\bllm\b|embedding", "NLP / language-model coursework", "NLP / 语言模型课业", "machine-learning"),
]


def load_themes() -> dict[str, dict]:
    themes: dict[str, dict] = {}
    for path in THEMES_DIR.glob("*.yml"):
        docs = list(yaml.safe_load_all(path.read_text()))
        data = next((d for d in docs if isinstance(d, dict)), {}) or {}
        themes[path.stem] = data
    return themes


def infer_theme(haystack: str, themes: dict[str, dict]) -> str:
    text = haystack.lower()
    for theme_id in THEME_ORDER:
        meta = themes.get(theme_id) or {}
        pattern = re.sub(r"\s+", "", str(meta.get("match") or ""))
        if not pattern:
            continue
        try:
            if re.search(pattern, text, re.I):
                return theme_id
        except re.error:
            continue
    # lexicon fallback
    for rx, _en, _cn, hint in TOPIC_LEXICON:
        if re.search(rx, text, re.I):
            return hint
    return "general"


def normalize_theme(raw: str, haystack: str, themes: dict[str, dict]) -> str:
    t = (raw or "").strip()
    if "," in t or " " in t and t not in themes:
        # invalid multi-theme field from admin
        t = ""
    if t in themes:
        return t
    return infer_theme(haystack, themes)


def resolve_material_path(root: Path, material: str) -> Optional[Path]:
    direct = root / material
    if direct.is_dir():
        return direct
    matches = [p for p in root.iterdir() if p.is_dir() and p.name.lower() == material.lower()]
    return matches[0] if matches else None


def scan_files(material_path: Path, limit: int = 500) -> list[Path]:
    out: list[Path] = []
    for p in material_path.rglob("*"):
        if not p.is_file():
            continue
        if any(part in SKIP_DIR or part.startswith(".") for part in p.relative_to(material_path).parts):
            continue
        if p.suffix.lower() not in CONTENT_EXT:
            continue
        out.append(p)
        if len(out) >= limit:
            break
    return out


def clean_token(name: str) -> str:
    s = Path(name).stem
    s = re.sub(r"[_\-]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_pdf_snippet(path: Path, max_chars: int = 900) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        return ""
    try:
        reader = PdfReader(str(path))
        text = ""
        for page in reader.pages[:2]:
            text += page.extract_text() or ""
            if len(text) >= max_chars:
                break
        text = re.sub(r"\s+", " ", text).strip()
        return text[:max_chars]
    except Exception:
        return ""


def gather_evidence(material_path: Path) -> dict:
    files = scan_files(material_path)
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
    # Prefer assignment / overview PDFs for snippets
    pdfs = [p for p in files if p.suffix.lower() == ".pdf"]
    scored: list[tuple[int, Path]] = []
    for p in pdfs:
        name = p.name.lower()
        score = 0
        if ASSN_RE.search(name):
            score += 5
        if re.search(r"overview|intro|syllabus|spec|rubric|week\s*1|lecture", name, re.I):
            score += 3
        if WEEK_RE.search(name):
            score += 1
        scored.append((score, p))
    scored.sort(key=lambda x: (-x[0], len(x[1].name)))
    snippets = []
    for _score, p in scored[:3]:
        snip = extract_pdf_snippet(p)
        if snip:
            snippets.append({"file": str(p.relative_to(material_path)), "text": snip})

    hay = " ".join(rels + bases + tops + [s["text"] for s in snippets]).lower()
    topic_hits = []
    seen = set()
    for rx, en, cn, hint in TOPIC_LEXICON:
        if re.search(rx, hay, re.I) and en not in seen:
            topic_hits.append({"en": en, "cn": cn, "hint": hint})
            seen.add(en)
        if len(topic_hits) >= 8:
            break

    # Week-labelled lecture titles as concept hints (never publish "Week N: file").
    # Skip week 0 and administrivia / unit-intro fluff — tutoring does not cover that.
    week_labels = []
    for rel in lect_files + rels:
        m = WEEK_RE.search(rel)
        if not m:
            continue
        week_n = int(m.group(1))
        if week_n <= 0:
            continue
        label = tidy_lecture_topic_label(
            re.sub(r"(?i)week\s*\d+\s*", "", clean_token(Path(rel).name))
        )
        if len(label) < 4 or is_admin_topic_label(label):
            continue
        week_labels.append((week_n, label))
    # Prefer later substantive weeks; keep stable order by week then label.
    week_labels = sorted(set(week_labels))[:14]

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
        "snippets": snippets,
        "hay": hay[:4000],
        "has_notebooks": any(p.suffix.lower() == ".ipynb" for p in files),
        "has_code": any(p.suffix.lower() in {".py", ".java", ".c", ".cpp", ".r", ".scala"} for p in files),
        "has_pdf": bool(pdfs),
    }


def title_case_course(s: str) -> str:
    small = {"of", "and", "the", "in", "for", "to", "a", "an", "on", "with", "vs"}
    parts = re.split(r"(\s+|-)", s.strip())
    out = []
    for i, p in enumerate(parts):
        if p.isspace() or p == "-":
            out.append(p)
            continue
        low = p.lower()
        if i > 0 and low in small:
            out.append(low)
        elif p.isupper() and len(p) <= 5:
            out.append(p)  # keep acronyms
        else:
            out.append(p[:1].upper() + p[1:] if p else p)
    return "".join(out)


def fix_identity(uni: str, course: dict, material: str) -> tuple[str, str, list[str]]:
    """Return (code, title, notes)."""
    notes = []
    code = (course.get("code") or "").strip()
    title = (course.get("title") or "").strip()
    key = (uni, code)
    mat_key = (uni, material.lower())

    if key in TITLE_FIXES:
        title = TITLE_FIXES[key]
        notes.append(f"title fix via map → {title}")
    elif (uni, "") in TITLE_FIXES and not code:
        title = TITLE_FIXES[(uni, "")]
        notes.append(f"title fix empty-code map → {title}")

    if mat_key in CODE_FIXES and not code:
        code = CODE_FIXES[mat_key]
        notes.append(f"code from material map → {code}")

    if key in TITLE_FROM_MATERIAL and (not title or title == code or len(title) < 6):
        title = TITLE_FROM_MATERIAL[key]
        notes.append(f"title from material map → {title}")

    # Heuristic: title is just the code
    if title and code and title.replace(" ", "").lower() == code.replace(" ", "").lower():
        # try material folder pretty name
        pretty = clean_token(material)
        pretty = re.sub(r"(?i)^" + re.escape(code) + r"\s*", "", pretty).strip(" -_:")
        if len(pretty) >= 6:
            title = title_case_course(pretty)
            notes.append(f"title from material folder → {title}")

    # Empty code: pull leading code-like token from title or material
    if not code:
        m = re.match(
            r"^((?:COMP|CS|ECE|ELEC|MATH|INFO|SOFT|FIT|INFS|DDA|APPL|CSCI|ITSD|COMPS?|MAST|MANG|EEEE|FINA)[\s_-]?[A-Z0-9./-]+)",
            title,
            re.I,
        )
        if m:
            code = m.group(1).strip()
            rest = title[m.end():].strip(" -_:")
            if rest:
                title = title_case_course(rest)
            notes.append(f"code extracted from title → {code}")
        else:
            m2 = re.match(
                r"^((?:COMP|CS|ECE|ELEC|MATH|INFO|SOFT|FIT|INFS|DDA|APPL|CSCI|ITSD|COMPS?)[\s_-]?[A-Z0-9./-]+)",
                material,
                re.I,
            )
            if m2:
                code = m2.group(1).replace("_", " ").strip()
                notes.append(f"code extracted from material → {code}")

    # Typo fixes
    typo_map = {
        "Methedology": "Methodology",
        "Foundamentals": "Fundamentals",
        "Programmierspache": "Programmiersprache",
        "Advancedcomparchitecture": "Advanced Computer Architecture",
    }
    for bad, good in typo_map.items():
        if bad.lower() in title.lower():
            title = re.sub(bad, good, title, flags=re.I)
            notes.append(f"typo {bad} → {good}")

    if not title and material:
        title = title_case_course(clean_token(material))
        notes.append("title fallback from material")

    return code, title, notes


def bilingual(en: str, cn: str) -> dict:
    return {"en": en, "cn": cn}


# Public topics must be syllabus concepts — never "Week N: <filename>".
WEEK_FILENAME_TOPIC_RE = re.compile(r"(?i)^Week\s+\d+\s*:")
# Week 0 / early "Administrivia / unit intro" are not tutoring topics.
ADMIN_TOPIC_RE = re.compile(
    r"(?i)\b(?:"
    r"administrivia|admin(?:istration)?\b|"
    r"unit\s+intro(?:duction)?|course\s+intro(?:duction)?|"
    r"course\s+(?:overview|structure|outline|welcome|info)|"
    r"more\s+administrivia|housekeeping|logistics|timetable|"
    r"getting\s+started|meet\s+the\s+team|welcome\s+week|"
    r"platform(?:s)?\b|announcements?\b"
    r")\b"
)
QUOTED_BLOB_RE = re.compile(r"[“\"「]([^”\"」]{6,})[”\"」]")
# Quoted emphasis like “almost works” is fine; private file stems are not.
FILENAME_INNER_RE = re.compile(
    r"(?i)(?:"
    r"Midterm|Final|Tutorial|Worksheet|Lecture\s*\d|answers|"
    r"AY\s*\d|S[12]\s*\+|Assig|HW\s*\d|Lab\s*\d|"
    r"\+\+|\.pdf|\.pptx?|\.docx?|"
    r"[A-Z]{2,6}\s*\d{3,5}.*\d{4}"  # e.g. CS1010E 202223 …
    r")"
)
ASSESS_FILENAME_RE = re.compile(
    r"assessments such as\s+.+?(?:,\s*with you|, with you)|针对\s+.+\s+一类评估",
    re.I,
)
# Word-bounded "the" so we never eat "There is…".
INVENTORY_OVERVIEW_RE = re.compile(
    r"Evidence from(?:\s+the\b)?(?:\s+course\b)?(?:\s+folder\b)?(?:\s+show\b)?\s*"
    r"|Private materials\s*"
    r"|现有材料(?:里能看到)?\s*"
    r"|从课程材料(?:看)?\s*",
    re.I,
)


def is_filename_topic(en: str) -> bool:
    """True when a topic chip is a week-prefixed file/folder dump."""
    return bool(WEEK_FILENAME_TOPIC_RE.match((en or "").strip()))


def is_admin_topic_label(label: str) -> bool:
    """True for unit-intro / administrivia labels — skip on tutoring pages."""
    return bool(ADMIN_TOPIC_RE.search((label or "").strip()))


def tidy_lecture_topic_label(label: str) -> str:
    """Turn a lecture basename into a syllabus-style topic chip."""
    s = clean_token(label or "")
    s = re.sub(r"(?i)\(\d+\)\s*$", "", s).strip(" -_:")
    s = re.sub(r"\s{2,}", " ", s)
    # Drop leading numbering like "1. Further Thread…"
    s = re.sub(r"^\d+\.\s*", "", s)
    return s.strip(" -_:")


def has_filename_quote(text: str) -> bool:
    """True when quotes wrap a private file/assessment stem (not emphasis)."""
    for m in QUOTED_BLOB_RE.finditer(text or ""):
        if FILENAME_INNER_RE.search(m.group(1)):
            return True
    return False


def dedupe_topics(topics: list[dict], limit: int = 8) -> list[dict]:
    out, seen = [], set()
    for t in topics:
        if not isinstance(t, dict) or not t.get("en"):
            continue
        k = t["en"].lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(bilingual(t["en"], t.get("cn") or t["en"]))
        if len(out) >= limit:
            break
    return out


def theme_topic_pool(theme: str, themes: dict[str, dict]) -> list[dict]:
    pool: list[dict] = []
    for t in (themes.get(theme) or {}).get("topics") or []:
        if isinstance(t, dict) and t.get("en"):
            pool.append(bilingual(t["en"], t.get("cn") or t["en"]))
    if theme != "general":
        for t in (themes.get("general") or {}).get("topics") or []:
            if isinstance(t, dict) and t.get("en"):
                pool.append(bilingual(t["en"], t.get("cn") or t["en"]))
    return pool


def build_topics(ev: dict, theme: str, themes: dict[str, dict]) -> list[dict]:
    """Syllabus concepts from this course’s lectures — never theme-pack filler first.

    Never publish week-labelled filenames or administrivia / unit-intro chips.
    Theme topics are a last resort only when the folder has no lecture titles and
    no lexicon hits (otherwise every systems course looks identical).
    """
    topics: list[dict] = []
    lecture_topics: list[dict] = []
    for _week, label in ev.get("week_labels") or []:
        if is_admin_topic_label(label) or is_filename_topic(label):
            continue
        lecture_topics.append(bilingual(label, label))
    topics.extend(lecture_topics)
    for hit in ev.get("topic_hits") or []:
        en = hit.get("en") or ""
        if is_admin_topic_label(en) or is_filename_topic(en):
            continue
        topics.append(bilingual(en, hit.get("cn") or en))
    # Only when we still have nothing course-specific.
    if not topics:
        topics.extend(theme_topic_pool(theme, themes))
    return dedupe_topics(topics, limit=8)


def build_stuck(ev: dict, theme: str, themes: dict[str, dict], code: str) -> list[dict]:
    """Stuck points grounded in this course’s lecture titles when available."""
    stuck: list[dict] = []
    handle = code or "this course"
    lectures = [lab for _w, lab in (ev.get("week_labels") or []) if lab and not is_admin_topic_label(lab)]
    # Course-specific: name early / mid / late technical lectures.
    if len(lectures) >= 1:
        stuck.append(bilingual(
            f"“{lectures[0]}” makes sense in lecture; your lab does not replay alone",
            f"课上「{lectures[0]}」听得懂，Lab 自己对着就复不出来",
        ))
    if len(lectures) >= 3:
        mid = lectures[len(lectures) // 2]
        stuck.append(bilingual(
            f"Later weeks move to “{mid}” before last week’s method is solid",
            f"后面已讲到「{mid}」，上一周的方法还没练稳",
        ))
    if len(lectures) >= 5:
        late = lectures[-1]
        stuck.append(bilingual(
            f"Exam figures for “{late}” look new even when the slide felt familiar",
            f"「{late}」课上看过，考试换一张图仍觉得陌生",
        ))
    if ev.get("assn_labels") or ev.get("assn_files"):
        stuck.append(bilingual(
            "Assignment brief makes sense in class, freezes you alone",
            "作业 brief 课上听得懂，自己对着就停住",
        ))
    if ev.get("has_code") and theme in {"programming", "algorithms", "systems", "security"}:
        stuck.append(bilingual(
            "Code almost works; edge cases and error messages have no debug method",
            "代码差不多能跑，边界和报错却没有 debug 方法",
        ))
    if not lectures:
        # No lecture titles — only then fall back to theme stuck chips.
        for t in (themes.get(theme) or {}).get("stuck") or []:
            if isinstance(t, dict) and t.get("en"):
                stuck.append(bilingual(t["en"], t.get("cn") or t["en"]))
            if len(stuck) >= 5:
                break
        stuck.append(bilingual(
            f"Lecture vocabulary for {handle} outruns bilingual notes",
            f"{handle} 的课堂术语跑在双语笔记前面",
        ))
    out, seen = [], set()
    for t in stuck:
        k = t["en"].lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(t)
        if len(out) >= 5:
            break
    return out


def build_session_angles(ev: dict, theme: str, themes: dict[str, dict]) -> list[dict]:
    angles = []
    if ev.get("week_span"):
        angles.append(bilingual(
            "Rebuild this week’s lecture idea before touching the homework",
            "先重建当周讲座概念，再碰作业",
        ))
    if ev.get("assn_files"):
        angles.append(bilingual(
            "Map the brief to a short checklist; you drive, we chase the failing case",
            "把 brief 映射成短清单；你来操作，一起追失败测例",
        ))
    if ev.get("has_notebooks") or ev.get("has_code"):
        angles.append(bilingual(
            "You type in your environment; we keep the plan visible beside the editor",
            "你在自己的环境里打字；计划始终放在编辑器旁边",
        ))
    for t in (themes.get(theme) or {}).get("session_angles") or []:
        if isinstance(t, dict) and t.get("en"):
            angles.append(bilingual(t["en"], t.get("cn") or t["en"]))
        if len(angles) >= 3:
            break
    while len(angles) < 3:
        angles.append(bilingual(
            "Timed short drills in the same shape as workshops / exams",
            "用与 workshop / 考试同构的题型做限时短练",
        ))
        break
    return angles[:3]


def build_tutoring(code: str, title: str, ev: dict) -> list[dict]:
    handle = code or title
    # Never list private assessment filenames on the public page.
    return [
        bilingual(
            f"Lecture sync for {handle}: bilingual catch-up so workshop / tutorial language matches the lectures.",
            f"按 {handle} 当周进度跟课：用中英双语把没听清的概念补上，并保留 workshop / tutorial 会用的英文说法。",
        ),
        bilingual(
            "Assignments: approach review and live debug against the real brief.",
            "作业辅导：对着真实 brief 拆题、梳思路、现场 debug。",
        ),
        bilingual(
            "Exam / quiz prep: timed drills on weak spots and recurring question shapes for this course.",
            "考试 / quiz 突击：按这门课常考题型和你的薄弱点做限时练习。",
        ),
    ]


def build_copy(uni_label: str, uni_cn: str, code: str, title: str, theme: str, ev: dict, themes: dict[str, dict]) -> dict:
    hard = THEME_HARD.get(theme) or THEME_HARD["general"]
    if code and title and code.replace(" ", "").lower() == title.replace(" ", "").lower():
        label = title
    else:
        label = " ".join(x for x in [code, title] if x).strip() or title or code
    cn_handle = code or title
    theme_label_en = (themes.get(theme) or {}).get("label_en") or theme
    theme_label_cn = (themes.get(theme) or {}).get("label_cn") or theme

    # Prefer substantive lecture titles — these are the course-specific signal.
    lecture_ens = [lab for _w, lab in (ev.get("week_labels") or []) if lab][:6]
    hit_ens = [t["en"] for t in (ev.get("topic_hits") or [])]
    hit_cns = [t["cn"] for t in (ev.get("topic_hits") or [])]
    topic_ens = (lecture_ens + [e for e in hit_ens if e not in lecture_ens])[:4]
    topic_cns = (lecture_ens[:3] + [c for c in hit_cns if c not in lecture_ens])[:3]
    content_weeks = [w for w, _ in (ev.get("week_labels") or [])]

    # Never claim we hold/redistribute university lecture notes or course packs.
    # Lead must name this course’s topics when we have them — not only theme difficulty.
    if lecture_ens:
        arc = " → ".join(lecture_ens[:4])
        lead_en = (
            f"{label} at {uni_label} covers {arc}"
            f"{'…' if len(lecture_ens) > 4 else ''}. "
            f"Tutoring rebuilds that arc, then debugs the lab or assignment where it breaks."
        )
        lead_cn = (
            f"在{uni_cn}上 {cn_handle}，主线是{' → '.join(lecture_ens[:4])}"
            f"{'…' if len(lecture_ens) > 4 else ''}。"
            f"辅导先把这条主线重建清楚，再对着 Lab / 作业里断开的地方 debug。"
        )
    else:
        hard_en = hard["en"]
        if hard_en and hard_en[0].islower():
            hard_en = hard_en[0].upper() + hard_en[1:]
        lead_en = (
            f"{label} at {uni_label}. {hard_en}. "
            f"Tutoring: lecture catch-up, assignment debug, exam drills — methods you can reuse."
        )
        lead_cn = (
            f"在{uni_cn}上 {cn_handle}，{hard['cn']}。"
            f"辅导对着你卡住的步骤练，练到下一份 brief 还能自己做。"
        )

    overview_en = f"One-to-one tutoring for {label} at {uni_label}. "
    if topic_ens:
        overview_en += f"Covered topics: {'; '.join(topic_ens)}. "
    if content_weeks:
        a, b = min(content_weeks), max(content_weeks)
        overview_en += f"Technical weeks about {a}–{b} (admin / unit-intro omitted). "
    elif ev.get("assn_files"):
        overview_en += "Assessed work expects you to apply the same ideas under deadline. "
    overview_en += (
        "Bring your brief, failing test, or the lecture topic you are stuck on."
    )

    overview_cn = f"{cn_handle}（{uni_cn}）一对一辅导。"
    if topic_cns:
        overview_cn += f"覆盖主题包括{'、'.join(topic_cns)}。"
    if content_weeks:
        a, b = min(content_weeks), max(content_weeks)
        overview_cn += f"技术周大约第 {a}–{b} 周（行政 / 单元介绍不做）。"
    overview_cn += "预约时请带上 brief、失败测例，或你卡住的那课主题。"

    topics = build_topics(ev, theme, themes)
    stuck = build_stuck(ev, theme, themes, code)
    session_angles = build_session_angles(ev, theme, themes)
    tutoring = build_tutoring(code, title, ev)

    # SERP meta: code + short uni + concrete stuck hook (≤155; plugin truncates at 160).
    # Never put private filenames into meta.
    uni_short = uni_label.split()[0] if len(uni_label) > 22 else uni_label
    if ev.get("topic_hits"):
        hook = ev["topic_hits"][0]["en"]
    elif topic_ens:
        hook = topic_ens[0]
    elif ev.get("assn_files"):
        hook = "assignment briefs under deadline"
    else:
        hook = "lecture speed and dense vocabulary"
    st = title if len(title) <= 36 else (title[:33].rsplit(" ", 1)[0] + "…")
    head = f"{code} {uni_short} tutoring ({st})" if code else f"{st} {uni_short} tutoring"
    mid = f"Stuck on {hook.rstrip('.')}?"
    meta = f"{head}. {mid} Bilingual Chinese/English help."
    if len(meta) > 155:
        meta = f"{head}. {mid}"
    if len(meta) > 155:
        meta = meta[:152] + "…"

    return {
        "theme": theme,
        "meta_description_en": meta,
        "lead_en": lead_en,
        "lead_cn": lead_cn,
        "overview_en": overview_en,
        "overview_cn": overview_cn,
        "topics": topics,
        "stuck": stuck,
        "session_angles": session_angles,
        "tutoring": tutoring,
    }


def load_admin(path: Path) -> dict:
    docs = list(yaml.safe_load_all(path.read_text()))
    return next(d for d in docs if isinstance(d, dict))


def dump_admin(doc: dict) -> str:
    body = yaml.safe_dump(doc, allow_unicode=True, sort_keys=False, width=100).rstrip() + "\n"
    return f"---\n{body}---\n"


BANNED_COPY_RE = re.compile(
    r"Private materials|Evidence from(?:\s+the)?|现有材料|其余讲义|领取完整|"
    r"full course pack|we have the lecture|download the notes|"
    r"磁盘上的作业|on disk|named assignment / lab files",
    re.I,
)


def public_copy_is_leaky(course: dict) -> bool:
    """True when topics/stuck/tutoring/overview leak filenames or inventory claims."""
    for t in course.get("topics") or []:
        if isinstance(t, dict) and is_filename_topic(t.get("en") or ""):
            return True
    for s in course.get("stuck") or []:
        if isinstance(s, dict) and has_filename_quote(s.get("en") or ""):
            return True
    for s in course.get("tutoring") or []:
        if not isinstance(s, dict):
            continue
        blob = f"{s.get('en') or ''} {s.get('cn') or ''}"
        if ASSESS_FILENAME_RE.search(blob) or has_filename_quote(blob):
            return True
    blob = " ".join(
        str(course.get(k) or "")
        for k in ("lead_en", "lead_cn", "overview_en", "overview_cn", "meta_description_en")
    )
    if BANNED_COPY_RE.search(blob) or INVENTORY_OVERVIEW_RE.search(blob):
        return True
    if has_filename_quote(course.get("meta_description_en") or ""):
        return True
    # Broken leftover from older inventory scrubs: "…. re is a weekly arc…"
    if re.search(r"(?i)\.\s*re is a\b", course.get("overview_en") or ""):
        return True
    return False


def scrub_stuck_item(item: dict) -> dict:
    en = item.get("en") or ""
    cn = item.get("cn") or en
    if has_filename_quote(en) or has_filename_quote(cn):
        return bilingual(
            "Assignment brief makes sense in class, freezes you alone",
            "作业 brief 课上听得懂，自己对着就停住",
        )
    return bilingual(en, cn)


def scrub_tutoring_item(item: dict, code: str, title: str) -> dict:
    en = item.get("en") or ""
    cn = item.get("cn") or en
    blob = f"{en} {cn}"
    if ASSESS_FILENAME_RE.search(blob) or has_filename_quote(blob):
        return bilingual(
            "Assignments: approach review and live debug against the real brief.",
            "作业辅导：对着真实 brief 拆题、梳思路、现场 debug。",
        )
    # Drop accidental filename lists that slipped past the "such as" pattern.
    if re.search(r"(?i)Midterm\+|Final\+|Tutorial\+|Worksheet\+|S1\+|S2\+|AY\d", en):
        handle = code or title or "this course"
        if en.lower().startswith("assignments"):
            return bilingual(
                "Assignments: approach review and live debug against the real brief.",
                "作业辅导：对着真实 brief 拆题、梳思路、现场 debug。",
            )
        if "lecture sync" in en.lower():
            return bilingual(
                f"Lecture sync for {handle}: bilingual catch-up so workshop / tutorial language matches the lectures.",
                f"按 {handle} 当周进度跟课：用中英双语把没听清的概念补上，并保留 workshop / tutorial 会用的英文说法。",
            )
    return bilingual(en, cn)


def scrub_overview_text(text: str) -> str:
    if not text:
        return text
    cleaned = INVENTORY_OVERVIEW_RE.sub("", text)
    # Repair prior bad scrub that ate "The" from "There is…".
    cleaned = re.sub(r"(?i)(\.\s*)re is a\b", r"\1There is a", cleaned)
    cleaned = re.sub(r"(?i)^re is a\b", "There is a", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r" +\n", "\n", cleaned)
    return cleaned.strip()


def scrub_course_public_copy(course: dict, themes: dict[str, dict]) -> list[str]:
    """Fix filename topics / inventory leaks without rewriting curated leads."""
    notes: list[str] = []
    theme = normalize_theme(str(course.get("theme") or ""), f"{course.get('code')} {course.get('title')}", themes)
    if course.get("theme") != theme:
        course["theme"] = theme
        notes.append("theme normalized")

    topics = [t for t in (course.get("topics") or []) if isinstance(t, dict) and t.get("en")]
    kept = [
        t for t in topics
        if not is_filename_topic(t.get("en") or "") and not is_admin_topic_label(t.get("en") or "")
    ]
    dropped = len(topics) - len(kept)
    if dropped:
        notes.append(f"dropped {dropped} week-filename / admin topics")
    # Do not silently refill with theme-pack clones — empty is better than fake specificity.
    kept = dedupe_topics(kept, limit=8)
    if not kept:
        kept = dedupe_topics(theme_topic_pool(theme, themes), limit=8)
        notes.append("no course topics left; theme pack fallback")
    if kept != topics:
        course["topics"] = kept
        if "dropped" not in " ".join(notes) and "filled" not in " ".join(notes):
            notes.append("topics deduped")

    stuck_in = [s for s in (course.get("stuck") or []) if isinstance(s, dict)]
    stuck_out = [scrub_stuck_item(s) for s in stuck_in]
    if stuck_out != stuck_in:
        course["stuck"] = dedupe_topics(stuck_out, limit=5)
        notes.append("scrubbed filename quotes in stuck")

    code = (course.get("code") or "").strip()
    title = (course.get("title") or "").strip()
    tut_in = [s for s in (course.get("tutoring") or []) if isinstance(s, dict)]
    tut_out = [scrub_tutoring_item(s, code, title) for s in tut_in]
    if tut_out != tut_in:
        course["tutoring"] = tut_out
        notes.append("scrubbed assessment filenames in tutoring")

    for key in ("overview_en", "overview_cn", "lead_en", "lead_cn"):
        raw = course.get(key) or ""
        cleaned = scrub_overview_text(raw) if key.startswith("overview") else (raw or "")
        if BANNED_COPY_RE.search(cleaned):
            cleaned = INVENTORY_OVERVIEW_RE.sub("", cleaned).strip()
        if cleaned != raw:
            course[key] = cleaned
            notes.append(f"scrubbed inventory phrasing in {key}")

    meta = course.get("meta_description_en") or ""
    if has_filename_quote(meta) or re.search(
        r"(?i)assignment briefs such as|Midterm\+|Tutorial\+|AY\d|S1\+|S2\+", meta
    ):
        hook = "assignment briefs under deadline"
        if course.get("topics"):
            hook = course["topics"][0].get("en") or hook
        st = title if len(title) <= 36 else (title[:33].rsplit(" ", 1)[0] + "…")
        head = f"{code} tutoring ({st})" if code else f"{st} tutoring"
        course["meta_description_en"] = f"{head}. Stuck on {hook}? Bilingual Chinese/English help."[:155]
        notes.append("scrubbed filenames in meta")

    return notes


def scrub_all_public_copy(themes: dict[str, dict], write: bool, only_uni: Optional[Set[str]] = None) -> dict:
    stats = {"courses": 0, "files": 0, "notes": []}
    for path in sorted(COURSES_DIR.glob("*.yml")):
        if only_uni and path.stem not in only_uni:
            continue
        data = load_admin(path)
        changed = False
        for course in data.get("courses") or []:
            if not isinstance(course, dict):
                continue
            topics = [t for t in (course.get("topics") or []) if isinstance(t, dict) and t.get("en")]
            needs = (
                public_copy_is_leaky(course)
                or any(is_filename_topic(t.get("en") or "") for t in topics)
                or (bool(course.get("lead_en")) and 0 < len(topics) < 3)
            )
            if not needs:
                continue
            before = yaml.safe_dump(course, allow_unicode=True, sort_keys=False)
            notes = scrub_course_public_copy(course, themes)
            after = yaml.safe_dump(course, allow_unicode=True, sort_keys=False)
            if before != after:
                changed = True
                stats["courses"] += 1
                label = course.get("code") or course.get("title") or "?"
                stats["notes"].append(f"{path.stem}/{label}: {', '.join(notes) or 'updated'}")
        if changed:
            stats["files"] += 1
            if write:
                path.write_text(dump_admin(data), encoding="utf-8")
    return stats


def copy_needs_refresh(course: dict, force: bool = False) -> bool:
    """True if empty lead, banned inventory phrasing in lead/overview, or --force.

    Filename topic chips are fixed via scrub_course_public_copy / --scrub-public-copy
    so we do not overwrite curated leads just to clean chips.
    """
    if force:
        return True
    lead = (course.get("lead_en") or "").strip()
    blob = " ".join(
        str(course.get(k) or "")
        for k in ("lead_en", "lead_cn", "overview_en", "overview_cn")
    )
    if BANNED_COPY_RE.search(blob) or INVENTORY_OVERVIEW_RE.search(blob):
        return True
    return not lead


def process_uni(
    path: Path,
    themes: dict,
    write: bool,
    only_codes: Optional[Set[str]],
    force: bool = False,
) -> dict:
    data = load_admin(path)
    uni_slug = path.stem
    uni_label = data.get("university") or uni_slug
    uni_cn = data.get("name_cn") or uni_label
    tf = (data.get("teaching_folder") or uni_slug).strip()
    root = TEACHING / tf
    stats = {
        "uni": uni_slug,
        "enriched": 0,
        "scrubbed": 0,
        "skipped_curated": 0,
        "skipped_no_material": 0,
        "missing_folder": 0,
        "fixes": [],
        "courses": [],
    }
    if not root.is_dir():
        return stats

    changed = False
    courses = data.get("courses") or []
    for course in courses:
        if not isinstance(course, dict):
            continue
        material = (course.get("material") or "").strip()
        if not material:
            stats["skipped_no_material"] += 1
            continue
        code0 = (course.get("code") or "").strip()
        if only_codes and code0 not in only_codes and code0.replace(" ", "") not in only_codes:
            continue

        # Light scrub for filename chips even when lead is curated.
        if not force and public_copy_is_leaky(course) and not copy_needs_refresh(course, force=False):
            notes = scrub_course_public_copy(course, themes)
            if notes:
                changed = True
                stats["scrubbed"] += 1
                stats["fixes"].extend([f"{code0 or course.get('title')}: {n}" for n in notes])
            stats["skipped_curated"] += 1
            continue

        if not copy_needs_refresh(course, force=force):
            stats["skipped_curated"] += 1
            continue

        mpath = resolve_material_path(root, material)
        if mpath is None:
            stats["missing_folder"] += 1
            continue

        code, title, fix_notes = fix_identity(uni_slug, course, material)
        # Theme from identity first — file haystacks are noisy (datasets, deps).
        identity_hay = f"{code} {title} {material}"
        theme = normalize_theme(str(course.get("theme") or ""), identity_hay, themes)
        ev = gather_evidence(mpath)
        if ev["n_files"] == 0:
            stats["missing_folder"] += 1
            continue
        # If still general, allow topic hits from materials to nudge theme.
        if theme == "general" and ev.get("topic_hits"):
            theme = ev["topic_hits"][0].get("hint") or theme
            theme = normalize_theme(theme, identity_hay, themes)
        copy = build_copy(uni_label, uni_cn, code, title, theme, ev, themes)

        if write:
            if code and course.get("code") != code:
                course["code"] = code
            if title and course.get("title") != title:
                course["title"] = title
            course["material"] = material
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
            changed = True

        stats["enriched"] += 1
        if fix_notes:
            stats["fixes"].extend([f"{code or title}: {n}" for n in fix_notes])
        stats["courses"].append({
            "code": code,
            "title": title,
            "theme": theme,
            "n_files": ev["n_files"],
            "weeks": ev.get("week_span"),
            "assn": len(ev.get("assn_files") or []),
            "fixes": fix_notes,
        })

    if write and changed:
        path.write_text(dump_admin(data), encoding="utf-8")
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--force", action="store_true", help="Overwrite existing lead_en")
    ap.add_argument(
        "--scrub-public-copy",
        action="store_true",
        help="Drop week-filename topics and inventory leaks without rewriting leads",
    )
    ap.add_argument("--uni", action="append", default=[], help="Limit to admin slug(s)")
    ap.add_argument("--code", action="append", default=[], help="Limit to course code(s)")
    args = ap.parse_args()

    themes = load_themes()
    only = set(args.uni) if args.uni else None
    only_codes = {c.replace(" ", "") for c in args.code} | set(args.code) if args.code else None

    if args.scrub_public_copy:
        st = scrub_all_public_copy(themes, write=args.write, only_uni=only)
        mode = "WROTE" if args.write else "DRY-RUN"
        print(f"\n{mode} scrub: {st['courses']} courses across {st['files']} files")
        for note in st["notes"][:80]:
            print(f"  {note}")
        if len(st["notes"]) > 80:
            print(f"  … {len(st['notes']) - 80} more")
        return 0

    paths = sorted(COURSES_DIR.glob("*.yml"))
    if only:
        paths = [p for p in paths if p.stem in only]

    total_enr = 0
    total_fix = []
    for path in paths:
        # Skip monash if already curated unless explicitly requested
        if path.stem == "monash" and not only and not args.force:
            continue
        st = process_uni(
            path, themes, write=args.write, only_codes=only_codes, force=args.force
        )
        if st["enriched"] or st["fixes"]:
            print(f"\n== {st['uni']} enriched={st['enriched']} missing={st['missing_folder']} ==")
            for c in st["courses"]:
                print(f"  {c['code'] or '?':16} theme={c['theme']:16} n={c['n_files']:4} weeks={c['weeks']} assn={c['assn']} | {c['title'][:50]}")
                for f in c["fixes"]:
                    print(f"    fix: {f}")
            total_enr += st["enriched"]
            total_fix.extend(st["fixes"])

    mode = "WROTE" if args.write else "DRY-RUN"
    print(f"\n{mode}: enriched {total_enr} courses; {len(total_fix)} identity fixes noted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
