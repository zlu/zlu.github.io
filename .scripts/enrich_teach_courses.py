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

    # Week-labelled basenames as soft topics
    week_labels = []
    for rel in lect_files + rels:
        m = WEEK_RE.search(rel)
        if not m:
            continue
        label = clean_token(Path(rel).name)
        label = re.sub(r"(?i)week\s*\d+\s*", "", label).strip(" -_:")
        if len(label) >= 4:
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


# Filename leftovers that must never become public "topics" (inventory leak + junk UX).
WEEK_LABEL_JUNK_RE = re.compile(
    r"(?i)^(?:"
    r"lecture\s*slides?|slides?|lectures?|notes?|handouts?|"
    r"lab\s*\d*\s*\(?\)?|prac(?:tical)?s?\s*\d*|"
    r"tutorial\s*\d*|workshop\s*\d*|seminar\s*\d*|"
    r"pdf|pptx?|docx?|"
    r"[A-Z]{2,6}\s*\d{3,5}[A-Z]?"  # bare course codes e.g. COMP2200
    r")$",
)


def usable_week_label(label: str) -> bool:
    """Keep week labels only when they name a real subject, not a file type."""
    s = (label or "").strip(" -_:()[]")
    if len(s) < 6:
        return False
    if WEEK_LABEL_JUNK_RE.match(s):
        return False
    # Mostly punctuation / digits left after stripping week token
    if re.fullmatch(r"[\d\s\-_()./]+", s):
        return False
    return True


def build_topics(ev: dict, theme: str, themes: dict[str, dict]) -> list[dict]:
    topics: list[dict] = []
    for hit in ev.get("topic_hits") or []:
        topics.append(bilingual(hit["en"], hit["cn"]))
    for week, label in ev.get("week_labels") or []:
        if not usable_week_label(label):
            continue
        short = label if len(label) <= 60 else label[:57] + "…"
        topics.append(bilingual(f"Week {week}: {short}", f"第 {week} 周：{short}"))
    if ev.get("has_notebooks"):
        topics.append(bilingual(
            "Notebook workflows the marker can re-run",
            "改卷人能复跑的 notebook 工作流",
        ))
    if ev.get("assn_labels"):
        topics.append(bilingual(
            "Assignment briefs mapped to a step-by-step plan",
            "把作业 brief 映射成可执行步骤",
        ))
    # Prefer theme topics over thin filename leftovers
    if len(topics) < 4:
        for t in (themes.get(theme) or {}).get("topics") or []:
            if isinstance(t, dict) and t.get("en"):
                topics.append(bilingual(t["en"], t.get("cn") or t["en"]))
            if len(topics) >= 6:
                break
    # Dedup by en
    out, seen = [], set()
    for t in topics:
        k = t["en"].lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(t)
        if len(out) >= 8:
            break
    return out


def build_stuck(ev: dict, theme: str, themes: dict[str, dict], code: str) -> list[dict]:
    stuck: list[dict] = []
    handle = code or "this course"
    if ev.get("week_span"):
        stuck.append(bilingual(
            f"Weekly lecture arc moves on before last week’s method is solid",
            f"周课往前赶，上一周的方法还没练稳",
        ))
    if ev.get("assn_labels") or ev.get("assn_files"):
        sample = (ev.get("assn_labels") or ["the assignment"])[0]
        stuck.append(bilingual(
            f"Assignment brief (“{sample[:48]}”) makes sense in class, freezes you alone",
            f"作业 brief（「{sample[:48]}」）课上听得懂，自己对着就停住",
        ))
    if ev.get("has_notebooks"):
        stuck.append(bilingual(
            "Notebook runs on the sample cell, fails on the hidden / full dataset case",
            "notebook 在样例 cell 能过，全量 / 隐藏数据就挂",
        ))
    if ev.get("has_code") and theme in {"programming", "algorithms", "systems", "security"}:
        stuck.append(bilingual(
            "Code almost works; edge cases and error messages have no debug method",
            "代码差不多能跑，边界和报错却没有 debug 方法",
        ))
    if theme in {"math-for-cs", "automata", "algorithms"}:
        stuck.append(bilingual(
            "Can follow a worked example, cannot start from a blank page under time pressure",
            "能看懂例题，限时白纸上却开不了头",
        ))
    if theme in {"data-science", "machine-learning"}:
        stuck.append(bilingual(
            "Tool output produced, write-up cannot defend method choice",
            "工具输出有了，报告却说不清为何选这个方法",
        ))
    if theme == "security":
        stuck.append(bilingual(
            "Lab setup unfinished; report lists tools without a clear finding",
            "Lab 环境还没配好；报告只罗列工具没有清晰发现",
        ))
    # Theme defaults
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
    assn_bit_en = "approach review and live debug against the real brief"
    assn_bit_cn = "对着真实 brief 拆题、梳思路、现场 debug"
    if ev.get("assn_labels"):
        labels = ", ".join(ev["assn_labels"][:2])
        assn_bit_en = f"approach review for assessments such as {labels}, with you at the keyboard"
        assn_bit_cn = f"针对 {labels} 一类评估拆题，你来操作键盘"
    return [
        bilingual(
            f"Lecture sync for {handle}: bilingual catch-up so workshop / tutorial language matches the lectures.",
            f"按 {handle} 当周进度跟课：用中英双语把没听清的概念补上，并保留 workshop / tutorial 会用的英文说法。",
        ),
        bilingual(
            f"Assignments: {assn_bit_en}.",
            f"作业辅导：{assn_bit_cn}。",
        ),
        bilingual(
            "Exam / quiz prep: timed drills on weak spots and recurring question shapes for this course.",
            "考试 / quiz 突击：按这门课常考题型和你的薄弱点做限时练习。",
        ),
    ]


def evidence_phrase(ev: dict) -> tuple[str, str]:
    bits_en, bits_cn = [], []
    if ev.get("week_span"):
        a, b = ev["week_span"]
        bits_en.append(f"weekly notes spanning roughly week {a}–{b}")
        bits_cn.append(f"大约第 {a}–{b} 周的周课笔记")
    if ev.get("assn_labels"):
        bits_en.append("assignment briefs / labs on disk")
        bits_cn.append("磁盘上的作业 brief / lab")
    elif ev.get("assn_files"):
        bits_en.append("named assignment / lab files")
        bits_cn.append("点名的作业 / lab 文件")
    if ev.get("has_notebooks"):
        bits_en.append("notebooks")
        bits_cn.append("notebook")
    if ev.get("has_code"):
        bits_en.append("code submissions / practice files")
        bits_cn.append("代码提交 / 练习文件")
    if not bits_en:
        bits_en.append("lecture and practice materials")
        bits_cn.append("讲座与练习材料")
    return (", ".join(bits_en), "、".join(bits_cn))


def build_copy(uni_label: str, uni_cn: str, code: str, title: str, theme: str, ev: dict, themes: dict[str, dict]) -> dict:
    hard = THEME_HARD.get(theme) or THEME_HARD["general"]
    if code and title and code.replace(" ", "").lower() == title.replace(" ", "").lower():
        label = title
    else:
        label = " ".join(x for x in [code, title] if x).strip() or title or code
    cn_handle = code or title
    ev_en, ev_cn = evidence_phrase(ev)
    theme_label_en = (themes.get(theme) or {}).get("label_en") or theme
    theme_label_cn = (themes.get(theme) or {}).get("label_cn") or theme

    topic_ens = [t["en"] for t in (ev.get("topic_hits") or [])][:4]
    topic_cns = [t["cn"] for t in (ev.get("topic_hits") or [])][:3]

    # Never claim we hold/redistribute university lecture notes or course packs.
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

    overview_en = f"{label} ({theme_label_en}) at {uni_label}. "
    if topic_ens:
        overview_en += f"Common topics: {'; '.join(topic_ens)}. "
    if ev.get("week_span"):
        a, b = ev["week_span"]
        overview_en += f"Weekly arc about week {a}–{b}, plus assessed work under deadline. "
    elif ev.get("assn_files"):
        overview_en += "Assessed work expects you to apply the same ideas under deadline. "
    overview_en += (
        "People stall when terms outrun notes, when a class demo does not replay alone, "
        "or when the write-up / autograder wants a method the notebook never showed."
    )

    overview_cn = (
        f"{cn_handle} 是{uni_cn}的「{theme_label_cn}」课，既要真懂，也要能动手。"
    )
    if topic_cns:
        overview_cn += f"常见内容包括{'、'.join(topic_cns)}。"
    if ev.get("week_span"):
        a, b = ev["week_span"]
        overview_cn += f"大约第 {a}–{b} 周有连续周课，作业会要求你在 DDL 下用同一套方法。"
    overview_cn += (
        "很多人不是完全没学过，而是课上术语太快、自己复不出来，"
        "或报告 / autograder 要的方法在 notebook 里从没写清。"
    )

    topics = build_topics(ev, theme, themes)
    stuck = build_stuck(ev, theme, themes, code)
    session_angles = build_session_angles(ev, theme, themes)
    tutoring = build_tutoring(code, title, ev)

    # SERP meta: code + short uni + concrete stuck hook (≤155; plugin truncates at 160).
    uni_short = uni_label.split()[0] if len(uni_label) > 22 else uni_label
    if ev.get("assn_labels"):
        hook = f"assignment briefs such as {ev['assn_labels'][0][:40]}"
    elif ev.get("topic_hits"):
        hook = ev["topic_hits"][0]["en"]
    elif topic_ens:
        hook = topic_ens[0]
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
    r"Private materials|Evidence from the|现有材料|其余讲义|领取完整|"
    r"full course pack|we have the lecture|download the notes",
    re.I,
)


def copy_needs_refresh(course: dict, force: bool = False) -> bool:
    """True if empty lead, banned inventory phrasing, or --force."""
    if force:
        return True
    lead = (course.get("lead_en") or "").strip()
    blob = " ".join(
        str(course.get(k) or "")
        for k in ("lead_en", "lead_cn", "overview_en", "overview_cn")
    )
    if BANNED_COPY_RE.search(blob):
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
        if not copy_needs_refresh(course, force=force):
            stats["skipped_curated"] += 1
            continue
        code0 = (course.get("code") or "").strip()
        if only_codes and code0 not in only_codes and code0.replace(" ", "") not in only_codes:
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
    ap.add_argument("--uni", action="append", default=[], help="Limit to admin slug(s)")
    ap.add_argument("--code", action="append", default=[], help="Limit to course code(s)")
    args = ap.parse_args()

    themes = load_themes()
    only = set(args.uni) if args.uni else None
    only_codes = {c.replace(" ", "") for c in args.code} | set(args.code) if args.code else None

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
