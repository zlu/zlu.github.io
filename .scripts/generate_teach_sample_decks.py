#!/usr/bin/env python3
"""Generate concept-review sample slides for teach course pages.

Priority:
  1. Existing tutoring Marp in teaching/<folder>/<material>/decks/
     (anything except tutoring-intro / sales course-intro shells)
  2. Branded theme concept pack from marp/theme-packs/<theme>.md

Never emits outline + "how tutoring runs" shells.
Never claims university lecture notes. Framing is always our tutoring deck.

Usage:
  python3 .scripts/generate_teach_sample_decks.py
  python3 .scripts/generate_teach_sample_decks.py --write
  python3 .scripts/generate_teach_sample_decks.py --write --include-existing
  python3 .scripts/generate_teach_sample_decks.py --write --force-course COMP9414
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
    LECT_HINT_RE,
    TEACHING,
    TOPIC_LEXICON,
    WEEK_RE,
    clean_token,
    load_themes,
    normalize_theme,
    resolve_material_path,
    scan_files,
)

COURSES_DIR = ROOT / "_data" / "courses"
MARP = ROOT / "marp" / "zlu-marp"
THEME_PACKS = ROOT / "marp" / "theme-packs"
ASSETS = ROOT / "assets" / "teach"

MIN_FILES_DEFAULT = 3

# Already content-led on the site; skip unless --force-course matches.
PROTECTED_CODES = {
    "COMP3011",
    "COMP SCI 4023",
    "COMPSCI4023",
    "CIVE70107",
    "FIT5196",
    "DDM 理学硕士笔试",
    "DDM",
}

INTRO_NAME_RE = re.compile(
    r"(?i)^(tutoring-intro|course-intro|concept-mini)(\.|$)|tutoring\s*intro"
)
LOGISTICS_RE = re.compile(
    r"(?i)how tutoring runs|where people get stuck|tutoring plan|"
    r"what we cover|session logistics|catch-up · debug|bring to session|"
    r"tutoring focus|course intro —|why the pace|how we structure tutoring"
)
CONTENTISH_RE = re.compile(
    r"(?i)\b(definition|define|theorem|lemma|example|trap|pitfall|habit|"
    r"formula|algorithm|proof|model|tuple|lemma|checklist|method|"
    r"interleaving|sigmoid|gradient|pointer|heap|stack|join|index|"
    r"dfa|nfa|petri|bayes|mutex|race|deadlock|bfs|dfs|dp\b|"
    r"encrypt|hash|threat|socket|page fault|expectation)"
)
SECTION_RE = re.compile(r"(?i)^part\s+[a-z]\b|^roadmap\b")


def ascii_slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower())
    return re.sub(r"-+", "-", s).strip("-")


def load_admin(path: Path) -> dict:
    docs = list(yaml.safe_load_all(path.read_text()))
    return next(d for d in docs if isinstance(d, dict))


def dump_admin(doc: dict) -> str:
    body = yaml.safe_dump(doc, allow_unicode=True, sort_keys=False, width=100).rstrip() + "\n"
    return f"---\n{body}---\n"


def gather_evidence_light(material_path: Path) -> dict:
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
        if len(label) >= 4 and not re.search(r"(?i)^lecture\s*notes?", label):
            week_labels.append((int(m.group(1)), label))
    week_labels = sorted(set(week_labels))[:10]
    return {
        "n_files": len(files),
        "weeks": weeks,
        "tops": tops,
        "assn_files": assn_files[:15],
        "lect_files": lect_files[:15],
        "week_labels": week_labels,
        "topic_hits": topic_hits,
        "hay": hay[:4000],
    }


def fast_content_count(material_path: Path, limit: int = 600) -> int:
    return len(scan_files(material_path, limit=limit))


def is_intro_shell_sample(course: dict) -> bool:
    ss = course.get("sample_slides") or {}
    if not ss:
        return False
    lead = str(ss.get("lead_en") or "")
    caps = " ".join(str(s.get("caption_en") or "") for s in (ss.get("slides") or []) if isinstance(s, dict))
    blob = f"{lead}\n{caps}"
    if "intro deck" in lead.lower() or "course-intro" in lead.lower() or "how sessions run" in lead.lower():
        return True
    if LOGISTICS_RE.search(blob) and "Part A" not in caps and "Interleaving" not in caps:
        if any(x in caps for x in ("Stuck", "How tutoring", "Topics", "Tutoring plan", "Difficulty")):
            return True
    srcs = " ".join(str(s.get("src") or "") for s in (ss.get("slides") or []) if isinstance(s, dict))
    if "03-stuck" in srcs and "04-tutoring" in srcs:
        return True
    return False


def has_wrong_lang_c_sample(course: dict, lang: str) -> bool:
    """True if samples still show C memory while the course is not C/C++."""
    if lang in {"c", "cpp"}:
        return False
    ss = course.get("sample_slides") or {}
    caps = " ".join(str(s.get("caption_en") or "") for s in (ss.get("slides") or []) if isinstance(s, dict))
    srcs = " ".join(str(s.get("src") or "") for s in (ss.get("slides") or []) if isinstance(s, dict))
    return "Stack, heap, data (C)" in caps or "stack-heap-data-c" in srcs


def code_matches_force(code: str, force: set[str]) -> bool:
    if not force:
        return False
    c = code.strip()
    c_compact = c.replace(" ", "")
    for f in force:
        if f == c or f == c_compact or f.lower() == c.lower():
            return True
        if f.lower() in c.lower() or c.lower() in f.lower():
            return True
    return False


def is_protected(code: str, force: set[str]) -> bool:
    if code_matches_force(code, force):
        return False
    c = code.strip()
    if c in PROTECTED_CODES:
        return True
    compact = c.replace(" ", "")
    return compact in {p.replace(" ", "") for p in PROTECTED_CODES}


def list_real_deck_mds(deck_dir: Path) -> list[Path]:
    if not deck_dir.is_dir():
        return []
    out = []
    for p in sorted(deck_dir.glob("*.md")):
        if INTRO_NAME_RE.search(p.name):
            continue
        # Prefer concept / review / lecture-style names; still allow l1-*.md etc.
        out.append(p)
    # Prefer files with concept/review/foundations in the name
    def score(p: Path) -> tuple:
        n = p.name.lower()
        return (
            0 if re.search(r"concept|review|foundation|wrangling|construct", n) else 1,
            0 if re.match(r"l\d", n) else 1,
            n,
        )
    return sorted(out, key=score)


def page_texts(pdf_path: Path) -> list[str]:
    import fitz

    doc = fitz.open(pdf_path)
    return [doc[i].get_text("text") for i in range(doc.page_count)]


def score_page(text: str, index: int) -> float:
    t = text or ""
    head = "\n".join(t.splitlines()[:8])
    if LOGISTICS_RE.search(t):
        return -100.0
    score = 0.0
    if index == 0:
        score += 40.0  # always keep cover as candidate
    if SECTION_RE.search(head) or re.search(r"(?i)^roadmap", head):
        score += 25.0
    if CONTENTISH_RE.search(t):
        score += 30.0
    if re.search(r"(?i)\b(trap|pitfall|habit|tip)\b", t):
        score += 12.0
    if re.search(r"\$|\\\\begin\{|\\frac", t):
        score += 8.0
    if re.search(r"(?i)^#?\s*part\s+[a-z]", head):
        score += 5.0
    # Prefer mid-deck teaching over last "book tutoring" end cards
    if re.search(r"(?i)book tutoring|wechat|end of deck", t):
        score -= 20.0
    score += max(0, 6 - index) * 0.3
    return score


def pick_sample_pages(texts: list[str], limit: int = 6) -> list[int]:
    """Return 0-based page indices: cover + roadmap + content pages."""
    if not texts:
        return []
    n = len(texts)
    scored = [(score_page(texts[i], i), i) for i in range(n)]
    scored.sort(reverse=True)

    chosen: list[int] = []
    # Force cover
    chosen.append(0)

    # Best roadmap / section among early pages
    roadmap_candidates = [
        i for s, i in scored if i != 0 and (SECTION_RE.search("\n".join((texts[i] or "").splitlines()[:6])) or re.search(r"(?i)roadmap", texts[i] or ""))
    ]
    if roadmap_candidates:
        # earliest high-scoring roadmap-ish
        roadmap_candidates.sort(key=lambda i: (-score_page(texts[i], i), i))
        if roadmap_candidates[0] not in chosen:
            chosen.append(roadmap_candidates[0])

    for _s, i in scored:
        if i in chosen:
            continue
        if score_page(texts[i], i) < 10:
            continue
        if LOGISTICS_RE.search(texts[i] or ""):
            continue
        chosen.append(i)
        if len(chosen) >= limit:
            break

    # Fill if thin
    for i in range(n):
        if len(chosen) >= min(limit, n):
            break
        if i not in chosen and not LOGISTICS_RE.search(texts[i] or ""):
            chosen.append(i)

    chosen = sorted(set(chosen))[:limit]
    # Ensure cover first in display order by sorting then rotating? Better: sort by index for narrative
    return sorted(chosen)[:limit]


def caption_from_text(text: str, index: int) -> tuple[str, str]:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    # Drop footer-ish short lines
    lines = [ln for ln in lines if not re.match(r"(?i)^zlu\b", ln) and not re.fullmatch(r"\d+", ln)]
    title = lines[0] if lines else f"Slide {index + 1}"
    title = re.sub(r"^#+\s*", "", title)
    if len(title) > 48:
        title = title[:45].rsplit(" ", 1)[0] + "…"
    if index == 0:
        return "Title", "封面"
    if re.search(r"(?i)roadmap", title):
        return "Roadmap", "路线图"
    if re.search(r"(?i)^part\s+([a-z])\b", title):
        m = re.search(r"(?i)^part\s+([a-z].*)$", title)
        label = m.group(0) if m else title
        return label[:48], label[:48]
    return title, title


def slug_stem(caption_en: str, index_in_sample: int) -> str:
    base = ascii_slug(caption_en) or "slide"
    base = base[:40]
    return f"{index_in_sample + 1:02d}-{base}"


def export_pdf(md_path: Path, pdf_path: Path) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(MARP), str(md_path), "-o", str(pdf_path)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def render_selected_webps(pdf_path: Path, out_dir: Path, page_indices: list[int], stems: list[str]) -> None:
    import fitz
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*"):
        if old.is_file():
            old.unlink()

    doc = fitz.open(pdf_path)
    mat = fitz.Matrix(2.0, 2.0)
    for idx, stem in zip(page_indices, stems):
        if idx >= doc.page_count:
            continue
        pix = doc[idx].get_pixmap(matrix=mat, alpha=False)
        png = out_dir / f"{stem}.png"
        webp = out_dir / f"{stem}.webp"
        pix.save(str(png))
        Image.open(png).save(webp, "WEBP", quality=82, method=6)
        png.unlink()


PROGRAMMING_LANG_BLOCKS: dict[str, dict[str, str]] = {
    "java": {
        "COVER_PARTS": "BRIEF · STRUCTURE · RECURSION · OOP",
        "ROADMAP_D_TITLE": "Java OOP — inheritance / interfaces",
        "ROADMAP_D_DESC": "Override, polymorphism, where behaviour lives",
        "PART_D_SECTION_TITLE": "Part D · Java OOP",
        "PART_D_SECTION_SUB": "Inheritance, interfaces, polymorphism",
        "PART_D_SLIDE": """# Inheritance, interface, polymorphism

<p class="eyebrow">D1 · OO choices you must justify</p>

| Idea | Plain meaning |
|---|---|
| **Inheritance** | child class reuses / specialises a parent |
| **Interface** | a contract of methods; class **implements** it |
| **Overload** | same name, different parameter lists |
| **Override** | child replaces a parent instance method |
| **Polymorphism** | call site uses a parent / interface type; runtime picks the child body |

```java
Animal a = new Dog();  // compile-time type Animal
a.speak();             // runs Dog.speak if overridden
```

Put shared menu / I/O in one place; put domain behaviour on the class that owns the data.

<div class="pitfall">
<strong>Trap.</strong> A “parent” that is only a bag of unrelated fields. Inheritance needs a real is-a story the write-up can defend.
</div>
""",
    },
    "python": {
        "COVER_PARTS": "BRIEF · STRUCTURE · RECURSION · NAMES",
        "ROADMAP_D_TITLE": "Python names, objects, mutability",
        "ROADMAP_D_DESC": "Assignment, aliases, what mutates in place",
        "PART_D_SECTION_TITLE": "Part D · Python objects",
        "PART_D_SECTION_SUB": "Names versus objects",
        "PART_D_SLIDE": """# Names, objects, mutability

<p class="eyebrow">D1 · What assignment actually does</p>

A name refers to an object. `b = a` makes another name for the **same** object (alias), it does not copy.

| Kind | Examples | `b = a` then mutate |
|---|---|---|
| Immutable | `int`, `str`, `tuple` | rebinding `a` leaves `b` on the old object |
| Mutable | `list`, `dict`, `set` | in-place change via `a` is visible through `b` |

```python
a = [1, 2]
b = a
a.append(3)   # b is [1, 2, 3] too
c = a[:]      # shallow copy — new list
```

Default args are evaluated once: never use `def f(x=[])`.

<div class="pitfall">
<strong>Trap.</strong> Treating `=` as a deep copy. Autograders love alias bugs on nested lists / dicts.
</div>
""",
    },
    "javascript": {
        "COVER_PARTS": "BRIEF · STRUCTURE · RECURSION · OBJECTS",
        "ROADMAP_D_TITLE": "JS objects, references, modules",
        "ROADMAP_D_DESC": "Mutable objects, `const`, how modules share state",
        "PART_D_SECTION_TITLE": "Part D · JavaScript objects",
        "PART_D_SECTION_SUB": "References and module boundaries",
        "PART_D_SLIDE": """# Objects are references

<p class="eyebrow">D1 · What `const` does not freeze</p>

| Binding | Meaning |
|---|---|
| `const obj = {...}` | cannot rebind `obj`; **can** mutate properties |
| `let` / `var` | can rebind the name |
| Spread / `Object.assign` | shallow copy only |

```js
const a = { n: 1 };
const b = a;
a.n = 2;        // b.n is 2
const c = { ...a }; // shallow clone
```

Keep DOM / I/O at the edges; keep domain state in plain objects or a small module with a clear public API.

<div class="pitfall">
<strong>Trap.</strong> Mutating a shared object across modules, then wondering why another screen “randomly” changes.
</div>
""",
    },
    "c": {
        "COVER_PARTS": "BRIEF · STRUCTURE · RECURSION · MEMORY",
        "ROADMAP_D_TITLE": "Tests and memory (C / pointers)",
        "ROADMAP_D_DESC": "Autograder loops; stack vs heap",
        "PART_D_SECTION_TITLE": "Part D · Tests and memory",
        "PART_D_SECTION_SUB": "Autograder loops and C storage",
        "PART_D_SLIDE": """# Stack, heap, data (C)

<p class="eyebrow">D1 · Where values live</p>

| Region | Holds | Lifetime |
|---|---|---|
| **Stack** | locals, parameters, return addr | until the function returns |
| **Heap** | `malloc` / `new` blocks | until `free` / `delete` |
| **Data / BSS** | globals, `static` | whole program |

```c
int x = 10;      // data
int *p = &x;     // p on stack; stores address of x
int *h = malloc(sizeof(int));  // h on stack; *h on heap
```

Pointer arithmetic scales by `sizeof(*p)`. Off-by-one and use-after-free are undefined behaviour, not “sometimes works”.

<div class="pitfall">
<strong>Exam trap.</strong> Returning the address of a local stack variable is a dangling pointer the moment the function returns.
</div>
""",
    },
    "cpp": {
        "COVER_PARTS": "BRIEF · STRUCTURE · RECURSION · MEMORY",
        "ROADMAP_D_TITLE": "C++ ownership and RAII",
        "ROADMAP_D_DESC": "new/delete vs stack; who frees what",
        "PART_D_SECTION_TITLE": "Part D · C++ memory",
        "PART_D_SECTION_SUB": "Ownership and lifetimes",
        "PART_D_SLIDE": """# Ownership in C++

<p class="eyebrow">D1 · Who destroys the object</p>

| Pattern | Meaning |
|---|---|
| Stack object | destructor runs at scope end |
| `new` / `delete` | you own the lifetime; pair them |
| RAII / smart pointer | resource tied to object lifetime |

```cpp
std::vector<int> v{1,2,3};     // frees itself
auto* p = new int(3);
delete p;                      // required
```

Prefer containers and RAII over raw `new` in coursework unless the brief forces it.

<div class="pitfall">
<strong>Trap.</strong> Double-delete or using a pointer after `delete`. Markers hit both.
</div>
""",
    },
    "general": {
        "COVER_PARTS": "BRIEF · STRUCTURE · RECURSION · TESTS",
        "ROADMAP_D_TITLE": "Tests and autograder loops",
        "ROADMAP_D_DESC": "Minimal failing case; one change per experiment",
        "PART_D_SECTION_TITLE": "Part D · Tests",
        "PART_D_SECTION_SUB": "Autograder feedback loops",
        "PART_D_SLIDE": """# Autograder loop

<p class="eyebrow">D1 · Method</p>

1. Reproduce with the **smallest** failing input
2. Read the expected vs actual output carefully (whitespace, order)
3. Change **one** hypothesis; re-run
4. Add a regression test for that edge before moving on

| Anti-pattern | Replace with |
|---|---|
| Fix three things at once | one change per run |
| Only test the sample | invent empty / one / max / duplicate cases |

<div class="tip">
<strong>Habit.</strong> Keep a failing case in a file you can re-run in one command.
</div>
""",
    },
}


def detect_programming_lang(mpath: Path, course_blob: str = "") -> str:
    """Infer primary language for programming theme packs."""
    blob = (course_blob or "").lower()
    counts = {"java": 0, "python": 0, "javascript": 0, "c": 0, "cpp": 0}
    if mpath and mpath.is_dir():
        for p in scan_files(mpath, limit=500):
            suf = p.suffix.lower()
            if suf == ".java":
                counts["java"] += 3
            elif suf == ".py":
                counts["python"] += 3
            elif suf in {".js", ".jsx", ".ts", ".tsx"}:
                counts["javascript"] += 3
            elif suf in {".cpp", ".cc", ".cxx", ".hpp"}:
                counts["cpp"] += 3
            elif suf == ".c" or suf == ".h":
                counts["c"] += 2
    # Text hints (course copy / title)
    if re.search(r"\bjava\b|object[- ]oriented|oop\b", blob):
        counts["java"] += 5
    if re.search(r"\bpython\b|pandas|numpy|jupyter", blob):
        counts["python"] += 5
    if re.search(r"javascript|\bjs\b|typescript|node\.?js|react", blob):
        counts["javascript"] += 5
    if re.search(r"c\+\+|cpp\b", blob):
        counts["cpp"] += 5
    if re.search(r"\bpointers?\b|\bmalloc\b|systems programming|\bin c\b", blob):
        counts["c"] += 4
    best = max(counts, key=counts.get)
    if counts[best] <= 0:
        return "general"
    return best


def brand_theme_pack(
    theme: str,
    code: str,
    title: str,
    uni_label: str,
    lang: str = "general",
) -> str:
    path = THEME_PACKS / f"{theme}.md"
    if not path.exists():
        path = THEME_PACKS / "general.md"
    raw = path.read_text(encoding="utf-8")
    handle = code or title
    short_title = title if len(title) <= 52 else title[:49].rsplit(" ", 1)[0] + "…"
    uni = (uni_label or "").upper()[:28]
    footer = f"zlu · {handle}"
    out = (
        raw.replace("{{CODE}}", handle)
        .replace("{{TITLE}}", short_title)
        .replace("{{UNI}}", uni)
        .replace("{{FOOTER}}", footer)
    )
    if theme == "programming":
        block = PROGRAMMING_LANG_BLOCKS.get(lang) or PROGRAMMING_LANG_BLOCKS["general"]
        for key, val in block.items():
            out = out.replace("{{" + key + "}}", val)
    return out


def bullets_from_topics(ev: dict, themes: dict, theme: str, limit: int = 4) -> list[dict]:
    out = []
    for h in (ev.get("topic_hits") or [])[:limit]:
        out.append({"en": h["en"], "cn": h.get("cn") or h["en"]})
    if len(out) >= 2:
        return out[:limit]
    meta = themes.get(theme) or {}
    for t in (meta.get("topics") or [])[:limit]:
        if isinstance(t, dict) and t.get("en"):
            out.append({"en": t["en"], "cn": t.get("cn") or t["en"]})
    return out[:limit]


def sample_slides_block(
    code: str,
    title: str,
    asset_prefix: str,
    stems: list[str],
    captions: list[tuple[str, str]],
    bullets: list[dict],
    source_label_en: str,
) -> dict[str, Any]:
    handle = code or title
    slides = []
    for stem, (cap_en, cap_cn) in zip(stems, captions):
        slides.append(
            {
                "src": f"{asset_prefix}/{stem}.webp",
                "alt_en": f"{handle} tutoring · {cap_en}",
                "alt_cn": f"{handle} 辅导课件 · {cap_cn}",
                "caption_en": cap_en,
                "caption_cn": cap_cn,
            }
        )
    lead_bits = ", ".join(c[0] for c in captions[:6])
    return {
        "title_en": f"{handle} — sample slides",
        "title_cn": f"{handle} — 辅导课件样例",
        "lead_en": f"{lead_bits} from our {source_label_en}.",
        "lead_cn": f"我们概念复习辅导课件中的样例页：{lead_bits}。",
        "bullets": bullets,
        "slides": slides,
        "cta_en": "Book tutoring on WeChat",
        "cta_cn": "微信预约辅导",
    }


def scrub_intro_copy(course: dict) -> None:
    for key in ("overview_en", "overview_cn", "lead_en", "lead_cn"):
        val = course.get(key)
        if not isinstance(val, str):
            continue
        val2 = val
        val2 = re.sub(
            r"\s*Sample slides from our tutoring intro deck are below\.",
            " Sample slides from our concept-review tutoring deck are below.",
            val2,
            flags=re.I,
        )
        val2 = re.sub(
            r"下方有我们辅导入门课件的样例页。",
            "下方有我们概念复习辅导课件的样例页。",
            val2,
        )
        val2 = val2.replace("tutoring intro deck", "concept-review tutoring deck")
        val2 = val2.replace("辅导入门课件", "概念复习辅导课件")
        course[key] = re.sub(r"\s{2,}", " ", val2).strip()

    faqs = course.get("faq") or []
    new_faqs = []
    for f in faqs:
        if not isinstance(f, dict):
            new_faqs.append(f)
            continue
        f = dict(f)
        for k in ("a_en", "a_cn", "q_en", "q_cn"):
            if k in f and isinstance(f[k], str):
                f[k] = (
                    f[k]
                    .replace("tutoring intro deck", "concept-review tutoring deck")
                    .replace("辅导入门课件", "概念复习辅导课件")
                )
        new_faqs.append(f)
    if new_faqs and not any("sample" in str(f.get("q_en", "")).lower() for f in new_faqs if isinstance(f, dict)):
        new_faqs.append(
            {
                "q_en": "Are sample slides available?",
                "q_cn": "有辅导课件样例吗？",
                "a_en": "Yes — sample pages from our concept-review tutoring deck are on this page.",
                "a_cn": "有，本页有我们概念复习辅导课件的样例页。",
            }
        )
    if new_faqs:
        course["faq"] = new_faqs


def ensure_sample_faq(course: dict) -> None:
    faqs = list(course.get("faq") or [])
    if any("sample" in str(f.get("q_en", "")).lower() for f in faqs if isinstance(f, dict)):
        # refresh answers away from intro wording
        for f in faqs:
            if not isinstance(f, dict):
                continue
            if "sample" in str(f.get("q_en", "")).lower():
                f["a_en"] = "Yes — sample pages from our concept-review tutoring deck are on this page."
                f["a_cn"] = "有，本页有我们概念复习辅导课件的样例页。"
        course["faq"] = faqs
        return
    faqs.append(
        {
            "q_en": "Are sample slides available?",
            "q_cn": "有辅导课件样例吗？",
            "a_en": "Yes — sample pages from our concept-review tutoring deck are on this page.",
            "a_cn": "有，本页有我们概念复习辅导课件的样例页。",
        }
    )
    course["faq"] = faqs


def wire_sample(course: dict, block: dict) -> None:
    course["sample_slides"] = block
    scrub_intro_copy(course)
    ov = course.get("overview_en") or ""
    if "sample slide" not in ov.lower() and "tutoring deck" not in ov.lower():
        course["overview_en"] = (
            (ov + " " if ov else "")
            + "Sample slides from our concept-review tutoring deck are below."
        ).strip()
    ov_cn = course.get("overview_cn") or ""
    if "样例" not in ov_cn and "辅导课件" not in ov_cn:
        course["overview_cn"] = (
            (ov_cn + " " if ov_cn else "")
            + "下方有我们概念复习辅导课件的样例页。"
        ).strip()
    ensure_sample_faq(course)


def collect_targets(min_files: int) -> list[dict]:
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
                    "is_intro": is_intro_shell_sample(course),
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
        # Theme packs do not need many files; still skip empty ghosts
        if n_files < min_files and not list_real_deck_mds(mpath / "decks"):
            continue
        refs = sorted(g["refs"], key=lambda r: (0 if r["code"] else 1, r["code"]))
        primary = refs[0]
        code = primary["code"]
        title = primary["title"] or mat
        identity_hay = f"{code} {title} {mat}"
        theme = normalize_theme(str(primary["theme"] or ""), identity_hay, themes)
        course_slug = ascii_slug(code or title) or ascii_slug(mat)[:40]
        asset_folder = ascii_slug(folder) or ascii_slug(refs[0]["yml"].stem) or "teach-misc"
        real_decks = list_real_deck_mds(mpath / "decks")
        admin_blob = f"{code} {title} {mat}"
        try:
            admin = load_admin(primary["yml"])
            for course in admin.get("courses") or []:
                if not isinstance(course, dict):
                    continue
                if (course.get("material") or "").strip() != mat:
                    continue
                admin_blob = " ".join(
                    str(course.get(k) or "")
                    for k in (
                        "code",
                        "title",
                        "lead_en",
                        "overview_en",
                        "meta_description_en",
                        "theme",
                    )
                )
                break
        except Exception:
            pass
        lang = (
            detect_programming_lang(mpath, admin_blob)
            if theme == "programming"
            else "general"
        )
        wrong_lang = False
        if theme == "programming":
            for ref in refs:
                # peek sample from yaml
                try:
                    adm = load_admin(ref["yml"])
                    for course in adm.get("courses") or []:
                        if not isinstance(course, dict):
                            continue
                        if (course.get("material") or "").strip() != mat:
                            continue
                        if has_wrong_lang_c_sample(course, lang):
                            wrong_lang = True
                            break
                except Exception:
                    pass
                if wrong_lang:
                    break
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
                "lang": lang,
                "course_slug": course_slug,
                "refs": refs,
                "themes": themes,
                "already_sampled": any(r["has_sample"] for r in refs),
                "needs_regen": (
                    any(r["is_intro"] for r in refs)
                    or not any(r["has_sample"] for r in refs)
                    or wrong_lang
                ),
                "real_decks": real_decks,
            }
        )
    return sorted(targets, key=lambda t: (-t["n_files"], t["folder"], t["code"]))


def process_target(
    t: dict,
    write: bool,
    skip_existing: bool,
    force: set[str],
) -> dict:
    code, title = t["code"], t["title"]
    handle = code or title
    asset_folder = t["asset_folder"]
    asset_dir = ASSETS / asset_folder / t["course_slug"] / "sample"
    asset_prefix = f"/assets/teach/{asset_folder}/{t['course_slug']}/sample"

    result = {
        "handle": handle,
        "n_files": t["n_files"],
        "folder": t["folder"],
        "material": t["material"],
        "skipped": False,
        "error": None,
        "source": None,
    }

    if is_protected(code, force) and not code_matches_force(code, force):
        result["skipped"] = True
        result["source"] = "protected"
        return result

    # Skip non-intro existing samples unless include-existing / force / needs_regen
    if skip_existing and t["already_sampled"] and not t["needs_regen"] and not code_matches_force(code, force):
        result["skipped"] = True
        result["source"] = "existing-content"
        return result

    ev = gather_evidence_light(t["mpath"])
    theme = t["theme"]
    if theme == "general" and ev.get("topic_hits"):
        theme = ev["topic_hits"][0].get("hint") or theme
        theme = normalize_theme(theme, f"{code} {title}", t["themes"])
    bullets = bullets_from_topics(ev, t["themes"], theme)

    deck_dir = t["mpath"] / "decks"
    source = "theme-pack"
    md_path: Optional[Path] = None
    pdf_path: Optional[Path] = None
    branded_tmp: Optional[Path] = None

    lang = t.get("lang") or "general"
    if theme == "programming" and lang == "general":
        course_blob = f"{code} {title} {t.get('material') or ''}"
        lang = detect_programming_lang(t["mpath"], course_blob)

    real = t.get("real_decks") or []
    if real:
        md_path = real[0]
        pdf_path = md_path.with_suffix(".pdf")
        source = f"deck:{md_path.name}"
    else:
        deck_dir.mkdir(parents=True, exist_ok=True)
        md_path = deck_dir / "concept-mini.md"
        pdf_path = deck_dir / "concept-mini.pdf"
        branded_tmp = md_path
        source = f"theme:{theme}" + (f":{lang}" if theme == "programming" else "")

    result["source"] = source
    result["lang"] = lang
    result["preview_bullets"] = [b["en"] for b in bullets]

    if not write:
        return result

    try:
        if branded_tmp is not None:
            branded_tmp.write_text(
                brand_theme_pack(theme, code, title, t["uni_label"], lang=lang),
                encoding="utf-8",
            )
        if pdf_path is None or md_path is None:
            raise RuntimeError("missing md/pdf path")
        # Re-export if md newer than pdf or pdf missing
        if (not pdf_path.exists()) or (md_path.stat().st_mtime > pdf_path.stat().st_mtime):
            export_pdf(md_path, pdf_path)

        texts = page_texts(pdf_path)
        pages = pick_sample_pages(texts, limit=6)
        if not pages:
            raise RuntimeError("no pages to sample")

        captions = [caption_from_text(texts[i], i) for i in pages]
        stems = [slug_stem(cap[0], j) for j, cap in enumerate(captions)]
        # Ensure unique stems
        seen = set()
        uniq_stems = []
        for s in stems:
            base = s
            k = 2
            while s in seen:
                s = f"{base}-{k}"
                k += 1
            seen.add(s)
            uniq_stems.append(s)
        stems = uniq_stems

        render_selected_webps(pdf_path, asset_dir, pages, stems)
        label = "concept-review tutoring deck"
        if source.startswith("deck:"):
            label = "tutoring deck"
        block = sample_slides_block(code, title, asset_prefix, stems, captions, bullets, label)

        by_yml: dict[Path, list[str]] = defaultdict(list)
        for ref in t["refs"]:
            by_yml[ref["yml"]].append(ref["code"])

        for yml_path, _codes in by_yml.items():
            data = load_admin(yml_path)
            changed = False
            for course in data.get("courses") or []:
                if not isinstance(course, dict):
                    continue
                if (course.get("material") or "").strip() != t["material"]:
                    continue
                ccode = (course.get("code") or "").strip()
                if is_protected(ccode, force) and not code_matches_force(ccode, force):
                    continue
                wire_sample(course, block)
                changed = True
            if changed:
                yml_path.write_text(dump_admin(data), encoding="utf-8")

        result["stems"] = stems
        result["pages"] = pages
        result["asset"] = str(asset_dir.relative_to(ROOT))
    except Exception as e:
        result["error"] = str(e)
    return result


def scrub_all_intro_wording() -> int:
    """Pass over all course YAML to replace intro-deck wording even on protected rows."""
    n = 0
    for yml in COURSES_DIR.glob("*.yml"):
        data = load_admin(yml)
        changed = False
        for course in data.get("courses") or []:
            if not isinstance(course, dict):
                continue
            before = yaml.safe_dump(course, allow_unicode=True)
            scrub_intro_copy(course)
            after = yaml.safe_dump(course, allow_unicode=True)
            if before != after:
                changed = True
                n += 1
        if changed:
            yml.write_text(dump_admin(data), encoding="utf-8")
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--min-files", type=int, default=MIN_FILES_DEFAULT)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--include-existing",
        action="store_true",
        help="Regenerate even if a non-intro sample already exists",
    )
    ap.add_argument(
        "--force-course",
        action="append",
        default=[],
        help="Force regenerate protected / existing course code(s)",
    )
    ap.add_argument("--folder", action="append", default=[])
    ap.add_argument("--theme", action="append", default=[], help="Limit to theme id(s)")
    ap.add_argument("--scrub-copy-only", action="store_true", help="Only scrub intro wording in YAML")
    args = ap.parse_args()

    if not MARP.exists():
        print(f"missing marp wrapper: {MARP}", file=sys.stderr)
        return 1
    if not THEME_PACKS.exists():
        print(f"missing theme packs: {THEME_PACKS}", file=sys.stderr)
        return 1

    if args.scrub_copy_only:
        n = scrub_all_intro_wording() if args.write else 0
        print(f"scrubbed courses={n} write={args.write}")
        return 0

    force = set(args.force_course or [])
    targets = collect_targets(args.min_files)
    if args.folder:
        allow = set(args.folder)
        targets = [t for t in targets if t["folder"] in allow]
    if args.theme:
        allow_t = set(args.theme)
        targets = [t for t in targets if t["theme"] in allow_t]
    if args.limit:
        targets = targets[: args.limit]

    skip_existing = not args.include_existing
    print(
        f"targets={len(targets)} min_files={args.min_files} write={args.write} "
        f"include_existing={args.include_existing} force={sorted(force)}"
    )

    ok = skip = err = 0
    for i, t in enumerate(targets, 1):
        handle = t["code"] or t["title"]
        print(
            f"[{i}/{len(targets)}] {t['folder']}/{t['material'][:36]} · {handle} "
            f"n={t['n_files']} decks={len(t['real_decks'])} regen={t['needs_regen']}",
            flush=True,
        )
        r = process_target(t, write=args.write, skip_existing=skip_existing, force=force)
        if r.get("skipped"):
            skip += 1
            print(f"  skip ({r.get('source')})")
        elif r.get("error"):
            err += 1
            print(f"  ERROR: {r['error']}")
        else:
            ok += 1
            if not args.write:
                print(f"  dry source={r.get('source')} bullets={r.get('preview_bullets')}")
            else:
                print(
                    f"  wrote {r.get('asset')} source={r.get('source')} "
                    f"pages={r.get('pages')} stems={r.get('stems')}"
                )

    if args.write:
        scrub_all_intro_wording()

    mode = "WROTE" if args.write else "DRY-RUN"
    print(f"\n{mode}: ok={ok} skipped={skip} errors={err}")
    return 0 if err == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
