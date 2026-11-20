# zlu Marp brand theme

Personal slide brand for tutoring / teaching decks.  
Lives with the site repo: [`zlu.github.io/marp`](https://github.com/zlu/zlu.github.io/tree/main/marp) (excluded from the Jekyll build).

## Files

| File | Purpose |
|------|---------|
| `zlu-theme.css` | Design tokens + layouts (`/* @theme zlu */`) |
| `TEMPLATE.md` | Copy-paste starter with every pattern |
| `TEMPLATE.pdf` | Visual preview of the template |
| `marp.config.mjs` | CLI defaults (`html`, theme set) |

Teaching decks live in each course folder under `teaching/…/decks/` (not here). Point those files at this theme via absolute path in VS Code settings.

## Preview in VS Code

In this repo (or any deck that points at the theme):

```json
{
  "markdown.marp.themes": ["./marp/zlu-theme.css"],
  "markdown.marp.html": "all",
  "markdown.marp.mathTypesetting": "mathjax"
}
```

Open `TEMPLATE.md` → Marp: Open Preview.

From another project, point `markdown.marp.themes` at this file, e.g.:

`/Users/zlu/baidu-sync/projects/zlu.github.io/marp/zlu-theme.css`

## Slide sizes

Presets defined in `zlu-theme.css` via `@size`:

| Frontmatter | Use |
|---|---|
| `size: 16:9` | Default tutoring decks |
| `size: 4:3` | Older projectors |
| `size: A4` | Single-page print cheatsheets (portrait) |

## Export

Anywhere on this machine (script lives at `marp/zlu-marp`, installed to `~/.local/bin/zlu-marp`):

```bash
zlu-marp deck.md                 # writes deck.pdf next to the markdown
zlu-marp deck.md -o out.pdf      # custom output path
```

Local images (e.g. `![w:700](figures/plot.png)` next to the markdown) are allowed: the wrapper passes `--allow-local-files`.

PDF bookmarks (Preview sidebar TOC): enabled via `--pdf-outlines` from Markdown headings (`#` / `##`). Open **View → Table of Contents** (or the sidebar TOC button) in Preview.

Override the theme path if needed:

```bash
ZLU_MARP_THEME=/path/to/zlu-theme.css zlu-marp deck.md
```

From the blog repo root (raw CLI):

```bash
npx @marp-team/marp-cli --theme-set marp/zlu-theme.css --theme zlu --html \
  --allow-local-files \
  marp/TEMPLATE.md -o marp/TEMPLATE.pdf

npx @marp-team/marp-cli --theme-set marp/zlu-theme.css --theme zlu --html \
  --allow-local-files \
  your-deck.md -o your-deck.pptx
```

## Frontmatter for every deck

```yaml
---
marp: true
theme: zlu
paginate: true
size: 16:9
math: mathjax
html: true
footer: zlu
---
```

## Slide classes

Put on the line **before** the slide content:

```markdown
<!-- _class: cover -->
<!-- _class: section -->
<!-- _class: tight -->
<!-- _class: equation -->
<!-- _class: center -->
<!-- _class: end -->
```

`lead` is an alias of `cover`.

## HTML patterns (requires `html: true`)

- `.roadmap` / `.roadmap-row` / `.badge` / `.meta` — timed agenda
- `.columns` — two equal columns
- `.shape-cards` / `.shape-card.up|.down` — concavity cards
- `.eyebrow` — small uppercase label
- `.badge-inline` — inline letter chip
- `.tip` / `.pitfall` / `.note` — tutoring callouts

## Math

Use `$inline$` and `$$display$$` only. LaTeX `\(...\)` will not render in Marp.

**Do not put `$...$` inside raw HTML** (`<div>`, `<h3>`, callouts, roadmap rows). Marp’s math engine skips HTML blocks — use Unicode (`f″ > 0`, `β̂`) or keep the formula in Markdown **outside** the HTML.

**Tables:** prefer bullets or a display `$$...$$` above a plain-language list. Pipe tables often mangle `$...$` (especially subscripts / `\mid`).

## Brand tokens

- **Ink / Navy** — covers, titles
- **Signal blue** — accent bar, badges, links
- **Mist / Paper** — content ground
- **Mist + left accent** — tip / pitfall / note (cool slate · brand blue, not traffic-light fills)
- **Fonts** — Sora (display), Manrope (body), IBM Plex Mono (code)
- **Mark** — **zlu** (footer / cover signature)
