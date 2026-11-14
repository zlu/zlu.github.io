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

## Export

From the blog repo root:

```bash
npx @marp-team/marp-cli --theme-set marp/zlu-theme.css --theme zlu --html \
  marp/TEMPLATE.md -o marp/TEMPLATE.pdf

npx @marp-team/marp-cli --theme-set marp/zlu-theme.css --theme zlu --html \
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

**Do not put `$...$` inside raw HTML** (`<div>`, `<h3>`, callouts). Marp’s math engine skips HTML blocks — use Unicode (`f″ > 0`) or keep the formula in Markdown outside the HTML.

## Brand tokens

- **Ink / Navy** — covers, titles
- **Signal blue** — accent bar, badges, links
- **Mist / Paper** — content ground
- **Mist + left accent** — tip / pitfall / note (cool slate · brand blue, not traffic-light fills)
- **Fonts** — Sora (display), Manrope (body), IBM Plex Mono (code)
- **Mark** — **zlu** (footer / cover signature)
