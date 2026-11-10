# [zlu.github.io](http://zlu.github.io)

Personal site for [Zhao Lü](https://zlu.me).

Built with **Jekyll 4** and custom `_plugins`. Deployed to **GitHub Pages** on push to `main`.

## Local development

Requires Ruby (see `.ruby-version` / `mise.toml`).

```bash
bundle install
bundle exec jekyll serve
bundle exec jekyll serve --draft --future --livereload
```

Site: [http://127.0.0.1:4000](http://127.0.0.1:4000)

Drafts live in `_drafts/` (gitignored). Publish by moving a draft into `_posts/`.

## Layout

| Path | Role |
|------|------|
| `_posts/` | Blog posts |
| `_ai-glossary/`, `_med-glossary/`, `_cs-glossary/` | Glossary collections |
| `_albums/`, `_photos/` | Photo trips + per-shot front matter (images on R2) |
| `_data/courses/` | Teach catalog (source of truth for university/course SEO pages) |
| `_algos/` | Algorithm guides on `/algos/` (hub + per-algo pages linking AlgoScape demos) |
| `_plugins/` | Custom generators (required for `/teach/*`) |
| `redirects/` | Soft redirects for retired URLs |
| `admin/` | Decap CMS config |
| `.scripts/` | Photos upload, teach sync, IndexNow / Baidu URL push |

Web-sized photos are served from Cloudflare R2 (`photo_cdn` in `_config.yml` → `https://media.zlu.me`).

## Photos

```bash
pip install -r .scripts/requirements-photos.txt
cp .scripts/r2.env.example .scripts/r2.env   # fill R2 credentials
python3 .scripts/photos_serve.py             # drag-and-drop UI → http://127.0.0.1:8787
# or CLI:
python3 .scripts/photos_add.py ./IMG.jpg --title "…" --album austin-2018
```

Commit the generated `_photos/*.md` (and album files) after upload.

## Teach catalog

Admin YAML under `_data/courses/` drives public `/teach/` pages via `teach_pages.rb`. Optional sync against private teaching materials:

```bash
python3 .scripts/sync-teach-catalog.py --suggest
python3 .scripts/sync-teach-catalog.py --write-links
python3 .scripts/sync-teach-catalog.py
```

## SEO helpers

`/llms.txt` is generated on every build by [`_plugins/llms_txt.rb`](_plugins/llms_txt.rb)
(all posts, glossaries, algos, teach universities/courses, Matheion lessons, products;
see [llmstxt.org](https://llmstxt.org/)).

Dead teach shells still appearing in GSC: submit URLs from
[`.scripts/gsc_removals_teach_dead_urls.txt`](.scripts/gsc_removals_teach_dead_urls.txt)
via Search Console → Removals (temporary). `teach_retired_redirects` is off in
`_config.yml` (GitHub Pages sync timeout). For visitors, import only GSC-hit
retired shells as Cloudflare **Bulk Redirects** (Free ≤10k; we generate ~2.5k):

```bash
# After exporting GSC Performance → Pages CSV:
python3 .scripts/generate_teach_bulk_redirects.py --gsc-pages Pages.csv
# Dashboard: Bulk redirects → import .scripts/teach-uni-bulk-redirects.csv
# Or API (needs Account Filter Lists Edit + Rulesets Write token):
CLOUDFLARE_API_TOKEN=… .scripts/deploy-teach-bulk-redirects.sh
```

IndexNow should **stream** only changed URLs (Bing flags sitemap dumps as batch mode).

```bash
# After publishing: git-map the last commit (default). If that commit is
# data-only (teach catalog sync), the script falls back to sitemap URLs that
# are new vs .scripts/indexnow_submit_state.json (never-submitted first).
python3 .scripts/indexnow_submit_urls.py --dry-run
python3 .scripts/indexnow_submit_urls.py

# Explicit post URL (safest right after publish)
python3 .scripts/indexnow_submit_urls.py --url https://zlu.me/2026/11/06/automata-basics-for-coders-part-1.html

# Or force sitemap new-URL diff
python3 .scripts/indexnow_submit_urls.py --from-sitemap --dry-run

# Needs .scripts/indexnow.env (or root {key}.txt)

# Baidu active push — needs .scripts/baidu.env
python3 .scripts/baidu_submit_urls.py --dry-run

# After matheion.zlu.me serves Next from tutor_me/site (monorepo):
#   cd ../tutor_me/site && npm run indexnow -- --dry-run
```

## Web analytics (BetterReads admin)

First-party funnel (no GA). Browser → `PUT https://www.betterreads.dev/api/zlume-web-stats`
→ `zlume_web_visitor_stats` → BetterReads `/admin/analytics?app=zlume`.

GitHub Pages is static, so ingest lives on betterreads-fe (same project as `/admin`),
not a Cloudflare Worker.

1. Run [`docs/supabase-zlume-web-analytics.sql`](docs/supabase-zlume-web-analytics.sql) once in the BR Supabase SQL editor.
2. Deploy betterreads-fe (registry + `/api/zlume-web-stats`).
3. Ship the Jekyll site (tracker: [`assets/js/web-analytics.js`](assets/js/web-analytics.js)).

## Deploy

[`.github/workflows/jekyll.yml`](.github/workflows/jekyll.yml) runs `bundle exec jekyll build` and deploys the artifact to GitHub Pages. Custom plugins must run in CI (stock GitHub Pages Jekyll would skip them and leave `/teach/*` as 404).

## Jekyll Compose
- `bundle exec jekyll draft "My New Draft"`: Generates a new Markdown file inside your _drafts/ folder with basic predefined front matter.
- `bundle exec jekyll post "Hello World"`: Creates a fresh blog post directly in _posts/, automatically prepending today's date to the filename (e.g., 2026-10-27-hello-world.md).
- `bundle exec jekyll page "About Me"`: Instantly spins up a standalone site page (like about-me.md).
- `bundle exec jekyll publish _drafts/my-new-draft.md`: Moves a draft file from your _drafts/ directory to _posts/, renaming it with the current date so it is ready to publish.
- `bundle exec jekyll unpublish _posts/2026-10-27-hello-world.md`: Performs the reverse, stripping the date off the filename and shifting it back to your _drafts/ archive.
- `bundle exec jekyll rename _posts/old-title.md "New Title"`: Safely renames an existing post or draft filename and updates its corresponding internal title attributes.