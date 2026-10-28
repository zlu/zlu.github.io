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
| `_ai-glossary/`, `_med-glossary/` | Glossary collections |
| `_albums/`, `_photos/` | Photo trips + per-shot front matter (images on R2) |
| `_data/courses/` | Teach catalog (source of truth for university/course SEO pages) |
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
python3 .scripts/sync-teach-catalog.py
python3 .scripts/sync-teach-catalog.py --suggest
```

## SEO helpers

```bash
# IndexNow (Bing et al.) — needs .scripts/indexnow.env
python3 .scripts/indexnow_submit_urls.py --dry-run

# Baidu active push — needs .scripts/baidu.env
python3 .scripts/baidu_submit_urls.py --dry-run
```

## Deploy

[`.github/workflows/jekyll.yml`](.github/workflows/jekyll.yml) runs `bundle exec jekyll build` and deploys the artifact to GitHub Pages. Custom plugins must run in CI (stock GitHub Pages Jekyll would skip them and leave `/teach/*` as 404).

## Jekyll Compose
- `bundle exec jekyll draft "My New Draft"`: Generates a new Markdown file inside your _drafts/ folder with basic predefined front matter.
- `bundle exec jekyll post "Hello World"`: Creates a fresh blog post directly in _posts/, automatically prepending today's date to the filename (e.g., 2026-10-27-hello-world.md).
- `bundle exec jekyll page "About Me"`: Instantly spins up a standalone site page (like about-me.md).
- `bundle exec jekyll publish _drafts/my-new-draft.md`: Moves a draft file from your _drafts/ directory to _posts/, renaming it with the current date so it is ready to publish.
- `bundle exec jekyll unpublish _posts/2026-10-27-hello-world.md`: Performs the reverse, stripping the date off the filename and shifting it back to your _drafts/ archive.
- `bundle exec jekyll rename _posts/old-title.md "New Title"`: Safely renames an existing post or draft filename and updates its corresponding internal title attributes.