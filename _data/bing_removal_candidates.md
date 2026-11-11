# Bing Webmaster URL removal candidates

Submit in [Bing Webmaster Tools → URL Removal](https://www.bing.com/webmasters)
after Cloudflare Bulk Redirects are live for these retired teach shells
(`noindex`, out of sitemap, 301 → `/teach/universities/`).

Source: Bing Webmaster Pages export · Web · last 3 months
(~2026-09-20 → 2026-11-09). Same corpses dominate Bing impressions
(~7–8k of ~8.2k) at pos ~4–6 with **0% CTR** (BR/RU/IN name matches).

CF CSV (already includes twins below):

```bash
# .scripts/teach-uni-bulk-redirects.csv
# CLOUDFLARE_API_TOKEN=… .scripts/deploy-teach-bulk-redirects.sh
```

## Priority (Bing impression corpses)

| URL | Impr | Pos |
| --- | ---: | ---: |
| `/teach/universities/st-cloud-technical-and-community-college/preview/` | 721 | 3.9 |
| `/teach/universities/saint-cloud-technical-and-community-college/preview/` | 567 | 4.8 |
| `/teach/universities/canadian-mennonite-university/assignment/` | 455 | 4.6 |
| `/teach/universities/saint-cloud-technical-and-community-college/assignment/` | 216 | 5.4 |
| `/teach/universities/miami-international-university-of-art-and-design/assignment/` | 205 | 5.8 |
| `/teach/universities/miami-international-university-of-art-design/assignment/` | 200 | 6.0 |
| `/teach/universities/miami-international-university-of-art-design/preview/` | 188 | 5.3 |
| `/teach/universities/york-technical-college/` | 187 | 5.9 |
| `/teach/universities/north-central-kansas-technical-college/assignment/` | 174 | 6.5 |
| `/teach/universities/elizabethtown-community-and-technical-college/` | 165 | 5.7 |
| `/teach/universities/mcdaniel-college/exam/` | 164 | 5.4 |

## Prefix removals (slug twins)

- `https://zlu.me/teach/universities/st-cloud-technical-and-community-college/`
- `https://zlu.me/teach/universities/saint-cloud-technical-and-community-college/`
- `https://zlu.me/teach/universities/miami-international-university-of-art-design/`
- `https://zlu.me/teach/universities/miami-international-university-of-art-and-design/`
- `https://zlu.me/teach/universities/canadian-mennonite-university/`
- `https://zlu.me/teach/universities/elizabethtown-community-and-technical-college/`

## Keep / re-IndexNow (converting)

Stream these via `.scripts/indexnow_submit_urls.py` after deploy (≤5/run):

- `https://zlu.me/teach/courses/monash/fit2102/`
- `https://zlu.me/teach/courses/monash/fit3152/`
- `https://zlu.me/teach/courses/monash/etx3500/`
- `https://zlu.me/teach/`
- `https://zlu.me/2025/06/21/image-processing-with-r.html`
- `https://zlu.me/2025/06/20/deep-dive-into-vector-database.html`

After removals, score Bing `tutor_leads` on `/teach/courses/` + `/teach/` only —
ignore BR/RU/IN university-name impressions.
