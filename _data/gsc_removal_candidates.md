# GSC URL removal candidates

Submit these in [Google Search Console → Removals](https://search.google.com/search-console)
(Temporary removal → Remove this URL only / prefix as needed) after Cloudflare Bulk
Redirects are imported for GSC-hit retired shells (`noindex`, out of sitemap).

Source: GSC Pages export ~2026-09-20 → 2026-11-09 (high impressions, 0 clicks, thin/dead teach shells).

Regenerate CF CSV anytime:

```bash
python3 .scripts/generate_teach_bulk_redirects.py --gsc-pages Pages.csv
# then import .scripts/teach-uni-bulk-redirects.csv or:
# CLOUDFLARE_API_TOKEN=… .scripts/deploy-teach-bulk-redirects.sh
```

## Priority (top impression corpses / thin shells)

- `https://zlu.me/teach/universities/st-cloud-technical-and-community-college/preview/`
- `https://zlu.me/teach/universities/saint-cloud-technical-and-community-college/preview/`
- `https://zlu.me/teach/universities/canadian-mennonite-university/assignment/`
- `https://zlu.me/teach/universities/saint-cloud-technical-and-community-college/assignment/`
- `https://zlu.me/teach/universities/miami-international-university-of-art-and-design/assignment/`
- `https://zlu.me/teach/universities/miami-international-university-of-art-design/assignment/`
- `https://zlu.me/teach/universities/miami-international-university-of-art-design/preview/`
- `https://zlu.me/teach/universities/north-central-kansas-technical-college/assignment/`
- `https://zlu.me/teach/universities/elizabethtown-community-and-technical-college/preview/`
- `https://zlu.me/teach/universities/elizabethtown-community-and-technical-college/assignment/`
- `https://zlu.me/teach/universities/mcdaniel-college/exam/`
- `https://zlu.me/teach/universities/wayne-state-university/exam/`
- `https://zlu.me/teach/universities/cape-cod-community-college/exam/`
- `https://zlu.me/teach/universities/york-technical-college/exam/`

## Prefix removals (faster if GSC allows)

- `https://zlu.me/teach/universities/st-cloud-technical-and-community-college/`
- `https://zlu.me/teach/universities/saint-cloud-technical-and-community-college/`
- `https://zlu.me/teach/universities/miami-international-university-of-art-design/`
- `https://zlu.me/teach/universities/miami-international-university-of-art-and-design/`
- `https://zlu.me/teach/universities/elizabethtown-community-and-technical-college/`
- `https://zlu.me/teach/universities/canadian-mennonite-university/`

## Course alias

- `https://zlu.me/teach/courses/monash/etx3500-5500-high-dimensional-data-analysis/` → soft-redirect to `/teach/courses/monash/etx3500/`

After Removals + crawl, re-check GSC Pages: uni-service impressions should fall;
score `tutor_leads` on `/teach/courses/` and `/teach/` only.
