# GSC URL removal candidates (aggressive)

Submit in [Google Search Console → Removals](https://search.google.com/search-console)
→ **Temporary removals** → **Remove all URLs with this prefix**.

Indexed ~13k vs sitemap ~1.3k: the gap is retired `/teach/universities/<slug>/…`
shells (gone from the build; CF bulk redirects cover impressed ones only).

## How (manual)

1. Prefer **prefix** removals (one request clears hub + preview/assignment/exam/sync).
2. Do **not** submit `https://zlu.me/teach/universities/` as a blanket prefix — that also drops live hubs.
3. GSC pending temporary-removal quota is limited (~500). Work the impressed batch first.

## Files

| File | Count | Use |
|------|------:|-----|
| [`.scripts/gsc_removals_teach_dead_prefixes_top100.txt`](../.scripts/gsc_removals_teach_dead_prefixes_top100.txt) | 100 | Submit today |
| [`.scripts/gsc_removals_teach_dead_prefixes.txt`](../.scripts/gsc_removals_teach_dead_prefixes.txt) | 468 | Full impressed set |
| [`.scripts/gsc_removals_teach_dead_prefixes_all_retired.txt`](../.scripts/gsc_removals_teach_dead_prefixes_all_retired.txt) | 3828 | After quota frees; optional |
| [`.scripts/gsc_removals_teach_dead_urls.txt`](../.scripts/gsc_removals_teach_dead_urls.txt) | ~60 | Legacy exact-URL list (slower) |

Regenerate CF redirects if you have a fresh Pages export:

```bash
python3 .scripts/generate_teach_bulk_redirects.py --gsc-pages Pages.csv
```

## Priority prefixes (top impressions)

- `https://zlu.me/teach/universities/st-cloud-technical-and-community-college/`
- `https://zlu.me/teach/universities/saint-cloud-technical-and-community-college/`
- `https://zlu.me/teach/universities/canadian-mennonite-university/`
- `https://zlu.me/teach/universities/miami-international-university-of-art-and-design/`
- `https://zlu.me/teach/universities/miami-international-university-of-art-design/`
- `https://zlu.me/teach/universities/york-technical-college/`
- `https://zlu.me/teach/universities/north-central-kansas-technical-college/`
- `https://zlu.me/teach/universities/elizabethtown-community-and-technical-college/`
- `https://zlu.me/teach/universities/mcdaniel-college/`
- `https://zlu.me/teach/universities/miami-jacobs-career-college/`
- `https://zlu.me/teach/universities/wayne-state-university/`
- `https://zlu.me/teach/universities/northwest-kansas-technical-college/`
- `https://zlu.me/teach/universities/diablo-valley-college/`
- `https://zlu.me/teach/universities/cape-cod-community-college/`
- `https://zlu.me/teach/universities/charles-sturt-university/`
- `https://zlu.me/teach/universities/williams-baptist-university/`
- `https://zlu.me/teach/universities/cloud-county-community-college/`
- `https://zlu.me/teach/universities/lincoln-memorial-university/`
- `https://zlu.me/teach/universities/the-king-s-university-texas/`
- `https://zlu.me/teach/universities/francis-marion-university/`
- `https://zlu.me/teach/universities/siena-university/`
- `https://zlu.me/teach/universities/pine-technical-and-community-college/`
- `https://zlu.me/teach/universities/southeastern-technical-college/`
- `https://zlu.me/teach/universities/nhti-concord-s-community-college/`
- `https://zlu.me/teach/universities/delta-state-university/`
- `https://zlu.me/teach/universities/jacksonville-university/`
- `https://zlu.me/teach/universities/louisiana-state-university/`
- `https://zlu.me/teach/universities/california-state-university-dominguez-hills/`
- `https://zlu.me/teach/universities/siena-university-new-york/`
- `https://zlu.me/teach/universities/florida-polytechnic-university/`

Rest of the ranked 468: `.scripts/gsc_removals_teach_dead_prefixes.txt`.
