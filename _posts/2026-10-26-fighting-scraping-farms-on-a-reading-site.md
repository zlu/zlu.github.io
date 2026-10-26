---
layout: post
title: "Fighting Scraping Farms on a Reading Site"
date: 2026-10-26
tags:
  - security
  - bots
  - vercel
  - nextjs
  - product
  - seo
description: "Field notes from fighting EPUB-scraping farms on BetterReads — good bots vs bad bots, why fingerprints fade, and how observability can cost more than the attack."
keywords: "web scraping farms, bot detection, robots.txt, Googlebot, GPTBot, Meta crawlers, Vercel BotID, EPUB security, traffic policy, AI training crawlers"
comments: true
---

Over the last few weeks of October 2026, [BetterReads](https://www.betterreads.dev) got a sustained visit from scraping farms. Not Googlebot. Not a clumsy `curl` loop. Residential and cloud proxies, rotating IPs, plausible desktop Chrome UserAgents, and a clear goal: pull guest EPUB previews and deep Discover pages as cheaply as possible.

I won some rounds, lost some, and then stopped playing their game. This is the field notes of what burned money, which bots are *wanted*, which are fought, and what I would do again on day one.

Along the way we kept a living crawler catalog in the product repo (`docs/crawlers.md`): every token we care about, the official docs / UA `+url` / IP JSON when they publish one, and our posture (allow HTML, throttle, or Disallow). The tables below are the blog-sized version of that list.

## Patterns that do not look like a bot

The first suspicious wave was almost polite. Direct hits to `/reader?bookId=…&preview=1` — no referrer, short sessions, various geo, Windows + Chrome, sometimes `zh-CN` on Santa Clara egress, sometimes China/SG/HK with the same “open the reader and leave” shape. Real readers usually arrive from Discover, a share link, or search. Bot-Farms land on the URL that returns epubs.

A second surface showed up mixed into the same wave: deep `/discover/<book>/<section>/…` walks. Those pages are public on purpose (SEO). From the farm’s point of view they are also a free index of what to scrape next.

Analytics flagging alone does nothing for bandwidth. You still paid for the HTML and the EPUB rebuild. So we moved enforcement to the edge: classify the actor, then allow / rate-limit / 403 before expensive work runs.

Concrete pain from those weeks:

- **Oct 15:** a self-identified `Lightpanda/1.0` wave on guest EPUB paths — easy hard-block once named.
- **Stale desktop UAs:** anonymous Chrome/Firefox/Edge advertising versions **110–125** (while the real world was ~148+) hammered `/api/catalog/book` hard enough to trip multi‑MB Next data-cache warnings.
- **Oct 20 evening:** on the order of **~1,000 guest preview downloads in ~20 minutes from ~1000 distinct IPs**. Per-IP caps are theater against residential rotation.
- **Meta “Portland” walks:** Meta datacenter IPs with *ordinary browser* UAs enumerating `/discover…` — looks human in raw logs, burns Discover SSR. Preview bots that run enough JS also `PUT` analytics endpoints; a middleware 403 still runs Vercel's Fluid after Firewall Allowed, which burns cash.
- Quiet days were roughly **<$1/day** of Vercel; farm windows pushed that to 10x. Not catastrophic, but a clear signal that “free public HTML” has a floor cost when theft happens.

## Good bots, bad bots, and the ones in between

The rule of thumb I ended up with:

| Kind | What they do | Effect of blocking |
| --- | --- | --- |
| Search / citation | Feed rankings or AI answers that send readers | Hurts discoverability |
| Social preview | Unfurl shared links | Ugly naked URLs in chat apps |
| SEO / backlink tools | Fill Ahrefs / Semrush / Moz reports | No Google/Bing ranking effect; hides you from *that tool* |
| AI training | Bulk-ingest for model weights | Usually no reader traffic |
| Hostile / automation | Steal EPUBs, traverse catalog at farm scale | Saves money and nerves |

Farms ignore `robots.txt`. Still worth getting right for the bots that *do* listen, though.

### Good: search engines (readers)

Allowed on public HTML; expensive APIs and EPUB blocked. For Google and Bing **verified crawler IPs** are preferrable, not User-Agent alone because it is cheap to forge UA.

| Family | Operator | Why we keep them |
| --- | --- | --- |
| `Googlebot`, `Google-InspectionTool`, `Storebot-Google`, `AdsBot-Google` | Google | Primary web search |
| `bingbot`, `adidxbot`, `BingPreview` | Microsoft | Bing + Copilot |
| `DuckDuckBot` | DuckDuckGo | Smaller index, still readers |
| `YandexBot`, `PetalBot` | Yandex / Huawei Petal | Regional search (we also push via Baidu APIs) |
| `Amzn-SearchBot` | Amazon Alexa / Rufus | Search eligibility — **not** `Amazonbot` (training) |

**Classify order matters.** `PetalBot` must be recognized *before* “Huawei cloud egress = CN scraper,” or your own indexing and webmaster verification get rate-limited as abuse. Same idea for Ahrefs before a generic `\bbot\b` automation bucket.

**Verify IP ranges or you 429 yourself.** Incomplete Bing CIDRs once classified real Bingbot as `unverified_crawler` (tight rate limit). Fix was expanding published `bingbot.json` Azure ranges — not relaxing the “UA alone is not enough” rule.

### Good: AI search / citation (still readers)

Same HTML-friendly idea as search engines. Separate from training bots.

| Token | Operator | Role |
| --- | --- | --- |
| `OAI-SearchBot`, `ChatGPT-User` | OpenAI | ChatGPT search index / user-triggered fetch |
| `Claude-SearchBot`, `Claude-User` | Anthropic | Search / user-initiated fetch |
| `PerplexityBot`, `Perplexity-User` | Perplexity | Answer-engine index (not foundation training) |
| `Google-Extended` | Google | robots **control token** for Gemini / grounding — usually not a distinct HTTP UA |
| `meta-webindexer` | Meta AI | Citation / search — **not** `meta-externalagent` |

### Narrow good: social / link preview

When someone pastes a BetterReads URL into Facebook, Instagram, WhatsApp, etc., the app does **not** wait for a human browser. A preview bot GETs the URL, reads Open Graph tags, and builds a card. Without that GET, shares look like a naked URL — no title, no cover. That kills social distribution even when SEO is fine.

So: **serve the share HTML, refuse everything expensive.**

| Token | Treat as |
| --- | --- |
| `facebookexternalhit` | Classic  — share surfaces only |
| `LinkedInBot`, `Twitterbot`, Embedly, Quora / Pinterest previews | Same idea |
| `meta-externalagent` | Broader Meta product / ingest; often on the same shared links, can run JS into analytics — same edge bucket as preview, throttled hard off catalog |
| `meta-webindexer` | **Not** this bucket — AI citation, search posture |

Share-ish allowlist in practice: `/`, `/s/*`, `/b/*`, `/u/*`, `/robots.txt`, `/sitemap*`. Everything else — especially `/discover…`, `/reader`, EPUB, catalog, AI, `PUT /api/web-stats` — is **403** for the preview actor.

I 403’d `/robots.txt` for `facebookexternalhit` because robots wasn’t on the share list.  Robots and sitemaps belong on the preview allowlist.

### Bad (for us): AI training

`Disallow: /` in robots. Edge: rate-limit / automation where matched. Farms may ignore robots — enforcement is traffic policy + cache, not robots alone.

| Token | Operator | Notes |
| --- | --- | --- |
| `GPTBot` | OpenAI | Training crawl (≠ OAI-SearchBot) |
| `ClaudeBot` | Anthropic | Training (≠ Claude-SearchBot) |
| `CCBot` | Common Crawl | Public corpus many models use |
| `Bytespider`, `TikTokSpider` | ByteDance | Training / product ingest |
| `Diffbot`, `ImagesiftBot`, `cohere-ai` | various | Extract / training-adjacent |
| `meta-externalagent` | Meta | Product / training ingest — not WebIndexer |
| `Amazonbot` | Amazon | Product/AI training — **not** Amzn-SearchBot |

`Amazonbot` does not match a naive `\bbot\b` regex — it slips through as “default” unless you name it explicitly.

### Taste call: SEO & backlink tools

These build **third-party SEO products**, not Google rankings. Blocking them does not demote you in Google/Bing; it only empties that vendor’s index for your URLs.

| Token | Action |
| --- | --- |
| `AhrefsBot`, `AhrefsSiteAudit` | **Allow HTML** like search (deliberate — Site Explorer visibility). Named in robots. Still blocked on EPUB / APIs / reader. |
| `SemrushBot` (+ variants), Moz `DotBot` / `rogerbot`, `MJ12bot`, SE Ranking, DataForSEO, Serpstat, Babbar, BLEXBot, Screaming Frog | **Automation** — rate-limit HTML, 403 expensive paths |

Open policy question I left documented: if we want competitors’ Semrush / SE Ranking graphs to show our link profile the same way Ahrefs does, promote those UAs the same way. Until then they stay throttled scrapers. `SERankingBacklinksBot` showed up in crawler logs ~Oct 2026; still on default automation posture.

### Hostile / automation (the actual fight)

Caught by traffic policy / BotID / denylist rather than a polite robots conversation:

- Headless stacks: `HeadlessChrome`, Playwright, Puppeteer, Selenium, Cypress, Lightpanda, Chrome-Lighthouse
- Script clients: curl, wget, python-requests, Go HTTP, …
- Deep-link farms, stale desktop UAs, CN cloud (such as Tencent) egress scrapers, AWS us-west-2 device / scrape farms
- Meta DC “browser UA” catalog walks (above)

**EPUB doctrine:** PD or not, scrapers can fetch public-domain texts from Gutenberg / Internet Archive. They do not get book bytes from our bandwidth. Missing UA on a download path is also 403. Softening only for IP-only signals when the reader is signed in (attributable, quota’d).

## Fingerprints are a calendar, not a wall

The farms adapted on a schedule measured in hours, not months.

1. **Timezone lies.** Early waves used `Asia/Shanghai` (or similar) on non-CN egress with English locale — easy to hard-block on EPUB paths. By the evening of Oct 20 they had moved off that tell: spoofed `America/*` on GB/DE/FR proxies, or headless-default `UTC` on US egress. Same job, new costume.
2. **Stale desktop Chrome.** Anonymous traffic advertising Chrome/Firefox in the 110–125 band was almost never a human. High precision — until the fleet bumped to 126–149.
3. **Xvfb Linux Chrome.** Direct landings on guest preview / deep guide URLs from X11 Linux Chrome with no referrer were a clean “headless box” signal. Useful until they change the display stack.
4. **Wrong chapter referrals.** Some bots paired a `bookId` with chapter slugs harvested from *other* editions. Spine misses (`xxv-postscript` and friends) became a cheap 404 *before* BotID, quota, or any EPUB download — no text, no guest quota burned.

Every UA/TZ rule is temporary. Treat it as a patch while you build something the farm cannot rotate away from.

## What stuck: attach policy to behavior

### Challenge the door, not the costume
 
Direct top-level entry to a guest preview (typed URL, external link — anything that is not in-site navigation) now goes through a one-shot [Vercel BotID](https://vercel.com/docs/botid) challenge. Humans get a short-lived, IP-bound pass cookie. Bots get 403. In-site Discover → “Read free” never sees the challenge.

I deliberately **do not** put BotID on `/api/epub` itself. Script blockers and filtered networks break the client interceptor and would lock out real readers. Entry challenge + grant cookie + quotas is enough.

Rotating-proxy fleets hate IP-bound passes: every new identity must re-prove.

### Never ship the whole book to a guest

Once past the door, a guest preview archive contains **exactly one chapter** — front matter plus the requested (or first) content chapter. Everything else in the spine is a lock page. Next chapter requires signup. SEO is unaffected: `/reader` stays out of the crawlable surface; Discover and book pages remain the public index.

That single product change did more damage to the farm’s economics than any denylist snippet.

### EPUB is humans-only

Classified bot actors — search crawlers, SEO tools, CN cloud, automation UAs, Meta catalog walks — get **403** on book-content download paths, even if we only rate-limit them on HTML. Signed-in readers stay attributable and tier-gated.

### Quotas that survive IP rotation

Per-IP limits are necessary and insufficient on a serverless host: each warm isolate starts with a fresh budget, so a rotating farm leaks forever. We layered:

- Fast in-memory limits per isolate.
- Durable guest quotas in Postgres (per IP, global guest circuit-breaker, per-book across guests).
- Per-account daily caps for signed-in downloads (account farms cannot rotate around an IP key).
- A strike-based **dynamic IP blacklist** shared across isolates: repeated rate-limit trips escalate to multi-week bans, then longer, then effectively permanent — but **only for anonymous traffic**. Login and auth routes stay open. An account is the identity; anonymous browsing is the privilege.
- Stack fingerprints (UA + locale + timezone shape) with hourly budgets — defeats pure IP rotation when the costume stays the same.

IPv6 keys are `/64` prefixes. Farms rotate inside a `/64`; banning a single host address is whack-a-mole.

### CDN is a different planet

The direct CDN host (`cdn.betterreads.dev`) cannot apply app UA rules. Known scraper IPs get mirrored to Cloudflare IP Access Rules from the same static denylist the app uses. Exact IPs and tight `/24`s only — rage-banning whole Aceville/Tencent `/16`s hits real readers.

## Defense stack (outside → in)

| Layer | Role |
| --- | --- |
| `robots.txt` | ToS / named Allow vs training Disallow |
| Vercel WAF | Optional IP Deny; Deny crawler PUTs to analytics *before* Fluid |
| Host gate | Non-canonical `*.vercel.app` hosts 404 (farms fled there) |
| Edge traffic policy | Classify actor → allow / rate-limit / 403 |
| BotID | Direct guest-preview entry only |
| One-chapter guest EPUB | Product economics |
| Durable quotas + strike blacklist | Survive rotation |
| Static denylist + CF mirror | Known bad IPs on app + CDN |
| Analytics `likelyBot` | Keep dashboards honest — does **not** save SSR |
| Admin `/bots` + crawler-log rollups | Ban workbench; copy snippets into denylist |

## Observability can cost more than the attack

One morning we burned on the order of **tens of GB of Supabase Log Query** in a day — not because bots hit PostgREST that hard, but because Studio Logs Explorer / Live was open while noisy debugging ran. Log **ingest** (writes) and log **query** (bytes scanned when *reading* logs) are different meters. App REST traffic is not Log Query. Opening Logs Explorer is.

Hard rules I now follow:

- Prefer Vercel / Cloudflare / your own bounded admin tables over Supabase Studio logs.
- If you must open Logs Explorer: ≤15–60 minutes, filter before run, one shot, pause Live, close the tab.
- Do not leave agents free to paginate analytics tables “just to look,” and do not column-probe schemas in a loop (each `42703` is ingest noise Studio then re-scans).

For crawler analytics we eventually put a **Vercel Log Drain → private R2 → daily Supabase rollup** (later removed once things become manageable), with family buckets (`googlebot`, `bingbot`, `meta_webindexer`, `other_bot`, …). Tiny admin UI, no Studio. Watch for the drain feedback loop: if the receiver lives on the same project, every delivery creates new function logs that re-enter the drain unless you sample/filter `/api/internal/` out.  Although log drain does make live log noise and hard to read and filter.

## Product lessons dressed as security

1. **Anonymous is a privilege; an account is identity.** Ban the IP from anonymous use; never lock someone out of signup. Real readers on reassigned farm IPs just sign in.
2. **False positives have a geography.** Hard-blocking China-TZ on *HTML* false-positives diaspora and HK VPN readers. Keep harsh rules on EPUB download paths; leave HTML open where VPN is normal.
3. **Shared infrastructure needs allowlists, not rage-bans.** Google/Apple infra, verified search crawlers, Meta preview egress — strike systems must skip them or you DDoS yourself socially.
4. **Defense in depth beats a clever regex.** Proxy policy, BotID on direct entry, one-chapter archives, chapter-map guards, quotas, durable blacklist, CDN mirror of static denylists. Any single layer will be bypassed.
5. **Ship a ban workbench, not silent automation for /24s.** Exact IPs are the safe default. A `/24` only when every host in it is hostile.
6. **SEO-tool policy is product taste.** Ahrefs allow ≠ Semrush allow. Document the open question instead of pretending there is one true answer.
7. **Extract the IP intelligence.** I pulled ASN/range helpers into a small library ([farmfence](https://www.npmjs.com/package/farmfence)) so analytics and the edge stay consistent.

## What I would do on day one next time

1. Guest content is a tease by construction (one chapter / watermark / signup), not a full file behind a thin gate.
2. Challenge *direct* entry to the expensive surface; trust in-site navigation.
3. Humans-only on binary/book bytes; crawlers get HTML only.
4. Durable quotas + strike blacklist from the start — do not trust per-isolate memory.
5. Bounded observability pipeline before you open anyone’s Logs Explorer under load.
6. Keep a living crawler catalog (official docs, UA `+url`, IP JSON, posture) and update it when a new token shows up in logs — folklore ages worse than farms.
7. Fingerprint rules as temporary patches with regression tests (`check-traffic-policy.ts` style), knowing the farm will change clothes tonight.

Scraping farms are a tax on anything valuable and fetchable. You cannot shame them with `robots.txt`. You can tell good bots from bad ones in writing, make the anonymous path expensive, the signed-in path attributable, and your own debugging cheaper than their crawl. The rest is patching fingerprints until the next costume change.
