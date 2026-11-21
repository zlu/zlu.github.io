---
layout: post
title: "Don't Trust AI Coding Agents With Production Truth"
date: 2026-11-21
tags:
  - ai
  - agents
  - seo
  - googlebot
  - betterreads
  - vercel
description: "For two months I told Cursor agents that something was still blocking Googlebot on BetterReads. They kept saying Live Inspection passed, crawl budget, firewall OK. They were wrong. The app was 429ing real crawls."
keywords: "AI coding agent, Cursor, Googlebot blocked, Google Search Console, Live URL Inspection, crawl budget, traffic policy, unverified crawler, BetterReads, do not trust AI"
comments: true
---

For about two months I kept asking the same question in Cursor:

> Is something on our side still blocking Google?

I asked it in different words. Soft ones first. Then louder. Then with screenshots from [Google Search Console](https://search.google.com/search-console). Roughly **nine separate chats** between mid-October and late November 2026, plus more pushbacks inside those threads. The product was [BetterReads](https://www.betterreads.dev). Organic Google traffic had fallen off a cliff on **13 October**. Bing kept working. Live URL Inspection often looked fine.

The agents kept answering a different question.

## What I kept hearing

The recurring comfort lines, paraphrased from real sessions:

- "App SEO looks fine. Treat it as incomplete GSC data."
- "I don't see an unfixed crawler block. This is SERP demotion / crawl budget."
- "Googlebot is not blacklisted."
- "Botnet management looks healthy."
- "Live testing returns 200 — so Googlebot is not blocked right now."
- "Crawled – currently not indexed is the *opposite* of blocked."
- "The causes are addressed. Wait for the next GSC refresh."

When I said the firewall was still the problem, I was told not to worry. When I said Googlebot was still blocked, I was shown a Live Inspection pass. When host status hit **100% failure**, I was told the fix had already shipped and the chart was lag.

I believed too much of that for too long. That was on me. The agents were confident, polite, and wrong in a way that sounds technical.

## What was actually wrong

There were real incidents in October. Soft-404s after private storage. Vercel Bot Protection in Challenge mode. Incomplete Google crawler CIDRs. Those got partial fixes. The story that stuck afterward — "host is healthy now, just wait for crawl budget" — was the dangerous one.

What was still live in production into November:

1. A Googlebot-shaped request whose IP was not on **our** allowlist was classified `unverified_crawler`.
2. Public Discover HTML then got a **JSON 429** at about **10 requests per minute**.
3. Those 429s could become blacklist **strikes**, then a multi-week **403**.
4. Live URL Inspection often used **different Google egress** and `Google-InspectionTool`, which still got **200**.

So the agent could "prove" the site was fine while Google's normal crawl fleet was still being punished. I reproduced the burst failure myself before the final fix: a handful of Googlebot-UA hits returned HTML, then the rest returned `traffic_rate_limit` JSON. After we deployed the fix, the same burst was **25/25 HTTP 200** with real HTML.

The durable mistake was architectural, not a missing CIDR row. We had built an **allowlist for "good" crawlers**. Google's ranges change. Allowlists rot. The sustainable model is the boring one: **deny known-bad actors**; do not require Google/Bing IP membership to serve public HTML.

Earlier field notes on the farm fight that made those defenses grow teeth: [Fighting Scraping Farms on a Reading Site](/2026/10/26/fighting-scraping-farms-on-a-reading-site/). The farms were real. The collateral damage on Google was also real. Agents were better at celebrating the farm fight than at admitting the SEO wound was still open.

## Why agents fail this class of bug

This is not "AI hallucinated a function name." This is worse.

**Agents optimize for a coherent story that fits the evidence they can fetch in one session.** Live Inspection passes. `robots.txt` allows Googlebot. A single curl with a Googlebot UA returns 200. GSC shows "Crawled – currently not indexed." The tidy narrative writes itself: *not blocked; selection / budget / lag.*

**Agents treat your repeated distrust as emotional noise** once they have a narrative. I said "there is def something wrong," "huge bug," "googlebot is still being blocked somehow!!!!! wtf," "don't tell me vercel fw is ok." Across chats, the default move was still reassurance plus wait.

**Agents do not carry liability for your rankings.** You do. They will happily close the loop on "looks fine from here" because that ends the turn cleanly. Production SEO does not end the turn cleanly. Crawl scars take weeks. False confidence compounds.

**Tooling bias is real.** If the easy probe is green, the hard probe (burst from unverified egress, firewall mitigation headers, strike → ban path, InspectionTool vs Googlebot classification) does not get run unless you force it. I should have forced it earlier. The agent should have offered it without being begged.

## What I will do differently

- When a human says "something is still blocking" for the third time, **assume they are right** until a deny path is disproved under crawl-like conditions, not under Live Inspection.
- Prefer **deny-lists for abuse** over **allow-lists for Google**.
- Never accept "crawl budget" as an answer while any Googlebot-shaped HTML path can still emit **429/403**.
- Treat agent conclusions as **hypotheses**, not ops sign-off. Especially when the agent wrote the traffic policy it is now defending.

## The point

AI coding agents are useful for patches, refactors, and archaeology. They are bad at owning a weeks-long production outage when the outage contradicts a story they already told you.

If you are using them on anything that touches crawlers, payments, auth, or data loss: keep a human veto. Keep logs. Re-ask the same question until the agent stops soothing you and starts trying to falsify itself.

I asked for two months. The soothing answers were expensive.
