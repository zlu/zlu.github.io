---
title: WAF
synonyms:
  - Web Application Firewall
related:
  - JA4
  - JA4H
  - User-Agent
  - ASN
---
A **WAF** (Web Application Firewall) sits in front of an app and allows, challenges, rate-limits, or denies HTTP requests by rule. Cloud edges (Cloudflare, Vercel Firewall, AWS WAF, and similar) are the common form for public sites.

Useful signals: path, method, [User-Agent](/cs-glossary/user-agent/), IP / [ASN](/cs-glossary/asn/), bot scores, and TLS digests such as [JA4](/cs-glossary/ja4/). Managed "bot protection" that only logs still lets traffic hit origin compute until a custom Deny (or equivalent) fires.
