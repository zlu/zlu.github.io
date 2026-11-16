---
title: User-Agent
synonyms:
  - UA
  - User Agent
related:
  - JA4
  - JA4H
  - WAF
---
The **User-Agent** header is an HTTP string the client sends to describe itself (browser product, OS, version). It is trivial to forge. Scrapers rotate UAs while keeping the same TLS stack, so many strings can sit under one [JA4](/cs-glossary/ja4/).

Useful as an **equals** match once you have a stable bad string. Risky as a broad `contains` rule (`Chrome/110`) because real browsers and crawlers share substrings. Prefer combining UA with path, ASN, or JA4 when blocking at a [WAF](/cs-glossary/waf/).
