---
title: JA4H
synonyms:
  - JA4 HTTP
related:
  - JA4
  - JA3
  - User-Agent
  - WAF
---
**JA4H** is the HTTP companion to [JA4](/cs-glossary/ja4/). It fingerprints HTTP request header **names**, order, and related method / version / cookie / referer / auth flags, not the TLS handshake.

Spoofed [User-Agent](/cs-glossary/user-agent/) strings often keep a stable header layout from the real client library. JA4 groups by TLS stack; JA4H can split those groups when the HTTP header pattern differs. Many edges expose JA4 first; JA4H availability depends on the vendor.
