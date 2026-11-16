---
title: JA4
related:
  - JA3
  - JA4H
  - TLS
  - ClientHello
  - SNI
  - ALPN
  - Cipher suite
  - GREASE
  - User-Agent
  - WAF
---
**JA4** is a [TLS](/cs-glossary/tls/) [ClientHello](/cs-glossary/clienthello/) fingerprint from FoxIO. Format: `JA4_a_JA4_b_JA4_c`.

- **JA4_a:** transport (`t`/`q`), TLS version, [SNI](/cs-glossary/sni/) present (`d`/`i`), cipher count, extension count, first [ALPN](/cs-glossary/alpn/) chars
- **JA4_b:** truncated SHA-256 of sorted [cipher suite](/cs-glossary/cipher-suite/) IDs ([GREASE](/cs-glossary/grease/) removed)
- **JA4_c:** truncated SHA-256 of sorted extension type IDs (SNI/ALPN omitted) plus signature algorithms in wire order

JA4 ignores HTTP headers. Rotating [User-Agent](/cs-glossary/user-agent/) under one TLS stack keeps the same digest. [JA4H](/cs-glossary/ja4h/) covers the HTTP header side. Edges and [WAFs](/cs-glossary/waf/) use JA4 to cluster bots; bare-denying a common Chrome digest can hit real browsers that share the stack.
