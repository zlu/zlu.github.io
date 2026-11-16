---
title: JA3
related:
  - JA4
  - JA4H
  - TLS
  - ClientHello
  - Cipher suite
---
**JA3** is an older TLS client fingerprint: an MD5 of selected [ClientHello](/cs-glossary/clienthello/) fields (version, [cipher suites](/cs-glossary/cipher-suite/), extensions, elliptic curves, and point formats) in the order they appeared on the wire.

Modern browsers shuffle extension order and inject [GREASE](/cs-glossary/grease/), so the same browser build often produced many JA3s. [JA4](/cs-glossary/ja4/) replaces that model by sorting key lists before hashing and encoding a short plaintext prefix (transport, version, SNI flag, counts, ALPN).
