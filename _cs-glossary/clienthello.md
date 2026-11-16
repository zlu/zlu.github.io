---
title: ClientHello
synonyms:
  - TLS ClientHello
related:
  - TLS
  - Cipher suite
  - SNI
  - ALPN
  - GREASE
  - JA4
  - JA3
---
The **ClientHello** is the first handshake message a TLS client sends. It advertises supported TLS versions, [cipher suites](/cs-glossary/cipher-suite/), and extensions such as [SNI](/cs-glossary/sni/), [ALPN](/cs-glossary/alpn/), supported groups, and signature algorithms.

Browsers often insert [GREASE](/cs-glossary/grease/) values and shuffle extension order. [JA3](/cs-glossary/ja3/) hashed fields in wire order and drifted; [JA4](/cs-glossary/ja4/) sorts selected fields before hashing so the digest stays stable across those tricks. HTTP headers such as User-Agent are not part of the ClientHello.
