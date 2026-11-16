---
title: Cipher suite
synonyms:
  - Cipher suites
  - TLS cipher suite
related:
  - TLS
  - ClientHello
  - JA4
  - JA3
---
A **cipher suite** is a named set of cryptographic algorithms a [TLS](/cs-glossary/tls/) peer is willing to use (key exchange, bulk encryption, MAC / AEAD). The client lists suites in the [ClientHello](/cs-glossary/clienthello/); the server picks one both sides support.

TLS 1.3 suites are shorter (mostly AEAD + HKDF labels such as `TLS_AES_128_GCM_SHA256`). In fingerprints, suites appear as 16-bit IDs. [JA4](/cs-glossary/ja4/) sorts those IDs (after dropping [GREASE](/cs-glossary/grease/)) and hashes the comma-joined hex list into JA4_b.
