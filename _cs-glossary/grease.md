---
title: GREASE
synonyms:
  - Generate Random Extensions And Sustain Extensibility
related:
  - TLS
  - ClientHello
  - JA4
  - Cipher suite
---
**GREASE** (Generate Random Extensions And Sustain Extensibility) is a TLS practice from RFC 8701: clients insert reserved "fake" cipher suite and extension IDs so broken middleboxes that reject unknown values fail in testing instead of in production years later.

GREASE values match the pattern `0x?a?a` (for example `0x0a0a`, `0x1a1a`). [JA4](/cs-glossary/ja4/) and similar fingerprints strip them before counting or hashing [ClientHello](/cs-glossary/clienthello/) fields, otherwise the digest would change every few connections.
