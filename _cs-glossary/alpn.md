---
title: ALPN
synonyms:
  - Application-Layer Protocol Negotiation
related:
  - TLS
  - ClientHello
  - SNI
  - QUIC
  - JA4
---
**ALPN** (Application-Layer Protocol Negotiation) is a [TLS](/cs-glossary/tls/) extension where the client lists application protocols it can speak after the handshake (for example `h2`, `http/1.1`, `h3`). The server picks one.

[JA4](/cs-glossary/ja4/) keeps only the **first** ALPN string the client offers, encoded as its first and last alphanumeric characters (`h2`, `h1`, or `00` if ALPN is missing). It ignores later preferences and ignores which protocol the server actually selected.
