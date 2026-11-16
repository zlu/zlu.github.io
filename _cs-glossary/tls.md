---
title: TLS
synonyms:
  - Transport Layer Security
  - SSL
related:
  - ClientHello
  - SNI
  - ALPN
  - Cipher suite
  - QUIC
  - TCP
  - JA4
---
**TLS** (Transport Layer Security) encrypts an application protocol (usually HTTP) on top of a transport such as [TCP](/cs-glossary/tcp/). The client and server negotiate version, [cipher suites](/cs-glossary/cipher-suite/), and extensions during a handshake that starts with a [ClientHello](/cs-glossary/clienthello/).

TLS 1.2 and TLS 1.3 are the versions you still see in production. TLS 1.3 shortens the handshake and drops many weak ciphers. People still say "SSL" for the same stack; modern servers speak TLS.

JA3 / [JA4](/cs-glossary/ja4/) fingerprint the ClientHello so a WAF can group clients by TLS stack without reading HTTP headers.
