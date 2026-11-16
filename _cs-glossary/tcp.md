---
title: TCP
synonyms:
  - Transmission Control Protocol
related:
  - TLS
  - QUIC
---
**TCP** (Transmission Control Protocol) is a reliable, ordered byte-stream transport over IP. Most classic HTTPS is [TLS](/cs-glossary/tls/) over TCP (`https://` on port 443).

TCP needs a three-way handshake before data, and head-of-line blocking inside a connection when a packet is lost. [QUIC](/cs-glossary/quic/) moves reliability and TLS-like crypto onto UDP to avoid that model. In [JA4](/cs-glossary/ja4/), transport `t` means TLS over TCP; `q` means QUIC.
