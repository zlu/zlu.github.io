---
title: QUIC
related:
  - TLS
  - TCP
  - ALPN
  - JA4
---
**QUIC** is a UDP-based transport that bundles congestion control, multiplexing, and crypto (TLS 1.3-derived) in one layer. HTTP/3 runs on QUIC. Browsers prefer it when the path and server allow it.

Compared with [TLS](/cs-glossary/tls/) over [TCP](/cs-glossary/tcp/), QUIC avoids TCP head-of-line blocking across streams and can resume faster with 0-RTT. The ClientHello still exists inside QUIC's crypto handshake, so [JA4](/cs-glossary/ja4/) can fingerprint it with transport prefix `q` instead of `t`.
