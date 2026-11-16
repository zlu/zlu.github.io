---
title: SNI
synonyms:
  - Server Name Indication
related:
  - TLS
  - ClientHello
  - ALPN
  - JA4
---
**SNI** (Server Name Indication) is a [TLS](/cs-glossary/tls/) extension in the [ClientHello](/cs-glossary/clienthello/). The client sends the hostname it wants so a server (or CDN) that hosts many certificates on one IP can pick the right one before the handshake finishes.

[JA4](/cs-glossary/ja4/) records only whether SNI is present (`d`) or absent (`i`). It does **not** hash the hostname string. Two clients talking to different hosts can share a JA4 if the rest of the ClientHello matches.
