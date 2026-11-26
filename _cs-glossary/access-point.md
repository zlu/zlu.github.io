---
title: Access point
synonyms:
  - AP
  - AP mode
  - Wireless AP
related:
  - Bridge
  - LAN
  - DHCP
  - WAN
---
An **access point (AP)** is a Wi‑Fi radio that attaches wireless clients to an existing wired [LAN](/cs-glossary/lan/). In **AP mode** (often labeled bridge mode on consumer gear) the box does not run its own [DHCP](/cs-glossary/dhcp/) or [NAT](/cs-glossary/nat/); clients get IPs from the upstream gateway.

Use AP mode when a second Wi‑Fi router sits behind an [ONT](/cs-glossary/ont/) and you want one subnet so [Samba](/cs-glossary/samba/) / [DLNA](/cs-glossary/dlna/) see everyone as LAN peers.
