---
title: DHCP
synonyms:
  - Dynamic Host Configuration Protocol
related:
  - LAN
  - Subnet
  - NAT
  - Access point
---
**DHCP** (Dynamic Host Configuration Protocol) assigns IP address, mask, gateway, and DNS to clients on a [LAN](/cs-glossary/lan/). Home routers run a DHCP server on the LAN side by default.

Only one DHCP server should be active on a flat LAN. When you put a Wi‑Fi router into [AP](/cs-glossary/access-point/) / [bridge](/cs-glossary/bridge/) mode, turn its DHCP **off** so clients get addresses from the upstream gateway (e.g. the [ONT](/cs-glossary/ont/)).
