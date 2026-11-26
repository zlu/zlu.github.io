---
title: NAT
synonyms:
  - Network Address Translation
  - Double NAT
related:
  - LAN
  - WAN
  - Subnet
  - OSI model
  - ONT
---
**NAT** (network address translation) lets many devices on a private [LAN](/cs-glossary/lan/) share one upstream address. The router rewrites source IPs on the way out and tracks flows so replies return to the right host.

**Double NAT** is two NAT devices in series (ISP [ONT](/cs-glossary/ont/) + Wi‑Fi router). Each creates its own private [subnet](/cs-glossary/subnet/). Devices on the inner LAN are not on-link peers of services on the outer LAN, so [Samba](/cs-glossary/samba/) / [DLNA](/cs-glossary/dlna/) discovery often fails even when you can still reach the outer box’s HTTP admin UI through routing.

Fix: put the inner router in AP / [bridge](/cs-glossary/bridge/) mode (DHCP off) so there is one LAN and one NAT at the edge.
