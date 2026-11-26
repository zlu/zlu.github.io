---
title: PPPoE
synonyms:
  - Point-to-Point Protocol over Ethernet
related:
  - ONT
  - WAN
  - NAT
---
**PPPoE** (Point-to-Point Protocol over Ethernet) is how many DSL / fiber ISPs authenticate a home session: username and password on the [WAN](/cs-glossary/wan/) face, then an IP session toward the Internet.

In China Mobile setups the [ONT](/cs-glossary/ont/) often dials PPPoE. Keep that job on the ONT; put media / VPN on a side box rather than moving dial into a second locked router unless you know you need to.
