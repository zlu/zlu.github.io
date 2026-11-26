---
title: Bridge
synonyms:
  - Network bridge
  - Bridging
  - Bridge mode
related:
  - Access point
  - LAN
  - OSI model
  - NAT
---
A **bridge** forwards frames between segments at [OSI](/cs-glossary/osi-model/) **L2** so they stay one [LAN](/cs-glossary/lan/) / broadcast domain. It does not create a new IP [subnet](/cs-glossary/subnet/) or run [NAT](/cs-glossary/nat/).

Home “bridge mode” / AP mode on a Wi‑Fi router turns off its [DHCP](/cs-glossary/dhcp/) server and WAN routing so the box becomes mostly a wireless bridge into the upstream LAN. Contrast **router mode**, which is an L3 hop with its own subnet.
