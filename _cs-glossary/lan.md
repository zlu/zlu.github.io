---
title: LAN
synonyms:
  - Local Area Network
related:
  - WAN
  - Subnet
  - NAT
  - DHCP
  - OSI model
---
A **LAN** (local area network) is the private network inside a home or office: Ethernet switches, Wi‑Fi APs, and the devices that share one broadcast domain or one private [subnet](/cs-glossary/subnet/).

Home [NAT](/cs-glossary/nat/) routers hand out LAN addresses (often `192.168.x.x`) via [DHCP](/cs-glossary/dhcp/). Services that assume “same LAN” ([Samba](/cs-glossary/samba/), [DLNA](/cs-glossary/dlna/), AirPlay) usually need clients on that subnet, not behind a second NAT toward a [WAN](/cs-glossary/wan/).
