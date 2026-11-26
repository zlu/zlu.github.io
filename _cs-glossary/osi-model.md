---
title: OSI model
synonyms:
  - OSI
  - OSI layers
  - Open Systems Interconnection
related:
  - LAN
  - WAN
  - NAT
  - Bridge
  - TCP
---
The **OSI model** splits networking into seven layers. Home setup failures usually sit in four of them:

| Layer | Name | Home example |
| --- | --- | --- |
| L1 | Physical | Fiber into the [ONT](/cs-glossary/ont/); Ethernet cable; Wi‑Fi radio; USB disk |
| L2 | Data link | Switch / AP [bridge](/cs-glossary/bridge/); same broadcast domain; MAC / ARP |
| L3 | Network | IP [subnets](/cs-glossary/subnet/), routing, [NAT](/cs-glossary/nat/) |
| L4-L7 | Transport and up | [TCP](/cs-glossary/tcp/) 445 for [SMB](/cs-glossary/samba/); HTTP admin UI; app share logic |

**Router mode** is an L3 hop with its own LAN subnet. **AP / bridge mode** is mostly L2: Wi‑Fi clients join the upstream subnet and talk to the gateway as on-link peers. [Samba](/cs-glossary/samba/) often needs that L2/L3 sameness; a second NAT breaks it even when HTTP to the upstream admin UI still works.
