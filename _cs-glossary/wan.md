---
title: WAN
synonyms:
  - Wide Area Network
related:
  - LAN
  - NAT
  - ONT
  - PPPoE
  - OSI model
---
A **WAN** (wide area network) is the uplink side of a home router: the link toward the ISP and the public Internet, as opposed to the private [LAN](/cs-glossary/lan/) where phones and PCs sit.

On a dual-router setup the second box’s WAN port plugs into the first box’s LAN. That WAN face usually gets a private IP via DHCP and runs [NAT](/cs-glossary/nat/) again ([OSI](/cs-glossary/osi-model/) L3). Bridging / AP mode turns that second box into mostly L2 and removes the extra WAN hop.
