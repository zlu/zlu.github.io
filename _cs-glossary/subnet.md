---
title: Subnet
synonyms:
  - Subnetwork
  - IP subnet
related:
  - LAN
  - NAT
  - DHCP
  - OSI model
---
A **subnet** is a block of IP addresses that share a network prefix (e.g. `192.168.1.0/24`). Hosts on the same subnet talk as on-link peers ([OSI](/cs-glossary/osi-model/) L3 local); other destinations go via a gateway.

Two home routers in series usually mean two subnets (`192.168.1.0/24` and `192.168.0.0/24`). That is the topology behind [double NAT](/cs-glossary/nat/). Flattening to one subnet (AP / [bridge](/cs-glossary/bridge/) mode) is what fragile LAN protocols often need.
