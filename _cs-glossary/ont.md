---
title: ONT
synonyms:
  - Optical Network Terminal
  - Optical modem
related:
  - LAN
  - WAN
  - PPPoE
  - NAT
---
An **ONT** (optical network terminal) is the customer-side box that terminates fiber from an ISP. In China Mobile (CMCC) homes it is often a combo unit: fiber modem + Wi‑Fi + Ethernet switch + optional USB.

Toward the ISP it speaks the carrier’s fiber / [PPPoE](/cs-glossary/pppoe/) path ([WAN](/cs-glossary/wan/) side). Toward your devices it presents a private [LAN](/cs-glossary/lan/). Many units also run [NAT](/cs-glossary/nat/) and DHCP, so a second Wi‑Fi router behind them creates [double NAT](/cs-glossary/nat/) unless you put that router in AP / bridge mode.
