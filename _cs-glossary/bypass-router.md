---
title: Bypass router
synonyms:
  - Side router
  - Side gateway
  - 旁路网关
related:
  - NIC
  - LAN
  - NAT
  - OpenWrt
  - VPN
---
A **bypass router** (side router / 旁路网关) is a second box on the home [LAN](/cs-glossary/lan/) that does **not** replace the main Wi‑Fi / dial gateway. It takes a [NIC](/cs-glossary/nic/) into the existing LAN and runs services ([OpenWrt](/cs-glossary/openwrt/), [VPN](/cs-glossary/vpn/) client, [DLNA](/cs-glossary/dlna/), ad-blocking) while the [ONT](/cs-glossary/ont/) and Wi‑Fi AP keep PPPoE and radio duty.

Contrast gateway mode, where the new box becomes the only [NAT](/cs-glossary/nat/) edge. Bypass / side placement is what you want when CMCC already dials and the TP-Link already does Wi‑Fi.
