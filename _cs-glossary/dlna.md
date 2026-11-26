---
title: DLNA
synonyms:
  - Digital Living Network Alliance
  - MiniDLNA
  - ReadyMedia
related:
  - UPnP
  - LAN
  - Samba
  - NAS
---
**DLNA** (Digital Living Network Alliance) is a media-sharing profile on top of [UPnP](/cs-glossary/upnp/) AV: a server on the [LAN](/cs-glossary/lan/) advertises libraries; TVs and apps browse folders and play streams without pasting URLs.

**MiniDLNA** ([ReadyMedia](https://minidlna.sourceforge.net/)) is the common lightweight DLNA server on [OpenWrt](/cs-glossary/openwrt/) / [iStoreOS](/cs-glossary/istoreos/). It needs a readable filesystem on the server and clients on the same LAN (no second [NAT](/cs-glossary/nat/) in the middle). That is the discovery path [Lume](https://zlu.me/lume) wants, beyond Finder [Samba](/cs-glossary/samba/) mounts.
