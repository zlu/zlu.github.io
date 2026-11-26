---
title: Samba
synonyms:
  - SMB
  - Server Message Block
  - CIFS
related:
  - LAN
  - NAT
  - TCP
  - Subnet
---
**Samba** is the common open-source implementation of **SMB** (Server Message Block), the Windows / macOS file-share protocol. Clients connect with URLs like `smb://192.168.1.1` (TCP port **445**).

The wire protocol is defined in Microsoft’s open specs: **[MS-SMB2](https://learn.microsoft.com/en-us/openspecs/windows_protocols/ms-smb2/5606ad47-5ee0-437a-817e-70c366052962)** (SMB 2/3) and the older [MS-SMB](https://learn.microsoft.com/en-us/openspecs/windows_protocols/ms-smb/f2104663-0795-4c2b-be05-53b8eab4a694) / [MS-CIFS](https://learn.microsoft.com/en-us/openspecs/windows_protocols/ms-cifs/d416ff7c-378e-452b-be39-bc6f2648b08b). The practical implementation and man pages live at **[samba.org](https://www.samba.org/)** (e.g. [`smb.conf`](https://www.samba.org/samba/docs/current/man-html/smb.conf.5.html)).

Consumer gateways sometimes ship a minimal `smbd` or only FTP. SMB assumes [LAN](/cs-glossary/lan/) peers: same [subnet](/cs-glossary/subnet/), port 445 open, and often no second [NAT](/cs-glossary/nat/) in the path. macOS Finder “Connect to Server” is a thin client over that stack.
