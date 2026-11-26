---
title: ExFAT
synonyms:
  - Extensible File Allocation Table
related:
  - FAT32
  - APFS
  - MBR
  - GPT
---
**ExFAT** is Microsoft’s flash-oriented filesystem: large files (no 4GB cap), large volumes, no journal. It is the usual “works on Mac and Windows” choice for USB movie disks.

Cheap gateways often claim ExFAT and still fail: broken drivers, or they mount the wrong [GPT](/cs-glossary/gpt/) slice (EFI) first. On OpenWrt-class boxes ExFAT is usually fine; on a CMCC [ONT](/cs-glossary/ont/) it may not be.
