---
title: GPT
synonyms:
  - GUID Partition Map
  - GUID Partition Table
related:
  - MBR
  - EFI system partition
  - ExFAT
---
**GPT** (GUID Partition Table; Disk Utility’s **GUID Partition Map**) is the modern partition scheme. An ExFAT erase on GPT usually creates:

1. a small [EFI system partition](/cs-glossary/efi-system-partition/)  
2. the large data volume  

Routers that blindly mount **`sda1`** often attach the EFI slice instead of your movies. For dumb USB gateways, use [MBR](/cs-glossary/mbr/) with a single data partition instead.
