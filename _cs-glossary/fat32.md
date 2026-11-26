---
title: FAT32
synonyms:
  - MS-DOS (FAT)
  - File Allocation Table 32
related:
  - ExFAT
  - MBR
  - Remux
---
**FAT32** is an old cross-platform filesystem every USB stack still speaks. Hard limit: **one file ≤ 4GB**. That kills many Blu-ray [remuxes](/cs-glossary/remux/).

Useful as a smoke test that an [ONT](/cs-glossary/ont/) USB port and share path work. Not a movie-library format; prefer [ExFAT](/cs-glossary/exfat/) on a real [NAS](/cs-glossary/nas/) / OpenWrt box.
