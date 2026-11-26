---
title: MBR
synonyms:
  - Master Boot Record
  - MBRFormat
related:
  - GPT
  - EFI system partition
  - ExFAT
  - FAT32
---
**MBR** (Master Boot Record) is the legacy disk partition scheme: one partition table in the first sector, at most four primary partitions. Many embedded USB implementations expect a single MBR data partition and mount `sda1` as that volume.

On macOS Disk Utility, pick the **whole disk** (**View → Show All Devices**), then **Scheme: Master Boot Record**. CLI: `diskutil eraseDisk … MBRFormat /dev/diskN`. Prefer MBR over [GPT](/cs-glossary/gpt/) for picky router/ONT USB shares.
