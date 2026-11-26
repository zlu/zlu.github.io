---
title: EFI system partition
synonyms:
  - ESP
  - EFI partition
related:
  - GPT
  - MBR
---
The **EFI system partition (ESP)** is a small FAT volume [GPT](/cs-glossary/gpt/) disks use for boot loaders. On a USB movie disk it is useless cargo: a few “text-looking” files, tiny capacity.

If a gateway’s storage UI shows **NaN%** / almost empty after an ExFAT erase, it probably mounted `sda1` = ESP, not the data partition. Reformat with [MBR](/cs-glossary/mbr/) so `sda1` is the real volume.
