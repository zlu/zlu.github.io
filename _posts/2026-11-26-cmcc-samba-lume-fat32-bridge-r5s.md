---
layout: post
title: CMCC box for media streaming
description: 'Trying USB Samba on a China Mobile Skyworth SK-D747 for home video:
  APFS, ExFAT, FAT32, double NAT, then NanoPi R5S.'
keywords: CMCC, ONT, optical modem, SK-D747, Samba, SMB, FAT32, ExFAT, APFS, double
  NAT, AP mode, NanoPi R5S, OpenWrt, iStoreOS, DLNA, Lume
comments: true
tags:
- networking
- home-lab
- media
- Lume
- OpenWrt
- China
date: 2026-11-26 12:12 +0800
---
I wanted a free, simple setup for video streaming at home using the optical modem/gateway CMCC gave me (Skyworth **SK-D747**). The box is an [ONT](/cs-glossary/ont/) with **two [USB](/cs-glossary/usb/) ports**, and I have a few old external HDDs and SSDs sitting around that could finally be media and file shares. I hit a few walls along the way. Here is what I learned.

The easy path on this class of modem is [Samba](/cs-glossary/samba/), which is usually bundled. Filesystem support killed that plan before Samba ever mattered.

## Mac filesystems

As a Mac user, my external drives were formatted for macOS (**[APFS](/cs-glossary/apfs/)** or **[Mac OS Extended/Journaled](/cs-glossary/hfs-plus/)**). The gateway runs a tiny embedded Linux. It does not mount those volumes in any useful way. The admin UI’s **Storage management** page showed **NaN%** capacity, a handful of “text” files, and **zero videos**, even though the disk had a large `movie` folder.

So the first fix was to reformat to **[ExFAT](/cs-glossary/exfat/)**.

## ExFAT and GUID

After erasing to ExFAT in Disk Utility, the modem still only saw a tiny slice of the disk. Movies stayed invisible.

On a Mac, **[GUID Partition Map](/cs-glossary/gpt/) + ExFAT** usually creates **two** partitions:

1. a small **[EFI](/cs-glossary/efi-system-partition/)** system partition  
2. the big ExFAT data volume  

Many cheap routers mount **`sda1` first**. That is often the EFI slice: small, a few odd files, looks like “text,” never your movies. The web UI is not a file manager; it reports whatever got mounted first.

## MBR in Disk Utility

I reformatted again with **[Master Boot Record (MBR)](/cs-glossary/mbr/)** so the disk has a single partition gateways like.

In Disk Utility that option is easy to miss:

1. **View → Show All Devices**  
2. Select the **whole disk** (e.g. “USB Disk”), not the indented volume under it  
3. Erase with **Scheme: Master Boot Record**, format **ExFAT**

Or from the terminal:

```sh
diskutil list   # find the external diskN - not disk0
diskutil eraseDisk ExFAT MOVIES MBRFormat /dev/diskN
```

## FAT32

**MBR + ExFAT** still gave **0B/NaN%** on this SK-D747. ExFAT support in that firmware is broken, or the disk/power path failed. I stopped trusting ExFAT on this box.

Out of options, I reformatted again to **MBR + MS-DOS (FAT)**/**[FAT32](/cs-glossary/fat32/)**. That finally showed real capacity and real video file counts.

Hard limit, and it matters for a movie library:

> **FAT32 max file size is 4GB.**  
> Many [remuxes](/cs-glossary/remux/) are larger.

At that point I only wanted to prove the USB ports were not dead. FAT32 worked for that. It is not a real library format.

## Double NAT

In Finder I tried `smb://192.168.1.1` (the CMCC [LAN](/cs-glossary/lan/) IP) and failed. The share lives on the ONT’s [subnet](/cs-glossary/subnet/). My Mac did not.

By design the house looks like this: CMCC does [PPPoE](/cs-glossary/pppoe/)/fiber and hands out `192.168.1.x`. A **[TP-Link TL-XDR5410](https://www.tp-link.com.cn/product_2568.html)** sits behind it as the real Wi‑Fi router, with its own LAN (`192.168.0.x`) and [NAT](/cs-glossary/nat/). Phones and the Mac sit on the TP-Link side.

```text
Internet ── SK-D747 192.168.1.1     ← USB disk/Samba live here
              │
              │ WAN
              ▼
            XDR5410 192.168.0.1     ← Wi‑Fi + DHCP for phones/Mac
              │
              ▼
            Mac 192.168.0.103
```

The TP-Link’s [WAN](/cs-glossary/wan/) port faces the ONT. Safari can still open `http://192.168.1.1` through that hop. Samba is pickier about LAN peers and port `445`. Crossing that NAT boundary puts the Mac off the ONT’s LAN, so SMB fails even when the admin UI still loads. In [OSI](/cs-glossary/osi-model/) terms: HTTP still routes at L3; SMB wanted on-link peers on the same L2/L3 LAN.

Fix for the experiment: turn off [DHCP](/cs-glossary/dhcp/) on the TP-Link and put it in [AP](/cs-glossary/access-point/)/[bridge](/cs-glossary/bridge/) mode. Clients then join the CMCC subnet (`192.168.1.x`) and talk to `192.168.1.1` as on-link peers. Samba can work in that layout.

## Why I am abandoning it

The bridged setup works as a smoke test. I am not keeping it as the home media path. CMCC USB storage is too weak for a real [NAS](/cs-glossary/nas/):

- Filesystem support is effectively **FAT32** (ExFAT flaky or absent)  
- Share UI is unclear (no obvious Samba enable page; ACL toggles; maybe [FTP](/cs-glossary/ftp/) only)  
- No usable [DLNA](/cs-glossary/dlna/) path for in-app browse in [Lume](https://zlu.me/lume)  
- Carrier firmware can change under you  
- USB power for old 2.5″ disks is unreliable without a powered hub  

It has USB ports. That does not make it a NAS.

## Hardware options

Keep the SK-D747 for **PPPoE/fiber**. Keep (or AP-mode) the XDR5410 for **Wi‑Fi**. Put USB media + optional [VPN](/cs-glossary/vpn/) on a small always-on box.

### NanoPi R5S

**[FriendlyElec](https://www.friendlyelec.com/) [NanoPi R5S](https://wiki.friendlyelec.com/wiki/index.php/NanoPi_R5S)** is a dual-[NIC](/cs-glossary/nic/) ARM [SBC](/cs-glossary/sbc/) people in China flash as a [side/bypass router](/cs-glossary/bypass-router/). A usable kit ([eMMC](/cs-glossary/emmc/)/case/PSU) is roughly **¥700-1100**.

Why it fits:

- Gigabit (or better) paths for video  
- USB for the disk  
- Low idle power  
- First-class [OpenWrt](/cs-glossary/openwrt/)/[ImmortalWrt](/cs-glossary/immortalwrt/) images  

Cheaper boards (e.g. [Orange Pi Zero 3](http://www.orangepi.org/html/hardWare/computerAndMicrocontrollers/details/Orange-Pi-Zero-3.html)) can run [MiniDLNA](/cs-glossary/dlna/) too; you get [USB 2](/cs-glossary/usb/) throughput and a less router-like board. An [Intel N100](https://www.intel.com/content/www/us/en/products/sku/231803/intel-processor-n100-6m-cache-up-to-3-40-ghz/specifications.html) mini PC is overkill if all you want is file serve + [WireGuard](/cs-glossary/wireguard/) later.

I would not replace the XDR5410 with another locked Wi‑Fi router just to get USB. I would **add** a small box for media and optional VPN.

### OpenWrt vs iStoreOS

[r5s.cooluc.com](https://r5s.cooluc.com/) style builds are **community OpenWrt** (often ImmortalWrt-based) aimed at the R5S. **[iStoreOS](/cs-glossary/istoreos/)** is also OpenWrt under the hood, with a China-oriented UI and an **iStore** app catalog.

| | OpenWrt (e.g. cooluc) | iStoreOS |
| --- | --- | --- |
| Base | OpenWrt/ImmortalWrt | Same family |
| UX | Classic [LuCI](/cs-glossary/luci/), more DIY | Appliance-like Chinese UI |
| Packages | `opkg`/software packages menu | One-click-ish apps (miniDLNA, etc.) |
| R5S images | Often excellent board-specific builds | Easy side-router setup |
| VPN later | WireGuard/[OpenVPN](/cs-glossary/openvpn/) as packages | Same, usually fewer clicks |

For me: **iStoreOS first** if the goal is USB → [miniDLNA](/cs-glossary/dlna/) → Lume browse without living in LuCI; **cooluc OpenWrt** if I want a lean image and I am fine installing packages myself. Either can do the job. Re-flashing later is cheap.

End state I actually want:

```text
ONT (dial) ── XDR5410 (AP/Wi‑Fi)
                │
                └── R5S (iStoreOS or OpenWrt)
                      USB disk (exFAT)
                      miniDLNA (+ Samba)
                      WireGuard client when needed
                            │
                            └── Lume: Browse network (UPnP)
```

Lume already plays network streams through VLCKit. The missing piece is **discovery** ([UPnP](/cs-glossary/upnp/)/[DLNA](/cs-glossary/dlna/) browse), not a better URL paste box. That needs a real DLNA box on the same home LAN, a filesystem the server understands, and no second NAT in the middle. The CMCC box was a free experiment. For a real library, something like an R5S is the right tool.
