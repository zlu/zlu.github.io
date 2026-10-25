---
layout: post
title: "When CleanMyMac Isn't Enough: Building My Own Disk Cleaner"
date: 2026-10-25
tags:
  - macos
  - disk-space
  - swift
  - developer-tools
  - product
description: "CleanMyMac missed my real disk hogs — a 175GB .next cache and a 60GB Cursor DB. I built a native macOS cleaner and reclaimed 500+ GB."
keywords: "CleanMyMac, macOS disk cleaner, disk space, Swift, .next cache, Turbopack, Cursor state.vscdb, Android SDK, CoreSimulator, developer tools"
comments: true
---

My 1TB MacBook Pro had a habit. Every few weeks it would throw the same orange warning: disk almost full. I'd open CleanMyMac, run the usual ritual, reclaim a few gigabytes of "system junk," feel briefly virtuous — and two days later hit **950 GB+** again.

CleanMyMac is fine at what it knows. Caches. Logs. Language files. But my disk was full of **developer archaeology** — folders with names I'd never think to search for, nested six levels under a sync root, quietly holding tens or hundreds of gigabytes.

So I started building my own native macOS cleaner. Not because the world needs another cleaner. Because I needed a tool that would show me *where the bytes actually went*, and which trees were safe to cut.

## The cleanup that never stuck

Before writing a line of Swift, I had already done the "responsible adult" checklist:

- Uninstalled apps I no longer used
- Moved ~**200 GB** of photos to an external drive
- Shoved a pile of files into a cloud drive
- Deleted local LLMs I'd downloaded but rarely used
- Emptied Trash, cleared Downloads, the whole folklore set

It was never enough. The free space would bounce up, then the used bar would crawl back toward the cliff. Something invisible was regenerating — or, more accurately, something large had always been there and none of the usual tools knew to look. `du -sh * | sort -h` could show me the sizes. It couldn't tell me which hundred-gigabyte trees were safe to delete — and at volume root it's slow, noisy, and still hits `Operation not permitted` on SIP/TCC-gated paths even with `sudo`.

## Build the map, then follow it

My app started as a space map: walk the Data volume, size the large folders, label what looks reclaimable vs. personal vs. leave-alone. Then it grew the features I kept needing while using my own dogfood:

- **Biggest wins** — ranked reclaim opportunities in plain language, not a passive folder list
- **Browse all contents** — brute-force drill-down when the "smart" map stopped too shallow
- **Guidance labels** — Reclaim / Archive / Protected, so I didn't trash the wrong tree
- **Similar photos** — Vision (and later PhotoKit) to cluster near-duplicates in the Photos library
- **Emulator / simulator inventory** — iOS CoreSimulator, Android AVDs, even Huawei images, with apps inside you can remove selectively
- **Android SDK package panel** — platforms, system images, NDK, CMake, build-tools, without opening Studio first
- **Honest sizing** — allocated bytes instead of sparse logical size (DerivedData once claimed to be ~40 GB; on disk it was ~300 MB), plus a size cache that invalidates ancestors when a child shrinks

That last one mattered more than it sounds. After deleting a huge cache, the parent folder's mtime often **doesn't change**, so a naive cache keeps serving the old multi‑hundred‑GB number. I was lied to about a project that was already cleaned. Fixing the cache was as important as finding the junk.

## The 175 GB that CleanMyMac never mentioned

The breakthrough was boring and humiliating in equal measure.

Deep under a sync folder → `projects` → a workspace → a Next.js app, a **`.next`** directory was sitting on roughly **175 GB**. Almost all of it was Turbopack / dev cache under `.next/dev`. Regenerable. Safe to delete. Invisible to CleanMyMac's idea of "system junk."

That one folder explained months of disk-full anxiety. I had been rearranging furniture while a wardrobe in the basement was full of concrete.

The other shocker was **Cursor**. Its global `state.vscdb` — a SQLite store for chat and agent history — had swollen to ~**60 GB**. Not photos. Not projects. A product database with a known growth problem. ⌘⇧P → **Developer: Delete Old Chats…** then **Developer: GC Agent KV Blobs** clawed some of it back (~**40 GB → ~21 GB**), but the file stayed absurd. I backed it up and nuked it to finish the reclaim.

Other discoveries:

- **Android SDK / tools** under `~/Library/Android` — NDK, CMake, emulator images from one-off experiments. Android Studio's SDK Manager hides half of them unless you tick **Show Package Details**, so I'd "cleaned" the list and still had gigabytes of ghosts
- **iOS simulators** with large installed apps and data (one iPhone sim alone was in the teens of GB)
- **Photos** still large after the external-drive move; similarity clustering made the remaining duplicates visible instead of abstract

Each of these lives in a different corner of the filesystem. A cleaner that only knows `/Library/Caches` will never see them. A volume map that lets you keep clicking does.

## What I reclaimed today (Oct 25)

Still iterating on the cleaner, still cleaning as I go. Today's notable wins and fixes:

| What | Roughly |
|------|---------|
| Android Studio SDK Manager cleanup (NDK and friends) | **~8.8 GB** in one pass; SDK tree went from ~**35 GB → ~10 GB** after further uninstalls |
| Cursor `state.vscdb` | GC ~**40 → ~21 GB**; then backup + nuke for the rest of the ~**60 GB** peak |
| Confirmed the old `.next` ghost | Parent project UI still showed **~155 GB** after delete; on disk it was already **~18 GB** — cache bug, not a second 128 GB |
| DerivedData "40 GB" scare | Measurement fix; real footprint ~**300 MB** |

Product work today that will save the *next* round: Refresh that actually remeasures, an Android "installed packages" breakdown, simulator app drill-down, and collapsing the Archive view so parent/child folders aren't listed as peers (Users → Home → sync → projects used to look like four separate 100 GB problems).

## The scoreboard so far

Between the early checklist (photos off-box, cloud moves, LLM deletes) and finds from the map (`.next`, Android SDK, simulators, editor databases, and ongoing browse sessions), I've clawed back **500+ GB**. The Mac currently has hundreds of gigabytes free. I'm still going — there's more under Library, more under sync roots, more in simulators I haven't opened in a year.

## What I learned building the tool I needed

1. **Generic cleaners optimize for common cases.** Developer machines are uncommon cases wearing common clothes.
2. **Hierarchy beats flat "largest folders."** If you sort every nested path by size, you double-count the same bytes and hide the structure. Show the top of a subtree; drill for the rest.
3. **Caches lie in two directions.** Apps cache build artifacts forever. Your *disk tool* can also cache stale sizes. Both will gaslight you.
4. **Safety labels matter.** "Safe regenerable" (`.next`, DerivedData) is a different action from "archive to external" (photos, sync trees) and from "leave alone" (System, VM, Preboot). One Trash button for all three is how people ruin their weekend.
5. **Dogfooding is the product roadmap.** Every confusing screen was a screen I got stuck on while trying to free space on this same laptop.

I don't know yet whether this becomes something I ship to other people. For today it's enough that the warning banner is gone — and that I finally know *why* it kept coming back.

If your Mac is "mysteriously" full and you've already done the CleanMyMac dance: don't buy a bigger SSD first. Buy an afternoon of curiosity. The 175 GB file is probably named something you've never googled.
