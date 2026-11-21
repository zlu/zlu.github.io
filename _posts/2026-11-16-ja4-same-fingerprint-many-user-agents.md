---
layout: post
title: "JA4 Fingerprinting: Why One Digest Can Cover Hundreds of User-Agents"
date: 2026-11-16
tags:
  - security
  - bots
  - vercel
  - tls
  - waf
  - betterreads
description: "JA4 hashes the TLS ClientHello, not the HTTP User-Agent. One digest can cover hundreds of UAs. Algorithm, Python code, JA4T/JA4TS, and Vercel Firewall rules from a BetterReads farm."
keywords: "JA4 fingerprint, JA4 algorithm, JA4 Python, JA4T, JA4TS, JA3, TLS ClientHello, User-Agent spoofing, mbedtls, quiche, Vercel Firewall, bot protection, Windows Defender, BetterReads, WAF rules"
comments: true
---

[JA4](/cs-glossary/ja4/) is a digest of the [TLS](/cs-glossary/tls/) **[ClientHello](/cs-glossary/clienthello/)**. The HTTP [User-Agent](/cs-glossary/user-agent/) is not an input. That is why one JA4 can cover hundreds of different UA strings, and why spoofing the header never splits the TLS bucket.

If you work in threat hunting, bot detection, or [WAF](/cs-glossary/waf/) ops, you have probably seen this confusing signal:

> One single JA4 fingerprint. Thousands of requests. Dozens, sometimes hundreds, of different User-Agent strings.

Below: the JA4 algorithm, a Python implementation, how JA4 differs from JA4T / JA4TS (the TCP pair often shown next to it in Defender-style tooling), then a BetterReads incident from mid-November 2026 where one digest hit **~13k/day** and I had to decide whether to Deny it.

Related field notes from the earlier farm wave: [Fighting Scraping Farms on a Reading Site](/2026/10/26/fighting-scraping-farms-on-a-reading-site/).

## JA4 algorithm

JA4 is a ClientHello fingerprint, a successor to [JA3](/cs-glossary/ja3/). JA3 broke easily because modern browsers randomize the order of TLS extensions. JA4 fixes that by sorting key fields before hashing, so fingerprints stay stable against extension reordering.

A JA4 has three parts joined by `_`:

```text
JA4_a _ JA4_b _ JA4_c
```

Example from production traffic:

```text
t13d1515h1_8daaf6152771_de4a06bb82e3
```

### JA4_a: plaintext metadata prefix

High-level handshake properties:

1. Transport: `t` = [TCP](/cs-glossary/tcp/)/TLS, `q` = [QUIC](/cs-glossary/quic/)
2. Advertised TLS version (`13` = TLS 1.3, `12` = TLS 1.2)
3. [SNI](/cs-glossary/sni/) flag: `d` = SNI present, `i` = absent
4. Count of [cipher suites](/cs-glossary/cipher-suite/) the client offered
5. Count of TLS extensions in the ClientHello
6. First [ALPN](/cs-glossary/alpn/) preference (`h2`, `h1`, `h3`, or `00` if ALPN is missing)

Important limits:

- The SNI flag is **presence only**. JA4 does not hash the hostname inside SNI.
- ALPN uses only the **first** protocol the client prefers, not the full list, and not what the server eventually picks.

### JA4_b: hash of sorted cipher suites

The client lists cipher suites it supports. JA4 sorts those numeric IDs, joins them, then takes a truncated SHA-256. JA3 used the original order; JA4's sort means random ordering no longer changes the fingerprint.

### JA4_c: extension types + signature algorithms

This is where most analysts trip up.

JA4 works with **TLS extension type IDs**, not the payload inside each extension.

Processing for JA4_c:

1. Take all extension type IDs from the ClientHello
2. Remove SNI (`0x0000`) and ALPN (`0x0010`); already represented in JA4_a
3. Strip [GREASE](/cs-glossary/grease/) reserved values
4. Sort the remaining extension type IDs
5. Append the signature-algorithms list (**not** sorted)
6. Hash and truncate

Example: `supported_groups` (`0x000a`) tells the server which curves the client likes. JA4 only cares that extension `0x000a` **exists**. Two clients can advertise different curves inside that extension and still share the same JA4.

## JA4 in Python

The FoxIO [JA4 spec](https://github.com/FoxIO-LLC/ja4/blob/main/technical_details/JA4.md) is short enough to implement from already-parsed ClientHello fields. User-Agent is not an input. That is the whole point of the "same JA4, many UAs" case.

```python
import hashlib

# GREASE values look like 0x?a?a (e.g. 0x0a0a, 0x1a1a, ...).
def is_grease(value: int) -> bool:
    return (value & 0x0F0F) == 0x0A0A


def truncated_sha256(payload: str) -> str:
    if not payload:
        return "000000000000"
    return hashlib.sha256(payload.encode("ascii")).hexdigest()[:12]


def alpn_chars(first_alpn: str | None) -> str:
    if not first_alpn:
        return "00"
    if len(first_alpn) == 1:
        return first_alpn[0] + first_alpn[0]
    return first_alpn[0] + first_alpn[-1]


def ja4(
    *,
    transport: str = "t",  # "t" TCP/TLS, "q" QUIC
    tls_version: str = "13",  # from supported_versions, else legacy
    sni_present: bool,
    cipher_suites: list[int],
    extension_types: list[int],
    signature_algorithms: list[int],
    first_alpn: str | None,
) -> str:
    ciphers = [c for c in cipher_suites if not is_grease(c)]
    extensions = [e for e in extension_types if not is_grease(e)]

    ja4_a = (
        f"{transport}{tls_version}"
        f"{'d' if sni_present else 'i'}"
        f"{min(len(ciphers), 99):02d}"
        f"{min(len(extensions), 99):02d}"
        f"{alpn_chars(first_alpn)}"
    )

    cipher_list = ",".join(f"{c:04x}" for c in sorted(ciphers))
    ja4_b = truncated_sha256(cipher_list)

    # SNI (0000) and ALPN (0010) are already in JA4_a; drop them from JA4_c.
    ext_for_hash = sorted(e for e in extensions if e not in (0x0000, 0x0010))
    ext_list = ",".join(f"{e:04x}" for e in ext_for_hash)
    sig_list = ",".join(f"{s:04x}" for s in signature_algorithms)
    ja4_c_payload = ext_list + (f"_{sig_list}" if sig_list else "")
    ja4_c = truncated_sha256(ja4_c_payload)

    return f"{ja4_a}_{ja4_b}_{ja4_c}"


# Cipher list from the FoxIO example. Sorted hex string hashes to 8daaf6152771,
# which is JA4_b in the BetterReads digest below.
EXAMPLE_CIPHERS = [
    0x1301, 0x1302, 0x1303, 0xC02B, 0xC02F, 0xC02C, 0xC030,
    0xCCA9, 0xCCA8, 0xC013, 0xC014, 0x009C, 0x009D, 0x002F, 0x0035,
]

assert truncated_sha256(
    ",".join(f"{c:04x}" for c in sorted(EXAMPLE_CIPHERS))
) == "8daaf6152771"

# Same TLS fields, two different HTTP User-Agents: same JA4.
digest = ja4(
    sni_present=True,
    cipher_suites=EXAMPLE_CIPHERS,
    extension_types=[
        0x001B, 0x0000, 0x0033, 0x0010, 0x0017, 0x002D, 0x000D,
        0x0005, 0x0023, 0x0012, 0x002B, 0xFF01, 0x000B, 0x000A, 0x0015,
    ],
    signature_algorithms=[
        0x0403, 0x0804, 0x0401, 0x0503, 0x0805, 0x0501, 0x0806, 0x0601,
    ],
    first_alpn="http/1.1",
)
print(digest)
# -> t13d1515h1_8daaf6152771_de4a06bb82e3
# Same digest as the BetterReads top JA4 below. User-Agent is not an input.
```

GREASE filtering, QUIC vs TCP, and the exact extension set of a given browser build still matter at the edge. The sample is enough to see why rotating the HTTP header cannot split the TLS bucket.

## What JA4 completely ignores

This list is the key to "same JA4, many UAs":

| Item | Layer | In JA4? | Notes |
| --- | --- | --- | --- |
| SNI | TLS extension | Yes | Presence flag only, not hostname |
| ALPN | TLS extension | Yes | First preference only |
| Cipher suites | TLS | Yes | Sorted before hashing |
| Extension type IDs | TLS | Yes | Sorted; SNI/ALPN excluded from JA4_c |
| Extension payloads | TLS | **No** | Ignored entirely |
| Signature algorithms | TLS | Yes | Not sorted |
| User-Agent | HTTP header | **No** | Separate fingerprint family ([JA4H](/cs-glossary/ja4h/)) |
| Other HTTP headers | HTTP | **No** | JA4H territory |
| IP / path / cookies | n/a | **No** | Orthogonal signals |

JA4 lives entirely at the TLS layer. User-Agent is an HTTP header sent **after** the handshake. They are decoupled by design.

## Why one JA4 maps to many User-Agents

A JA4 is a **TLS stack configuration profile**, not a device, browser, or person.

Common reasons the bucket splits at HTTP:

1. **Same TLS library, custom UA.** Go `crypto/tls`, OpenSSL, BoringSSL, mbedTLS, OkHttp, Cronet, Cloudflare quiche (and quiche-java): override `User-Agent`, leave ClientHello alone. Switch the library or its cipher/extension config and the JA4 changes; rewrite the header and it does not.
2. **Headless automation.** Playwright / Puppeteer / Selenium spoof arbitrary UAs; Chromium's TLS stack stays put, so JA4 stays stable.
3. **Shared mobile SDKs.** Many apps, one networking stack, app-specific UAs.
4. **Proxies.** Forward the original ClientHello, rewrite HTTP headers after encryption is up.

Operational takeaway: **do not treat JA4 as a unique client ID.** Treat it as "what ClientHello is talking to me." One bucket can hold legitimate browsers, SDKs, scrapers, and malware that happen to share a TLS stack. "Remove JA4 fingerprinting" from a custom stack (mbedTLS, a .NET TLS client, a VIPRE or other security agent, BIND-adjacent TLS frontends) really means change the ClientHello the library emits. There is no HTTP-header knob that erases the digest.

## JA4T and JA4TS (not the same as JA4)

SOC and endpoint tools (including Windows Defender / Microsoft Defender threat intel that surfaces FoxIO labels) often show **JA4**, **JA4T**, and **JA4TS** together. They are different layers:

| Name | Layer | Built from |
| --- | --- | --- |
| **JA4** | TLS | ClientHello (this post) |
| **JA4S** | TLS | ServerHello / session side |
| **[JA4H](/cs-glossary/ja4h/)** | HTTP | Request header names / order / flags |
| **JA4T** | TCP | Client SYN (window size, option order, MSS, scale) |
| **JA4TS** | TCP | Server SYN-ACK response |

JA4T fingerprints the OS / TCP stack, not the browser TLS config. FoxIO's Windows 11 example is `JA4T=64240_2-1-3-1-1-4_1460_8`. JA4TS is the server's SYN-ACK; the same host can emit more than one JA4TS depending on which TCP options the client offered.

So when a Defender alert or a WAF UI shows "JA4 T" and "JA4 TS," that is the TCP pair, not a broken spelling of the TLS digest. Clustering bots on BetterReads used JA4 (TLS) plus path and UA. JA4T would be the next pivot if I needed OS-stack confirmation that the TLS digest alone did not give.

## BetterReads, Nov 2026: 32k logged bots, top JA4 ~13k

[BetterReads](https://www.betterreads.dev) sits behind **Vercel Firewall**. I keep **Bot Protection on Log** (not Challenge). Challenge Mode has bitten good crawlers with `429` Security Checkpoint pages before. Logging still surfaces the noisy digests; it does **not** stop Fluid/SSR. Mitigated traffic (`x-vercel-mitigated: deny`) is what actually saves compute.

In one 24h window I saw roughly:

- **~32k** Bot Protection **Logged** requests (Denied: 0)
- Top JA4 alone: **`t13d1515h1_8daaf6152771_de4a06bb82e3`** ≈ **11-13k**
- Top User-Agents under Bot Protection: fragmented at **~1-2k each**

If you only look at "Top UA," you see seven mediocre offenders. If you look at JA4, you see one TLS stack driving most of them.

### Inspect without Observability Plus

Custom Query in the dashboard is read-only without Observability Plus. The last **24h** of Firewall Traffic is still available via CLI:

```bash
vercel firewall traffic inspect ja4 t13d1515h1_8daaf6152771_de4a06bb82e3 \
  --group-by user-agent --top 20 --json

vercel firewall traffic inspect ja4 t13d1515h1_8daaf6152771_de4a06bb82e3 \
  --group-by path --top 20

vercel firewall traffic inspect ja4 t13d1515h1_8daaf6152771_de4a06bb82e3 \
  --group-by asn --top 20

vercel firewall traffic inspect ja4 t13d1515h1_8daaf6152771_de4a06bb82e3 \
  --group-by ip --top 20
```

One `--group-by` at a time. Multiple flags collapsed to a single breakdown in my CLI version.

### What that digest actually was

| Dimension | Finding |
| --- | --- |
| Volume | **13.3k** requests (≈2.2k Allow + ≈11.1k Log) |
| User-Agents | **Seven** Windows Chrome strings, **~1.9k each**. Not one UA |
| Paths | Almost entirely deep `/discover/<book>/…/chapters/…` (and characters). Unique URLs, so top path only shows 5-7 hits each, but the pattern is a chapter walk |
| [ASNs](/cs-glossary/asn/) | Comcast, VNPT, Claro, Charter, Telefónica, Verizon, Maroc Telecom, Starlink…: **residential / ISP**, not a single OVH rack |
| IPs | Max **~5** hits per address. Classic rotating egress |

Full UA set under that JA4 (equals-match candidates):

```text
Mozilla/5.0 (Windows NT 10.0; Win64; x64) … Chrome/107.0.0.0 Safari/537.36
Mozilla/5.0 (Windows NT 10.0; Win64; x64) … Chrome/101.0.4951.67 Safari/537.36
Mozilla/5.0 (Windows NT 10.0; Win64; x64) … Chrome/110.0.0.0 Safari/537.36
Mozilla/5.0 (Windows NT 10.0; Win64; x64) … Chrome/100.0.4896.75 Safari/537.36
Mozilla/5.0 (Windows NT 10.0; Win64; x64) … Chrome/99.0.4844.51 Safari/537.36
Mozilla/5.0 (Windows NT 10.0; Win64; x64) … Chrome/104.0.0.0 Safari/537.36
Mozilla/5.0 (Windows NT 10.0; Win64; x64) … Chrome/116.0.0.0 Safari/537.36
```

Ancient majors, several pure `.0.0.0` builds, rotating under one Chromium-like ClientHello. Exactly the "same JA4, many UAs" case, except the HTTP identity is a farm label, not seven different products.

Same week I also saw a separate **fixed-UA** chapter walker at much higher volume (`X11 Linux Chrome/157.0.0.0` on the order of **~161k**/day in an earlier window). That one is a different playbook: exact UA Deny. The JA4 bucket above is the opposite problem: UA Deny alone only buys ~2k per string.

## How I chose the Firewall rules

Decision order I actually used:

### 1. Do not bare-Deny the JA4

The digest is a **common Chrome TLS ClientHello**. Residential ASNs mean real readers can share JA4_a/b/c with the farm. Denying the digest alone is a blunt instrument with collateral risk.

### 2. Do Deny JA4 **and** path

The path breakdown was decisive. Almost every top URL was a deep Discover chapter/character page: the SEO surface farms love, and the surface that burns SSR.

**Rule A, primary cut (~13k of this traffic):**

- **If:** JA4 Digest **equals** `t13d1515h1_8daaf6152771_de4a06bb82e3`  
  **AND** Request Path **starts with** `/discover`
- **Then:** Deny

Natural-language form that works in the Vercel UI:

> Deny requests when JA4 Digest equals t13d1515h1_8daaf6152771_de4a06bb82e3 AND Request Path starts with /discover

### 3. Exact-UA mop-up for leftovers

JA4+`/discover` misses the same stack on `/`, `/blog`, `/reader`, APIs, etc. The seven strings are stable enough to Deny by **equals** (not `contains`. Never `contains Chrome/110` in late 2026).

**Rule B, mop-up:**

- **If:** User-Agent **equals** any of the seven strings above  
- **Then:** Deny

### 4. Keep Bot Protection on Log

Flipping the managed rule to Challenge was tempting (32k "bots" in purple). I didn't. Logged ≠ free; Challenged ≠ free of false positives on crawlers I want. Custom Deny with a probe (`403` + `x-vercel-mitigated: deny`) is the cost lever.

### 5. Don't confuse search indexers with farms

Same week, `YouBot` and `ShapBot` showed up in UA lists. Those are **You.com / Parallel** search-index crawlers. Allow HTML, block EPUB/expensive APIs in app policy. FW Deny would save compute and delete discoverability. Different bucket from Chrome/107 labels.

### 6. Fixed high-volume UAs still get their own rules

When volume concentrates on one impossible string (`Chrome/157.0.0.0` on X11 Linux, twin Mac `Chrome/133.0.0.0`), exact UA Deny is correct and simpler than JA4. Use JA4 when the farm **rotates** UAs under one handshake.

## Clustering better than JA4 alone

When a high-volume JA4 fans out across many UAs, break the bucket:

| Signal | Role |
| --- | --- |
| **Path** | Often the cheapest truth (deep catalog walks vs homepage) |
| **User-Agent** | Secondary label. Great for equals Deny once the set is known |
| **ASN / IP** | Hosting vs residential; never AS-ban Comcast because a farm rented it |
| **[JA4H](/cs-glossary/ja4h/)** | HTTP header name/order/casing. Splits spoofed UA families when you have it |
| **App policy** | Defense in depth after the edge (`trafficPolicy` on BetterReads); middleware 403 still bills Fluid |

JA4 says "same ClientHello." Path + UA + ASN say "same abuse campaign."

## Cheat sheet

| Instinct | Better move |
| --- | --- |
| "Top JA4 = ban it" | Inspect UA / path / ASN first |
| "Top UA = ban only that" | Check whether one JA4 owns many UAs |
| "Bot Protection Logged = safe" | Still pays SSR until custom Deny |
| "Challenge everything" | Easy to break good crawlers |
| "AS Deny the ISPs in the chart" | Hits real readers; prefer JA4+path or exact UA |
| "JA4T = JA4 typo" | JA4T/JA4TS are TCP; JA4 is TLS ClientHello |

## Short FAQ

**Is JA4 a browser or device ID?**  
No. It is a TLS stack profile from the ClientHello.

**Does changing User-Agent change JA4?**  
No. UA is HTTP. JA4 is TLS.

**What is JA4T vs JA4TS in Windows Defender?**  
JA4T = client TCP SYN fingerprint. JA4TS = server SYN-ACK fingerprint. Separate from the TLS JA4 digest.

**Can I compute JA4 in Python?**  
Yes. See [JA4 in Python](#ja4-in-python) above. Feed already-parsed ClientHello fields; do not hash the UA string.

**Do mbedTLS, quiche, or .NET clients share Chrome's JA4?**  
Only if they emit the same ClientHello fields. Different libraries usually mean different digests. Same library with a custom UA still shares the digest.

---

*Numbers from Vercel Firewall Traffic inspect on BetterReads production, ~15-16 Nov 2026 UTC. Rule text may evolve as farms rotate digests and UAs.*
