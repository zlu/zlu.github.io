---
title: ASN
synonyms:
  - Autonomous System Number
  - Autonomous System
related:
  - WAF
  - JA4
  - User-Agent
---
An **ASN** (Autonomous System Number) identifies a network on the public Internet (an ISP, cloud provider, or enterprise). Each public IP maps to one ASN via BGP. Logs and [WAFs](/cs-glossary/waf/) often group abuse by ASN.

Hosting ASNs (AWS, Hetzner, OVH) are common scraper egress. Residential ISPs (Comcast, Verizon, Starlink, and similar) also show up when farms rent residential proxies. AS-wide Deny on a big ISP hits real users; prefer tighter rules (exact UA, JA4+path) when the ASN chart is mixed residential.
