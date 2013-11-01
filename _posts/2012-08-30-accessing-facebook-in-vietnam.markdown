---
layout: post
title: "Accessing Facebook in Vietnam"
date: 2012-08-30 21:56
comments: true
categories: [DNS, Censorship]
---

I just traveled to Saigon, Vietnam.  After checking hotel, I logged onto Wifi and browsed to Facebook.
It won't load!

```
zlu@zlu-mba:~$ ping www.facebook.com
ping: cannot resolve www.facebook.com: Unknown host
```

Hmm, that was strange.  A quick google search shows that some ISPs block Facebook due to government regulation.  But the
fix is quit easy.  All you need to do is to change your DNS server to Google DNS (8.8.8.8).

In OSX, open System Preferences -> Network -> DNS, add 8.8.8.8 to DNS servers (making sure it's the first in the list).

```
zlu@zlu-mba:~$ ping www.facebook.com
PING www.facebook.com (66.220.153.70): 56 data bytes
64 bytes from 66.220.153.70: icmp_seq=0 ttl=238 time=269.573 ms
```

Now you can browse Facebook again.