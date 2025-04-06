---
layout: post
title: "Lost Shell Commands?"
date: 2013-05-27 21:32
comments: true
categories: [Shell, Bash]
---

Sometime when you modify .bashrc or .bash_profile and source it, you may notice that you have 'lost' your shell commands.
You will see 'command not found' for the simplest commands such as `ls` and `which`.

Most like this happens because you have accidentally override your PATH.  So instead of doing:

`PATH=my/custom/path:$PATH`, you did `PATH=my/custom/path`.

Here's a simpel way to fix it.

```
echo 'PATH=/bin:/usr/bin' > foo && source foo
```

This will get recover your commands so you can modify your .bashrc or .bash_profile with nano.  Remember to back up a working
version before working on them next time :)