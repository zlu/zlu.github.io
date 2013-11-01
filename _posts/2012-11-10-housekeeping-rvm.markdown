---
layout: post
title: "Housekeeping RVM"
date: 2012-11-10 18:13
comments: true
categories: [Ruby, RVM]
---

When a new version of Ruby comes out, I like to use RVM to install it, migrate ruby gems to new ruby, and remove the
older version of ruby.  This prevent the disk bloated with different versions of Rubies and duplicate gem(sets).

First, always update the RVM because it'll also update known list of Ruby.

``` bash
rvm get stable
```

Then list known versions of Rubies

``` bash
rvm list known

...
[ruby-]1.9.3-p286
[ruby-]1.9.3-[p327]
[ruby-]1.9.3-head
[ruby-]2.0.0-preview1
...
```

The output shows abridged result since the full list is long.  I currently have p286 installed and I'm about to install p327
because it contains a security fix for DoS attack in p286.

``` bash
rvm install 1.9.3
```

Since p327 is the default stable version (as indicated by []), the command above is sufficient.

Now you can migrate the gem(sets)

``` bash
rvm migrate 1.9.3-p286 1.9.3-p327
```

As part of the migration, you could opt to remove old ruby version.  Then check the disk usage:

``` bash
rvm disk-usage all
```

Note
----

[RVM](https://github.com/wayneeseguin/rvm)