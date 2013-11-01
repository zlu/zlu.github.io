---
layout: post
title: "Install Native Ruby Gem in Mountain Lion Preview"
date: 2012-02-21 13:59
comments: true
categories: [ruby, os x, gem, mountain lion]
---

In Preview of OS X Mountain Lion, XCode has been distributed as a .app package verses the traditional installer.
Also the Commandline Tools are not installed by default.  Commandline Tools contains cc/gcc that you will need.

After download the preview of XCode 4.4, drop it to the Application directory.

Open XCode and cmd + , to open Preferences panel where you can install Commandline Tools:

![](https://img.skitch.com/20120221-r2cidbjd92nh9tmswq19hj6rdc.jpg)

Now in shell you should be able to `locate cc` and `locate gcc`

Now if you try to `gem install hpricot` or any gem that requires native extension (c), you may encounter problems
looks like this:

<pre>

checking for main() in -lc... *** extconf.rb failed ***
Could not create Makefile due to some reason, probably lack of
necessary libraries and/or headers.  Check the mkmf.log file for more
details.  You may need configuration options.

Provided configuration options:
        --with-opt-dir
        --without-opt-dir
        --with-opt-include
        --without-opt-include=${opt-dir}/include
        --with-opt-lib
        --without-opt-lib=${opt-dir}/lib
        --with-make-prog
        --without-make-prog
        --srcdir=.
        --curdir
        --ruby=C:/ruby/bin/ruby
        --with-hpricot_scan-dir
        --without-hpricot_scan-dir
        --with-hpricot_scan-include
        --without-hpricot_scan-include=${hpricot_scan-dir}/include
        --with-hpricot_scan-lib
        --without-hpricot_scan-lib=${hpricot_scan-dir}/lib
        --with-clib
        --without-clib

</pre>

If you cat the mkmf.log, you may see something like this:

"gcc-4.2 -o conftest ...."

So the fix is not to add configuration options but to simply create sym link of gcc-4.2:
{% highlight bash %}
ln -s /usr/bin/gcc /usr/bin/gcc-4.2
{% endhighlight %}