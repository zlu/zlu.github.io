---
layout: post
title: "Uninstall Multiple Versions of Multiple Ruby Gems"
date: 2012-07-31 10:13
comments: true
categories: [ruby, bash, gem]
---

Let's say you have multiple versions of spree gems installed and want to get rid of the older versions.

{% highlight bash %}
zlu@zlu-mba: gem list spree
spree (1.1.3, 1.0.0, 1.0.0.rc2)
spree_api (1.1.3, 1.0.0, 1.0.0.rc2)
spree_auth (1.1.3, 1.0.0, 1.0.0.rc2)
spree_cmd (1.1.3, 1.0.0, 1.0.0.rc2)
spree_core (1.1.3, 1.0.0, 1.0.0.rc2)
spree_dash (1.1.3, 1.0.0, 1.0.0.rc2)
spree_promo (1.1.3, 1.0.0, 1.0.0.rc2)
spree_sample (1.1.3, 1.0.0, 1.0.0.rc2)
{% endhighlight %}

Instead of doing this a bunch of times (for different gems and versions):

{% highlight bash %}
gem uninstall spree -v 1.0.0
{% endhighlight %}

Do this:

{% highlight bash %}
gem uninstall spree{,_api,_auth,_cmd,_core,_dash,_promo,_sample} -v {1.0.0, 1.0.0.rc2}
{% endhighlight %}

This is a feature of bash shell - [brace expansion](http://www.gnu.org/software/bash/manual/html_node/Brace-Expansion.html#Brace-Expansion).