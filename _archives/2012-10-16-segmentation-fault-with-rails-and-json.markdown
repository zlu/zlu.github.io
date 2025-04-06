---
layout: post
title: "Segmentation fault with Rails and JSON"
date: 2012-10-16 21:03
comments: true
categories: [Ruby, Rails, Gem, RVM]
---

After manually cleaning Ruby gems on my system, I got this error running `rails c` or `rails s`

{% highlight bash %}

zlu@zlu-mba:~/projects/foo (master *)$ rails c
/Users/zlu/.rvm/gems/ruby-1.9.3-p0/gems/json-1.7.5/lib/json/ext/parser.bundle: [BUG] Segmentation fault
ruby 1.8.7 (2012-02-08 patchlevel 358) [universal-darwin12.0]

Abort trap: 6

{% endhighlight %}

It is very weird if you think about it.  The `ruby -v` shows that the system is using ruby version 1.9.3 managed by rvm.
But the error shows ruby 1.8.7!  It must be something I did with uninstalling versions of gems where executable requirements got messed up.

Instead of imploding rvm, here's what I did to correct the problem.

{% highlight bash %}

gem list | cut -d" " -f1 | xargs gem uninstall -aIx
gem install bundler
bundle install

{% endhighlight %}

The first command uninstalls all the gems, including bundler.
After reinstalling bundler, you can run `bundle install`, assuming you are in a project directory where Gemfile.lock exists.