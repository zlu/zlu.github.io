---
layout: post
title: "Site Updates"
date: 2013-11-01
comments: true
categories: [update]
---

This is a _major_ update.  I've finally decided to move away from [Octopress](http://octopress.org) and use the good old [Jekyll](http://jekyllrb.org) instead.  After using Octopress for _exactly_ 2 years (since 2011/11/01), I feel the urge to go back to the basics.  It's much easier for me to directly poke into Jekyll without the extra layer of abstraction and somewhat, complication that Octopress provides.  Many Octopress-powered sites inherit the default look, which gets old quickly.  Even though Octopress provides decent customization, the basic layouts are not easy or intuitive to change.  It's easier for me to just customize plain liquid template.

The migration process isn't hard.  I did have to global replace `codeblock` to `highlight` and change some image plugin markups.  I've also lost all the sidebar widgets such as flickr and twitter.  But I don't intend to bring them back.  I don't think they provide much value to the blog.  It was fun to look at them for a while though.

I disliked the solarized theme of syntax highlight.  I'm glad that is gone.  I'll introduce a lighter theme supported by `pigments` in the future.

The Disqus-based commenting is also gone.  I may bring commenting back, in some form.  Being a static site, Jekyll can only support:

- Javascript based, hosted solution like Disqus
- Mail parser, where comments come in as emails with blog_id as identifier
- Twitter message, as used by [Roon](http://roon.io)

I also don't know if the old comments can be easily restored - it really depends on how Disqus stores comments in relationship to blog entries.
