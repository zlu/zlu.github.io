---
layout: post
title: "A Couple Things to Watch Out For Using PrivatePub"
date: 2012-02-12 20:38
comments: true
categories: [ruby]
---

[Private Pub](https://github.com/ryanb/private_pub) is a convinent gem wraps around [Faye](http://faye.jcoglan.com/ruby.html).

There are a few things to consider when using PrivatePub.

1. It does __not__ support ssl.
However, there is a [pull request](https://github.com/ryanb/private_pub/pull/33) for it.

2. You almost always want to add some sort of filtering on the server or client side, or maybe both.
For example, if you don't want your chat message to be received by everyone logged in, you will need to
publish to a channel uniquely identified by each session of the chat.

__UPDATE__ @rbates has merged this pull request.  I was able to verify ssl support indeed works!
