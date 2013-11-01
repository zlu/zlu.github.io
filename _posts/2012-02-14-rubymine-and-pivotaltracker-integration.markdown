---
layout: post
title: "RubyMine and PivotalTracker Integration"
date: 2012-02-14 20:32
comments: true
categories: [tools]
---

**Task Server** is a lesser known feature in RubyMine.  You can enable PivotalTracker (PT) integration by adding
PT project as a Task Server.

Press cmd + , to open Preferences.  Type task to see this panel:

![](http://f.cl.ly/items/0q0C3G471G1K3R2D0r1V/Screen%20Shot%202012-02-14%20at%208.56.43%20PM.png)

Add PivotalTracker project by supplying project ID and api token.

Now every story can be easily turned into a RubyMine changelist.
Press cmd + shift + a, and type task.  Select open task.  Type part of the story to see a list of matched tracker stories.

![](http://f.cl.ly/items/2O1I0v3M403D2W1N2P0X/Screen%20Shot%202012-02-14%20at%208.59.44%20PM.png)

Select the desired story to work on and check start story.  Now every change will be in this change list and RubyMine will
autostart the story.

Let's say you TDD'ed and finished the story.  Press cmd + k to open changelist view.

![](http://f.cl.ly/items/0w1K2E1E3S3S2m301W1Q/Screen%20Shot%202012-02-14%20at%209.03.29%20PM.png)

You will see that commit message has been filled with story text.  Commit the change and RubyMine will automatically
finish the story.  How cool is that?  Now try to do that in Emacs or VI, I dare you :P

P.S. I like Emacs.