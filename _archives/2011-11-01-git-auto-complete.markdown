---
layout: post
title: "Git auto complete"
date: 2011-11-01 22:04
comments: true
categories: git
---

Git auto complete is a convenient feature.

This feature is based on git-completion file comes with git.

On OS X, if you install git via homebrew, you can find git source directory using:

{% highlight bash %}
locate git | grep Cellar
{% endhighlight %}

Locate the git-completion file.

In my case, it is `/usr/local/Cellar/git/1.7.5/etc/bash_completion.d/git-completion.bash

Open this file and you will find this explanation:

The contained completion routines provide support for completing:

* local and remote branch names
* local and remote tag names
* .git/remotes file names
* git 'subcommands'
* tree paths within 'ref:path/to/file' expressions
* common --long-options

And the steps to enable this

* Copy this file to somewhere (e.g. ~/.git-completion.sh).
* Add the following line `source ~/.git-completion.sh` to your .bashrc
* Changing PS1 to show current branch `PS1='[\u@\h \W$(__git_ps1 " (%s)")]\$ '`

If you want some color in git prompt

`PS1='$$\033[32m$$\u@\h$$\033[00m$$:$$\033[34m$$\w$$\033[31m$$$(__git_ps1)$$\033[00m$$\$ '`

If you want the prompt to show git-ps1 state
`GIT_PS1_SHOWDIRTYSTATE=true`

`Generating Site with Jekyllbe rake generate`

Example:

`zlu@zlu-mba:~/projects/me/octopress (master *)`