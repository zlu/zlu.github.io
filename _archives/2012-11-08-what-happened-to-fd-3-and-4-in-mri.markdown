---
layout: post
title: "What Happened to FD 3 and 4 in MRI"
date: 2012-11-08 16:07
comments: true
categories: [ruby, system]
---

I got curious when this happened:

``` ruby
1.9.3p286 :032 > File.open('/etc/passwd').fileno
 => 5
```

I know that file descriptors (FD) 0, 1, and 2 are assigned to STDIN, STDOUT, and STDERR.  I also know that FDs are assigned
in order.  So what happened to 3 and 4?

``` ruby
1.9.3p286 :015 > STDIN.fileno
 => 0

1.9.3p286 :034 > IO.for_fd(3)
ArgumentError: The given fd is not accessible because RubyVM reserves it
	from (irb):34:in `for_fd'
	from (irb):34
	from /Users/zlu/.rvm/rubies/ruby-1.9.3-p286/bin/irb:16:in `<main>'
```

Evidently some ruby process is using it, but what process?

``` bash
zlu@zlu-mba:~$ lsof -d 3,4 | grep ruby
ruby      4980  zlu    3     PIPE 0xffffff8029fa7860     16384           ->0xffffff8029fa4370
ruby      4980  zlu    4     PIPE 0xffffff8029fa4370     16384           ->0xffffff8029fa7860

COMMAND    PID  USER   FD    TYPE DEVICE                  SIZE/OFF   NODE NAME
```

And

``` bash
zlu@zlu-mba:~$ ps -p 4980
  PID TTY           TIME CMD
 4980 ttys003    0:00.47 irb
```

So Ruby's RVM is using *pipes* with FD 3 and 4.  As far as what these pipes are used for, that'll be another discussion.