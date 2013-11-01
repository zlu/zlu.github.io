---
layout: post
title: "System Calls In Ruby"
date: 2013-06-03 00:42
comments: true
categories: [Ruby, System call, Security]
---

There are a few ways to execute system commands in Ruby: backticks, system, exec, %x[], and Open3#popen3.
We will take a look at the their differences and briefly discuss security concerns of applying them.

## Backticks (Kernel#`)

When using backticks, the result is returned as string.  Process status of method execution is stored in $?

{% highlight ruby %}
1.9.3p429 :006 > `ls`
 => "CHANGELOG.markdown\nGemfile\nGemfile.lock\nREADME.markdown\nRakefile\n_config.yml\nconfig.rb\nconfig.ru\nplugins\npublic\nsass\nsass.old\nsource\nsource.old\n"
1.9.3p429 :007 > $?
 => #<Process::Status: pid 35977 exit 0>
1.9.3p429 :008 > s = _
 => #<Process::Status: pid 35977 exit 0>
1.9.3p429 :009 > s.success?
 => true
 {% endhighlight %}

## %x

%x() or %x[] is similar to using backticks.

## Kernel#system

When using system method, the result is true or false depending on whether the command is executed successfully.

{% highlight ruby %}
1.9.3p429 :002 > system 'ls'
CHANGELOG.markdown README.markdown    config.rb          public             source
Gemfile            Rakefile           config.ru          sass               source.old
Gemfile.lock       _config.yml        plugins            sass.old
 => true
1.9.3p429 :003 > system 'ls a'
ls: a: No such file or directory
 => false
{% endhighlight %}

## Kernel#exec

exec replaces the current process by running the given external command.  If you invoke exec in irb, then the irb process
will be replaced by the running external command.  You will get the shell prompt back after exec finishes.  If you invoke
exec in a ruby program, that program will stop execution (just like the irb process).

{% highlight ruby %}
zlu@zlu-mba:~/projects/me/blog-zlu (master *)$ irb
1.9.3p429 :001 > exec 'ls'
CHANGELOG.markdown README.markdown    config.rb          public             source
Gemfile            Rakefile           config.ru          sass               source.old
Gemfile.lock       _config.yml        plugins            sass.old
zlu@zlu-mba:~/projects/me/blog-zlu (master *)$
{% endhighlight %}

## Open3#popen3

popen3 executes the command while opening stdin, stdout, and stderr and a thread to wait for the command execution.

**stdout with successfully execution**
{% highlight ruby %}
1.9.3p429 :010 > require 'open3'
 => true
1.9.3p429 :011 > Open3.popen3 'ls'
 => [#<IO:fd 6>, #<IO:fd 7>, #<IO:fd 9>, #<Thread:0x007fcd928ece18 run>]
1.9.3p429 :012 > i, o, e, t = _
 => [#<IO:fd 6>, #<IO:fd 7>, #<IO:fd 9>, #<Thread:0x007fcd928ece18 dead>]
1.9.3p429 :014 > o.read
 => "CHANGELOG.markdown\nGemfile\nGemfile.lock\nREADME.markdown\nRakefile\n_config.yml\nconfig.rb\nconfig.ru\nplugins\npublic\nsass\nsass.old\nsource\nsource.old\n"
{% endhighlight %}

**stderr with unsuccessfully execution**
{% highlight ruby %}
1.9.3p429 :027 > Open3.popen3('ls a') do |i, o, e, t|
1.9.3p429 :029?>   p e.read
1.9.3p429 :030?>   end
"ls: a: No such file or directory\n"
 => "ls: a: No such file or directory\n"
{% endhighlight %}

There are a few variations of popen3 methods such as popen2 where a couple of streams are merged 2&>1, for example.

## Security Concerns

Much like SQL Injection, similar things can happen when using these method calls.
Consider a command like `ls a;rm a`.  or even worse `ls a;rm -rf *`
We can address such concerns with another form of popen3(cmd, args).  For the command above, it is
`popen3('ls', 'a;rm -rf*')`.  The last part of the command is interpreted as the options for `ls`.

Take a look at this playful little app [web_shell](https://github.com/zlu/web_shell).