---
layout: post
title: "RSpec Arbitrary Handling of Arguments"
date: 2012-02-12 19:32
comments: true
categories: [rspec, tdd]
---

RSpec lets you test the number, type, and order of arguments.  For example,

{% highlight ruby %}
Foo.should_receive(:bar).with(1, kind_of(Hash), anything())
Foo.bar(1, {'a' => 'b'}, &b)
{% endhighlight %}
Pass!

This tests that class method __bar__ will be called against class __Foo__ with 3 arguments.  The first argument is integer 1,
the second argument is an instance of Hash, the third argument can be anything.

What if you want to test a bit more on the argument than that?

For example, in order to test a [private_pub](https://github.com/ryanb/private_pub) method **publish_to**:

{% highlight ruby %}
PrivatePub.should_receive(:publish_to) do |channel, data|
  channel.should eq 'messages/new'
  data[:foo].should eq 'foo'
end
PrivatePub.publish_to['messages/new', {:foo => 'foo'}
{% endhighlight %}
Pass!

This tests that **publish_to** takes 2 arguments.  The first is channel and should be equal to 'messages/new'.
The second is a hash and it equals to {:foo => 'foo'}
