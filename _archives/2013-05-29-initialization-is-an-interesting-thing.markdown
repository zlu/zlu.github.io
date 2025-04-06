---
layout: post
title: "Initialization Is An Interesting Thing"
date: 2013-05-29 22:51
comments: true
categories: [Rails, Ruby]
---

Have you ever tried to define `initialized` method for an `ActiveRecord`?

Try it if you haven't.

```
class MyRecord < ActiveRecord::Base
  def initialize
  end
end

MyRecord.new  #<MyRecord not initialized>

```

This is probably not what you have expected.  Why is MyRecord not initialized?

Well, ActiveRecord::Base has already defined `initialize` method.  It is used to instantiate Rails models.

```
def initialize(attributes = nil, options = {})
```

If you really want to override it with your own version, you would want to do this:

```
class MyRecord < ActiveRecord::Base
  def initialize(attributes = nil, options = {})
    super
    # now do your own initialization
  end
end

MyRecord.new
```
This should build a new AR for you (if you have a defined my_records table in database).

Rails provides a cleaner way to handle this via after_initilize callback.

```
class MyClass < ActiveRecord::Base
  after_initialize do
    # do your thing here
    puts 'more to do after initialization'
  end
end
```

`after_initialize` is called when ActiveRecord is instantiated and before it is saved.


