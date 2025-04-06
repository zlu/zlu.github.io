---
layout: post
title: "Install and Create Rails 4.0 App"
date: 2012-10-29 22:46
comments: true
categories: [rails]
---

Rails 4.0 development has started a while back.  It was merged into Rails' master branch on Github in 2011.
As the writing of this blog entry, the stable version of Rails is 3.2.8.  Release candidate is at 3.2.9.rc1.
Because 4.0 is not yet official available, you have to jump through hoops to get a Rails 4 application up and running.

Install Rails 4.0 gem
----

Create this Gemfile:

```
source 'http://rubygems.org'

gem 'rails', :git => 'git://github.com/rails/rails.git'
gem 'uglifier', '>= 1.0.30'
gem 'activerecord-deprecated_finders', :git => 'git://github.com/rails/activerecord-deprecated_finders.git'
gem 'journey', :git => 'git://github.com/rails/journey.git'
```

* This Gemfile specifies the master branch of Rails, which contains 4.0 beta.
* Gem activerecord-deprecated_finders is a dependency.
* Gem journey, the Rails router, is also required.

In the same directory of the Gemfile, issue `bundle install`.  This should install Rails 4.0.0.beta.
But running `gem list rails` won't show 4.0 being installed.  So do this instead:

```
zlu@zlu-mba:~/projects/test$ bundle show rails
/Users/zlu/.rvm/gems/ruby-1.9.3-p0/bundler/gems/rails-4e23c0ef341c
```

The Rails executable is at:
```
`bundle show rails`/railties/bin/rails
```

```
`bundle show rails`/railties/bin/rails -v should show: Rails 4.0.0.beta
```

Now you can use this version of rails to generate an app named foo.


Create Rails 4.0 app
----

```
`bundle show rails`/railties/bin/rails new foo
```
This command  creates a typical rails app called foo and output an error about not being able to find Rails 4 gem, which
should not be a surprise.

You will edit foo/Gemfile to make `bundle install` work.

Add the two gems Rails depends on:

```
gem 'journey', github: 'rails/journey'
gem 'activerecord-deprecated_finders', github: 'rails/activerecord-deprecated_finders'
```

Change the source definition for the following gems:

```
gem 'rails', '4.0.0.beta'
gem 'sass-rails',   '~> 4.0.0.beta'
gem 'coffee-rails', '~> 4.0.0.beta'

```

To

```
gem 'rails', github: 'rails/rails'
gem 'sass-rails', github: 'rails/sass-rails'
gem 'coffee-rails', github: 'rails/coffee-rails'
```

Now you should be `bundle install`, sit back and enjoy your new Rails 4.0 app.

