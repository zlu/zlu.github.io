---
layout: post
title: "Quick Survey Of Rails CMS"
date: 2013-09-27 21:37
comments: true
categories: [rails, cms]
---

**Update 1** (September 29, 2013) - Added references and more CMS.  
**Update 2** (September 30, 2013) - ComfortableMexicanSofa does _not_ use Liquid Template, as GBH has pointed out in the comment.

Selecting a Content Management System (CMS) is never an easy task.  I have recently spent a good amount of time tinkering with different implementations and would like to share my learnings here.

There are many solutions on the market so you should solidify your requirements to help you narrow down the choices.  This CMS will be used for [ForYogi](https://foryogi.com) with the following requirements:

- Admin can add tutorials on how to use the site for teachers/studios/users.  
- Admin can add articles for yoga-related topics into a knowledge base.  Topics can be as simple as definition of  _yogi_ or Sanskrit terminologies such as _asana_.  
- Admin can add help content which can be accessed by the application and displayed in various parts of the app.
- Admin can add FAQs. 
- Users can search contents in Knowledge Base.
- Admin can customize layout and css style.
- Drop-in support for existing Rails4 (Ruby 2) project.
 - Inexpensive - Some hosted solution can get pricey depending on storage and other factors.
 
####General Impression
 
 CMS is often used by developers to create customizable mostly static websites for clients.  Most of the Rails CMS gems have over the years, [enginfied](http://guides.rubyonrails.org/engines.html‎), providing drop-in support for existing Rails apps.
 
**[RefineryCMS](http://refinerycms.com)**

There are [free](http://railscasts.com/episodes/332-refinery-cms-basics) and [pro](http://railscasts.com/episodes/333-extending-refinery-cms) episodes of it on RailsCasts.com.  I don't have a pro-account so I only examined the free content.  My impression is it's easy to set it ground-up with customized style.  This CMS has been around for a long time and has a good community build around it.  There are two factors against me choosing it though:

- At the time of writing, its Rails4 support is [work in progress](https://github.com/refinery/refinerycms/commit/d9e7d4dfda3256ece0b527da269a1f2643a9afc2).  
- Integrating it with an existing application using devise is quite involved according to this [article](http://refinerycms.com/guides/with-an-existing-rails-31-devise-app).

**[LocomotivesCMS](http://locomotivecms.com)**

This one has a nicely styled site (compare to some other CMS) but has a few things that concerns me:

- It requires MongoDB.  It is not a big deal. I'm not using Mongo in my current setup.  I can get a small setup using Heroku add-on but I'd probably have to pay to get a storage bump-up in the near future.
- Weak [documentation](http://doc.locomotivecms.com).  I looked through the content and found documentation is quite superficial and lacking depth.  It is understandable as they offer a hosting solution.
- The [issue](https://github.com/locomotivecms/engine/issues/746) on Rails4 support concerns me the most:	
 	>Probably by the end of the year. However, It could happen sooner depending on how successful would be our hosting solution…Long story short, we need financial support to handle the next big version of the engine which will include a new UI.
 	
As there is no guarantee for a rather future release of Rails4 support (at least not in their current roadmap), I pass. 	
**[AlchemyCMS](http://alchemy-cms.com)**

I first learned about this CMS through the comments on the free RailsCasts episode.  They argued about why they are better than refineryCMS.  I liked their idea of storing pure content, not css style and layouts in database.  It'll only make migration to other systems easier.  I also noticed that they have a [3.0-dev branch](https://github.com/magiclabs/alchemy_cms/tree/3.0-dev) supporting Rails4 with this [sha](fe94bedc761484940071129277970a6cd65fba10).  Installation has proven to be hard as there are a few conflict in various gems (if you use activeadmin and devise).  I was able to install it after some struggle.  However, I encountered problems running it (devise login, actions_cache).  It was just wasting more time than I wanted at this point.

**[BrowserCMS]()**

The first problem I had is jquery-rails version being too low even for their [Rails 4 master branch](https://github.com/browsermedia/browsercms) for which [peakpg](https://github.com/peakpg) opened [an issue](https://github.com/browsermedia/browsercms/issues/625).  After fixing that in my own fork, I continue running into other problems.  Basically the Rails4 support is not quite ready yet.

**[ComfortableMexicanSofa]()**

This cleverly named gem is lesser popular than the other gems I've looked at based on the number of watches and forks on github.  However, it is the only one that worked out-of-box with an existing Rails4 app with devise authentication.  The documentation explains the theory well but lacks examples.  I suggest watching the RailsCasts [free episode](http://railscasts.com/episodes/118‎) on [Liquid template](http://liquidmarkup.org/‎) and really understand [tags](https://github.com/comfy/comfortable-mexican-sofa/wiki/Tags) and how to apply partials/helpers.  

Here is how to create a simple navigation for all the pages.  According to the [guide](https://github.com/comfy/comfortable-mexican-sofa/wiki/Creating-navigation-from-pages), you need to add something like this:

```ruby
<% @cms_site.pages.root.children.published.each do |page| %>
  <%= link_to(page.label, page.url) %><br>
<% end %>
```

But where do you add it?  I tried to create a new partial _kb.html.erb under app/views/layouts and got a no partial found error.  It is looking for cms_content and application directories for partials.  So I created app/views/cms_content directory and moved the partial there.  This kind/level of details should, ideally, be in the guide.
You can then invoke this partial inside of a layout: 

{% raw %}
    {{ cms:partial:kb }}
{% endraw %}

Once you get how Liquid template markup works with comfy sofa, it becomes intuitive to create sites/layouts/pages.

**[CMS_Admin](https://github.com/websitescenes/cms_admin)**

It is based on active_admin and uses devise for authentication.  It is Rails4 only but it is an app and not meant to be used with any existing Rails application.

###Conclusion

I selected ComfortableMexicanSofa for its ease in integration into an existing Rails4 application with devise authentication.  The customization seems to be quite flexible.  It is also possible to invoke it from the main app which enables me to display help content in various places inside of the app.  The only requirement I haven't looked at is search.

**References**

[A list of Rails CMS](https://gist.github.com/ffmike/242751)