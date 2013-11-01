---
layout: post
title: "The Never Ending Journey On The Perfect Blogging Setup"
date: 2013-09-18 09:58
comments: true
categories: ['jekyll', 'github', 'plugin']

---

**_Update 1_** October 9, 2013 - Added source for using [custom plugins](#custom_plugins), which is disabled by Github pages running in safe mode.

Once in a while I search for a better blogging solution.  Like many others I've used Wordpress, Blogger, and Tumblr.  I've also used more technical approaches such as self-hosted Wordpress.  I use [Octopress](http://octopress.org) for this blog.  I have tempted to write my own blog software but some developers do.  Remy wrote his own [blog](http://www.whoisremy.com) in RubyOnRails.

I recently needed to setup a blog for [ForYogi](https://foryogi.com).  I tried Tumblr, again, for a while.  I still don't like it.  I think it's better suited for a quick photo upload which some short description.  Instapaper's blog use it for obvious reason, but it's not visually appealing to me.  Knowing what Yahoo tends to do to their purchases, I decided to abandon it.  Wordpress is a bit too overwhelming for me, with its custom domain and many packages they sell.  I do like the plethora plugins and vibrant community Wordpress have though.  Blogger is the best choice for a hosted solution providing all I need for free.  I don't believe Blogger will be part of Google's killing spree as it is being used by many bloggers for advertising.  But its editing interface feels like MS Word in 1997.  Its post display is like a typical Google product, lacking the simplicity and elegancy of what Apple has.

I played around with [Roon](http://roon.io) and liked their editing interface which supports Markdown.  They charge $12 annually for custom domain which you can get for free from many other services.  But that's fine, I'm always up for supporting start-ups.  After making the first test post, I found something I don't quite like.  There's a rather large Roon icon on the posting, if you specify an image for it.  Roon seems to take the similar approach as Medium.  Focusing on content (hence no in-article commenting) is good.  But essentially you are writing _[for free](http://www.marco.org/2013/08/05/be-your-own-platform)_ for an online magazine.  The worst part though, is in their [terms](https://roon.io/terms):
> By submitting, posting or displaying content on or through the Service, you grant Nothing Magical Inc. a worldwide, non-exclusive, royalty-free license to use, adapt, publish, and display the content in any and all media or distribution methods (now known or later developed).

To me, this is a **deal-breaker**.

[Scriptgr.am](http://scriptgr.am), which integrates with Dropbox and has more features than Roon, is better suited for me.

Through the searching process, I figured out what I wanted:

- Support of markdown
- Clean interface
- True ownership of my content
- Free custom domain
- Least amount of coding
- Ease of deployment and minimum effort of maintenance 

__My final setup is simple__

- Jekyll, a blog-aware static site generator
- Mou, a desktop markdown editor
- Github Pages

Working with [Jekyll](http://jekyllrb.com) is a pleasure.  I can use my editor of my choice to write postings and git to version control them.  Currently I'm tinkering with [Mou](mouapp.com), a markdown editor.  I like its real-time preview pane and wishing emacs' markdown mode could offer it.  Github pages is fast and free.  I created a public repo - [foryogi.github.io](https://github.com/foryogi/foryogi.github.io) and pushed to master.  Some online documentation mentions pushing to gh-pages branch, which did not work for me.  I also had to create a CNAME file by `echo "blog.foryogi.com" > CNAME` at the root of project.  This is required for GH page redirect for subdomains to work.  Github has some problem with generating Jekyll-based pages and I haven't getting deployment emails at all.  

I also added Twitter bootstrap and font-awesome.  I plan to add Google font to it next.  I also added support for Heroku, which requires Procfile, config.ru, and add a Heroku config var for the [build pack](https://github.com/mattmanning/heroku-buildpack-ruby-jekyll).  This [pull request](https://github.com/mattmanning/heroku-buildpack-ruby-jekyll/pull/9) is useful to get around a build failure due to a change in Jekyll build command.

I prefer GH page to Heroku because the initial load of Heroku site is always very slow.  GH doesn't have that problem.  I added heroku for backup in case Github goes down. It's trivial to point CNAME to heroku app.  If Amazon EC2 is down, then I'm out of luck (I think GH and Heroku both use EC2, but maybe they also use Rackspace).

**<a id="custom_plugins"></a>Custom Plugins**

Jekyll supports plugins which are custom ruby code.  Github Pages disables custom plugins due to security reason.  In order to run custom plugins, you will need to compile the source into _site and push that to Github.  Detailed instruction is [here](http://ixti.net/software/2013/01/28/using-jekyll-plugins-on-github-pages.html).
