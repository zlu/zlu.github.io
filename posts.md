---
layout: default
title: "Posts"
permalink: /posts/
---

# Posts

{% for post in site.posts %}
<div style="margin-bottom: 10px;">
    <a href="{{ post.url }}">{{ post.title }}</a><br>
    <span>{{ post.date | date: "%m-%d-%Y" }}</span>
</div>
{% endfor %}