---
layout: default
title: "Posts"
permalink: /posts/
---

## Posts

[Archives](/archives) contain out-of-date posts just for memories.

{% for post in site.posts %}

<div style="margin-bottom: 10px;">
    <a href="{{ post.url }}">{{ post.title }}</a><br>
    <span>{{ post.date | date: "%m-%d-%Y" }}</span>
</div>
{% endfor %}
