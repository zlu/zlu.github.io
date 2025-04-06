---
layout: default
title: "Archives"
permalink: /archives/
---

## Blog Archive

{% for post in site.archives %}
  [{{ post.title }}]({{ post.url }}) – {{ post.date | date: "%B %d, %Y" }}
{% endfor %}