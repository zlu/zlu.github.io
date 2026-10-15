---
layout: default
title: "Blog Archives — Older Essays and Engineering Notes"
description: "Archive of older blog posts by Zhao Lü — engineering notes, Postgres, Rails, travel writing, and earlier experiments before the AI and tutoring focus of recent years."
permalink: /archives/
---

## Blog Archive

{% for post in site.archives %}
  [{{ post.title }}]({{ post.url }}) – {{ post.date | date: "%B %d, %Y" }}
{% endfor %}