---
layout: default
title: "Posts"
permalink: /posts/
description: "Essays and tutorials on AI, machine learning, data science, systems programming, and tutoring — by Zhao Lü."
keywords: "AI blog, machine learning tutorials, LLM fine-tuning, data science, computer science, Zhao Lü"
sitemap: true
---

## Posts

Practical notes on AI/ML, data science, and systems — plus a few older archives.

<p><a href="/ai-glossary/">AI Glossary</a> · <a href="/archives">Archives</a></p>

{% for post in site.posts %}
<div class="post-index-item" style="margin-bottom: 1rem;">
    <a href="{{ post.url }}">{{ post.title }}</a><br>
    <span class="post-date">{{ post.date | date: "%Y-%m-%d" }}</span>
    {% if post.description %}
    <p class="post-index-desc" style="margin: 0.25rem 0 0; color: #555; font-size: 0.95em;">{{ post.description }}</p>
    {% endif %}
    {% if post.tags %}
    <p class="post-index-tags" style="margin: 0.25rem 0 0; font-size: 0.85em;">
      {% for tag in post.tags %}
        {%- assign tag_slug = tag | downcase | replace: ' ', '-' -%}
        <a class="tag" href="/tags/{{ tag_slug }}/">{{ tag }}</a>{% unless forloop.last %} {% endunless %}
      {% endfor %}
    </p>
    {% endif %}
</div>
{% endfor %}
