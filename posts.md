---
layout: default
title: "Blog Posts on AI, Machine Learning, and Systems"
permalink: /posts/
description: "Essays and tutorials on AI, machine learning, data science, systems programming, and tutoring — by Zhao Lü."
keywords: "AI blog, machine learning tutorials, LLM fine-tuning, data science, computer science, Zhao Lü"
sitemap: true
---

{% assign en_posts = site.posts | where_exp: "post", "post.lang != 'zh'" %}
{% assign zh_posts = site.posts | where_exp: "post", "post.lang == 'zh'" %}

<div class="posts-heading-row">
  <h1>Posts</h1>
  <div class="posts-lang-tabs" role="tablist" aria-label="Language">
    <button type="button" class="posts-lang-tab is-active" role="tab" aria-selected="true" data-lang="en" id="tab-en">
      EN <span class="posts-lang-count">{{ en_posts.size }}</span>
    </button>
    <button type="button" class="posts-lang-tab" role="tab" aria-selected="false" data-lang="zh" id="tab-zh">
      中文 <span class="posts-lang-count">{{ zh_posts.size }}</span>
    </button>
  </div>
</div>

Practical notes on AI/ML, data science, and systems.

<div class="posts-index" id="posts-index" data-per-page="8">
  <div class="posts-toolbar">
    <div class="posts-search">
      <label for="posts-search-input" class="sr-only">Search posts</label>
      <span class="posts-search-icon" aria-hidden="true">
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
          <circle cx="6.5" cy="6.5" r="4.5" stroke="currentColor" stroke-width="1.5"/>
          <path d="M10 10l3.5 3.5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
        </svg>
      </span>
      <input
        type="search"
        id="posts-search-input"
        placeholder="Search titles, tags…"
        autocomplete="off"
        spellcheck="false"
        enterkeyhint="search">
      <button type="button" class="posts-search-clear" id="posts-search-clear" aria-label="Clear search" hidden>×</button>
    </div>
    <span class="posts-search-status" id="posts-search-status" aria-live="polite" hidden></span>
  </div>

  <div class="posts-list" id="posts-list">
    {% for post in site.posts %}
    {% assign post_lang = post.lang | default: 'en' %}
    {% assign post_tags = post.tags | join: ' ' %}
    <article
      class="post-index-item"
      data-lang="{{ post_lang }}"
      data-title="{{ post.title | escape }}"
      data-tags="{{ post_tags | escape }}"
      data-desc="{{ post.description | default: '' | escape }}"
      hidden>
      <h3 class="post-index-title"><a href="{{ post.url }}">{{ post.title }}</a></h3>
      <div class="post-index-meta">
        <time datetime="{{ post.date | date: '%Y-%m-%d' }}">{{ post.date | date: "%Y-%m-%d" }}</time>
      </div>
      {% if post.description %}
      <p class="post-index-desc">{{ post.description }}</p>
      {% endif %}
      {% if post.tags and post.tags.size > 0 %}
      <ul class="post-index-tags" aria-label="Tags">
        {% for tag in post.tags %}
          {%- assign tag_slug = tag | downcase | replace: ' ', '-' -%}
          <li><a class="tag" href="/tags/{{ tag_slug }}/">{{ tag }}</a></li>
        {% endfor %}
      </ul>
      {% endif %}
    </article>
    {% endfor %}
  </div>

  <p class="posts-empty" id="posts-empty" hidden>No posts match that search.</p>

  <nav class="posts-pagination" id="posts-pagination" aria-label="Posts pagination" hidden>
    <button type="button" class="posts-page-btn" data-action="prev" aria-label="Previous page">← Prev</button>
    <span class="posts-page-status" id="posts-page-status" aria-live="polite"></span>
    <button type="button" class="posts-page-btn" data-action="next" aria-label="Next page">Next →</button>
  </nav>
</div>

<noscript>
  <style>
    #posts-list .post-index-item[hidden] { display: block !important; }
    #posts-pagination { display: none !important; }
  </style>
</noscript>

<script src="/assets/js/posts-index.js" defer></script>
