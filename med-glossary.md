---
layout: page
title: Medicine Glossary
permalink: /med-glossary/
description: "Searchable medicine and health glossary — plain-language definitions for common lab markers, conditions, and nutrition terms."
keywords: "medicine glossary, health terms, medical dictionary, lab markers, nutrition"
sitemap: true
---

<p class="glossary-intro">
  Plain-language medicine and health terms. Each entry has its own page for easy linking and search.
</p>

<div id="glossary-search">
  <label for="glossary-search-input" class="sr-only">Search terms</label>
  <input type="text" id="glossary-search-input" placeholder="Search terms..." autocomplete="off">
</div>

<p class="glossary-count">{{ site.med-glossary.size }} terms</p>

<ul id="glossary-list" class="glossary-list">
  {% assign terms = site.med-glossary | sort: "title" %}
  {% for term in terms %}
    <li class="glossary-item" id="{{ term.slug }}">
      <a class="glossary-term" href="{{ term.url }}">{{ term.title }}</a>
      <div class="glossary-definition">
        {{ term.description }}
        <a class="glossary-more" href="{{ term.url }}">Read full definition →</a>
      </div>
    </li>
  {% endfor %}
</ul>

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "DefinedTermSet",
  "name": "Medicine Glossary",
  "description": {{ page.description | jsonify }},
  "url": "https://zlu.me/med-glossary/",
  "hasDefinedTerm": [
    {% assign terms = site.med-glossary | sort: "title" %}
    {% for term in terms %}
    {
      "@type": "DefinedTerm",
      "name": {{ term.title | jsonify }},
      "url": {{ term.url | absolute_url | jsonify }},
      "description": {{ term.description | jsonify }}
    }{% unless forloop.last %},{% endunless %}
    {% endfor %}
  ]
}
</script>

<script>
document.addEventListener('DOMContentLoaded', function() {
  const input = document.getElementById('glossary-search-input');
  const list = document.getElementById('glossary-list');
  const items = Array.from(list.getElementsByTagName('li'));

  input.addEventListener('input', function() {
    const query = input.value.toLowerCase().trim();
    items.forEach(item => {
      const text = item.textContent.toLowerCase();
      item.style.display = !query || text.includes(query) ? '' : 'none';
    });
  });
});
</script>
