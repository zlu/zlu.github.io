---
layout: page
title: "Medicine Glossary — Lab Markers, Conditions, Eye & Nutrition"
permalink: /med-glossary/
description: "Searchable medicine and health glossary with plain-language definitions for lab markers, conditions, eye terms, and nutrition, each on its own linkable page."
keywords: "medicine glossary, health terms, medical dictionary, lab markers, eye terms, glaucoma, dry eye, nutrition"
sitemap: true
---

<p class="glossary-intro">
  Short bilingual (中文 / English) medicine and health terms — lab markers, conditions, eye care, nutrition.
  Browse or search; each entry is a concise definition page for linking.
</p>

<div id="glossary-search">
  <label for="glossary-search-input" class="sr-only">Search terms</label>
  <input type="text" id="glossary-search-input" placeholder="Search 中文 or English..." autocomplete="off">
</div>

<p class="glossary-count">{{ site.med-glossary.size }} terms</p>

<ul id="glossary-list" class="glossary-list">
  {% assign terms = site.med-glossary | sort: "title_short" %}
  {% for term in terms %}
    <li class="glossary-item" id="{{ term.slug }}">
      <a class="glossary-term" href="{{ term.url }}">{{ term.title_short | default: term.title }}</a>
      {% if term.term-en and term.term-en.size > 0 %}
        <span class="glossary-aka">({{ term.term-en | join: ", " }})</span>
      {% endif %}
      <div class="glossary-definition">
        {{ term.description }}
        <a class="glossary-more" href="{{ term.url }}">词条 →</a>
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
