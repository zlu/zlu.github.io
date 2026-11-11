---
layout: page
title: "CS Glossary — Algorithms, Complexity, and Systems Terms"
permalink: /cs-glossary/
description: "Searchable computer science glossary covering complexity classes, algorithms, and systems terms, each with a short definition page you can link from posts and notes."
keywords: "CS glossary, computer science glossary, NP-hard, complexity theory, algorithms dictionary"
sitemap: true
---

<p class="glossary-intro">
  Concise definitions for algorithms, complexity, automata, and systems concepts.
  Each term has its own indexable page — browse below or search.
</p>

<div id="glossary-search">
  <label for="glossary-search-input" class="sr-only">Search terms</label>
  <input type="text" id="glossary-search-input" placeholder="Search terms..." autocomplete="off">
</div>

<p class="glossary-count">{{ site.cs-glossary.size }} terms</p>

<ul id="glossary-list" class="glossary-list">
  {% assign terms = site.cs-glossary | sort: "title" %}
  {% for term in terms %}
    <li class="glossary-item" id="{{ term.slug }}">
      <a class="glossary-term" href="{{ term.url }}">{{ term.title }}</a>
      {% if term.synonyms and term.synonyms.size > 0 %}
        <span class="glossary-aka">({% for s in term.synonyms %}{{ s }}{% unless forloop.last %}, {% endunless %}{% endfor %})</span>
      {% endif %}
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
  "name": "CS Glossary",
  "description": {{ page.description | jsonify }},
  "url": "https://zlu.me/cs-glossary/",
  "hasDefinedTerm": [
    {% assign terms = site.cs-glossary | sort: "title" %}
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
