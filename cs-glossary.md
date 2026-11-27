---
layout: page
title: "CS Glossary — Algorithms, Statistics, Probability, and Systems Terms"
permalink: /cs-glossary/
description: "Searchable computer science glossary covering algorithms, data structures, complexity, plus statistics and probability — moments, estimators, MLE, distributions, Bayes, OLS, RSE, R², F-tests, hypothesis tests — each on its own SEO-indexable page."
keywords: "CS glossary, computer science glossary, statistics glossary, probability glossary, estimator, method of moments, maximum likelihood, MLE, Bayes theorem, normal distribution, central limit theorem, OLS, linear regression, residual standard error, RSE, R-squared, p-value, Type I error, power, ACF, PACF, ARIMA, Moran's I, linear programming, assignment problem, hypothesis testing, confidence interval, sorting algorithms, binary search, BFS, DFS, Dijkstra, dynamic programming, NP-hard, complexity theory, TLS, JA4"
sitemap: true
---

<p class="glossary-intro">
  Concise definitions for sorting, searching, graph algorithms, data structures,
  complexity, automata, networking / TLS, systems concepts, and
  <strong>statistics &amp; probability</strong>
  (moments, estimators, distributions, Bayes, OLS, RSE, R², F-tests, ACF/PACF, ARIMA, spatial models, LP/IP, inference).
  Each term has its own indexable page. Browse below or search.
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
