---
layout: default
title: "Algorithm visualizations"
description: "Interactive algorithm visualizations by Zhao Lü — A* pathfinding, genetic algorithms, iterative local search, fractals, and more. Guides on zlu.me with live demos on AlgoScape."
keywords: "algorithm visualization, A* pathfinding, genetic algorithm, iterative local search, ILS, fractal generator, AlgoScape, interactive algorithms"
permalink: /algos/
sitemap: true
---

<h1>Algorithm visualizations</h1>

<p class="algo-intro">
  Short guides for classic algorithms, each paired with a live interactive demo on
  <a href="https://algoscape.zlu.me">AlgoScape</a>.
  New visualizations land here as they ship.
</p>

<ul class="algo-list">
  {% assign algos = site.algos | sort: "order" %}
  {% for algo in algos %}
  <li>
    <h2><a href="{{ algo.url }}">{{ algo.title }}</a></h2>
    <p class="algo-blurb">{{ algo.description }}</p>
    <p class="algo-links">
      <a href="{{ algo.url }}">Guide</a>
      {% if algo.demo %}<a href="{{ algo.demo }}">Live demo</a>{% endif %}
    </p>
  </li>
  {% endfor %}
</ul>

<p class="algo-intro">
  Also listed in <a href="/portfolio/">Portfolio</a> under AlgoScape.
</p>

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "CollectionPage",
  "name": {{ page.title | jsonify }},
  "description": {{ page.description | jsonify }},
  "url": "https://zlu.me/algos/",
  "isPartOf": {
    "@type": "WebSite",
    "name": "Zhao Lü",
    "url": "https://zlu.me"
  },
  "hasPart": [
    {% assign algos = site.algos | sort: "order" %}
    {% for algo in algos %}
    {
      "@type": "TechArticle",
      "name": {{ algo.title | jsonify }},
      "url": {{ algo.url | absolute_url | jsonify }},
      "description": {{ algo.description | jsonify }}
    }{% unless forloop.last %},{% endunless %}
    {% endfor %}
  ]
}
</script>
