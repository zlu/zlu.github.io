---
layout: default
title: "Photos"
description: "Photography by Zhao Lü — rangefinder travel and street work. Web-sized images; originals archived privately."
permalink: /photos/
---

## Photos

Rangefinder work (and more) from travel and everyday wandering. Web versions live here; full-resolution masters may available upon request.

{% assign gallery = site.photos | where_exp: "p", "p.image" | where_exp: "p", "p.published != false" | sort: "date" | reverse %}

{% if gallery.size == 0 %}
<p class="photos-empty">No photos published yet.</p>
{% else %}
<div class="photos-filters" id="photos-filters" hidden>
  <button type="button" class="photos-filter is-active" data-place="">All</button>
</div>

<div class="photos-grid" id="photos-grid">
  {% for photo in gallery %}
  <figure class="photos-item"
          data-place="{{ photo.place | default: '' | escape }}"
          data-year="{{ photo.date | date: '%Y' }}">
    <button type="button"
            class="photos-thumb"
            data-full="{{ photo.image }}"
            data-title="{{ photo.title | escape }}"
            data-meta="{% if photo.place %}{{ photo.place | escape }}{% endif %}{% if photo.place and photo.date %} · {% endif %}{% if photo.date %}{{ photo.date | date: '%Y' }}{% endif %}{% if photo.camera %} · {{ photo.camera | escape }}{% endif %}"
            aria-label="View {{ photo.title | escape }}">
      <img src="{{ photo.thumb | default: photo.image }}"
           alt="{{ photo.title | escape }}"
           loading="lazy"
           decoding="async"
           width="600"
           height="400">
    </button>
    <figcaption>
      <span class="photos-caption-title">{{ photo.title }}</span>
      {% if photo.place or photo.date %}
      <span class="photos-caption-meta">
        {% if photo.place %}{{ photo.place }}{% endif %}{% if photo.place and photo.date %} · {% endif %}{% if photo.date %}{{ photo.date | date: "%Y" }}{% endif %}
      </span>
      {% endif %}
    </figcaption>
  </figure>
  {% endfor %}
</div>
{% endif %}

<dialog class="photos-lightbox" id="photos-lightbox" aria-label="Photo viewer">
  <form method="dialog" class="photos-lightbox-close-wrap">
    <button type="submit" class="photos-lightbox-close" aria-label="Close">&times;</button>
  </form>
  <img src="" alt="" id="photos-lightbox-img">
  <div class="photos-lightbox-caption">
    <div id="photos-lightbox-title"></div>
    <div id="photos-lightbox-meta"></div>
  </div>
</dialog>

<link rel="stylesheet" href="/assets/css/photos.css">
<script src="/assets/js/photos.js" defer></script>
