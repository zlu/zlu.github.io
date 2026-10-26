---
layout: default
title: "Photos"
description: "Photography by Zhao Lü — rangefinder travel and street work, organized by trip."
permalink: /photos/
---

## Photos

Rangefinder work (and more) from travel and everyday wandering. Open an album for a trip; click a photo for the full frame and description.

{% assign album_list = site.albums | where_exp: "a", "a.published != false" | sort: "date" | reverse %}
{% assign all_photos = site.photos | where_exp: "p", "p.image" | where_exp: "p", "p.published != false" | sort: "date" | reverse %}
{% assign loose = "" | split: "" %}
{% for p in all_photos %}
  {% assign alb = p.album | default: "" | strip %}
  {% if alb == blank %}
    {% assign loose = loose | push: p %}
  {% endif %}
{% endfor %}

{% if album_list.size == 0 and loose.size == 0 %}
<p class="photos-empty">No photos published yet.</p>
{% else %}

{% if album_list.size > 0 %}
<p class="photos-order-note">Albums · newest trips first</p>
<div class="photos-albums">
  {% for album in album_list %}
    {% assign album_key = album.id | default: album.slug %}
    {% assign shots = all_photos | where: "album", album_key %}
    {% assign cover = album.cover | default: "" %}
    {% if cover == blank and shots.size > 0 %}
      {% assign cover = shots.first.thumb | default: shots.first.image %}
    {% endif %}
    <a class="photos-album" href="{{ album.url }}">
      <span class="photos-album-cover">
        {% if cover != blank %}
        <img src="{{ cover }}" alt="" loading="lazy" decoding="async" width="600" height="400">
        {% endif %}
      </span>
      <span class="photos-album-info">
        <span class="photos-album-title">{{ album.title }}</span>
        <span class="photos-album-meta">
          {% if album.place %}{{ album.place }}{% endif %}{% if album.place and album.date %} · {% endif %}{% if album.date %}{{ album.date | date: "%Y" }}{% endif %}
          {% if shots.size > 0 %} · {{ shots.size }} photo{% if shots.size != 1 %}s{% endif %}{% endif %}
        </span>
      </span>
    </a>
  {% endfor %}
</div>
{% endif %}

{% if loose.size > 0 %}
### Singles
<p class="photos-order-note">Photos not in an album yet</p>
{% assign gallery = loose %}
{% assign show_place_filters = true %}
{% include photos-gallery.html %}
{% include photos-lightbox.html %}
<script src="/assets/js/photos.js" defer></script>
{% endif %}

{% endif %}

<link rel="stylesheet" href="/assets/css/photos.css">
