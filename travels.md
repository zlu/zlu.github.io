---
layout: default
title: My Travels
permalink: /travels/
---

### Map of Some Places I Visited

{% include travel-static-map.html %}

<!-- Static map replaces interactive embeds; no JS needed here. -->


### Countries Visited

<div class="country-stats" style="margin: 8px 0 12px;">
  <span id="country-count" class="region-badge" style="display:inline-block;background:#eef2ff;color:#3730a3;border:1px solid #c7d2fe;border-radius:9999px;padding:2px 8px;font-size:0.9rem;">—</span>
  <span style="margin-left:8px;color:#6b7280;font-size:0.95rem;">Tap a chip to explore the map.</span>
  
</div>

<div class="country-chips" id="country-chips">
  <span class="chip"><span class="flag">🇦🇱</span>Albania</span>
  <span class="chip"><span class="flag">🇦🇷</span>Argentina</span>
  <span class="chip"><span class="flag">🇦🇲</span>Armenia</span>
  <span class="chip"><span class="flag">🇦🇹</span>Austria</span>
  <span class="chip"><span class="flag">🇦🇿</span>Azerbaijan</span>
  <span class="chip"><span class="flag">🇧🇪</span>Belgium</span>
  <span class="chip"><span class="flag">🇧🇦</span>Bosnia and Herzegovina</span>
  <span class="chip"><span class="flag">🇧🇷</span>Brazil</span>
  <span class="chip"><span class="flag">🇧🇬</span>Bulgaria</span>
  <span class="chip"><span class="flag">🇰🇭</span>Cambodia</span>
  <span class="chip"><span class="flag">🇨🇦</span>Canada</span>
  <span class="chip"><span class="flag">🇨🇱</span>Chile</span>
  <span class="chip"><span class="flag">🇨🇳</span>China</span>
  <span class="chip"><span class="flag">🇨🇴</span>Colombia</span>
  <span class="chip"><span class="flag">🇭🇷</span>Croatia</span>
  <span class="chip"><span class="flag">🇨🇾</span>Cyprus</span>
  <span class="chip"><span class="flag">🇨🇿</span>Czech</span>
  <span class="chip"><span class="flag">🇩🇰</span>Denmark</span>
  <span class="chip"><span class="flag">🇪🇪</span>Estonia</span>
  <span class="chip"><span class="flag">🇫🇮</span>Finland</span>
  <span class="chip"><span class="flag">🇫🇷</span>France</span>
  <span class="chip"><span class="flag">🇬🇪</span>Georgia</span>
  <span class="chip"><span class="flag">🇩🇪</span>Germany</span>
  <span class="chip"><span class="flag">🇬🇷</span>Greece</span>
  <span class="chip"><span class="flag">🇭🇺</span>Hungary</span>
  <span class="chip"><span class="flag">🇮🇪</span>Ireland</span>
  <span class="chip"><span class="flag">🇮🇱</span>Israel</span>
  <span class="chip"><span class="flag">🇮🇹</span>Italy</span>
  <span class="chip"><span class="flag">🇯🇵</span>Japan</span>
  <span class="chip"><span class="flag">🇰🇿</span>Kazakhstan</span>
  <span class="chip"><span class="flag">🇰🇷</span>Korea</span>
  <span class="chip"><span class="flag">🇱🇦</span>Laos</span>
  <span class="chip"><span class="flag">🇱🇻</span>Latvia</span>
  <span class="chip"><span class="flag">🇱🇹</span>Lithuania</span>
  <span class="chip"><span class="flag">🇱🇺</span>Luxembourg</span>
  <span class="chip"><span class="flag">🇲🇰</span>Macedonia</span>
  <span class="chip"><span class="flag">🇲🇹</span>Malta</span>
  <span class="chip"><span class="flag">🇲🇽</span>Mexico</span>
  <span class="chip"><span class="flag">🇲🇩</span>Moldova</span>
  <span class="chip"><span class="flag">🇲🇨</span>Monaco</span>
  <span class="chip"><span class="flag">🇲🇪</span>Montenegro</span>
  <span class="chip"><span class="flag">🇲🇦</span>Morocco</span>
  <span class="chip"><span class="flag">🇳🇱</span>Netherlands</span>
  <span class="chip"><span class="flag">🇵🇦</span>Panama</span>
  <span class="chip"><span class="flag">🇵🇪</span>Peru</span>
  <span class="chip"><span class="flag">🇵🇱</span>Poland</span>
  <span class="chip"><span class="flag">🇵🇹</span>Portugal</span>
  <span class="chip"><span class="flag">🇷🇴</span>Romania</span>
  <span class="chip"><span class="flag">🇷🇸</span>Serbia</span>
  <span class="chip"><span class="flag">🇸🇰</span>Slovakia</span>
  <span class="chip"><span class="flag">🇪🇸</span>Spain</span>
  <span class="chip"><span class="flag">🇸🇪</span>Sweden</span>
  <span class="chip"><span class="flag">🇨🇭</span>Switzerland</span>
  <span class="chip"><span class="flag">🇹🇭</span>Thailand</span>
  <span class="chip"><span class="flag">🇹🇷</span>Turkey</span>
  <span class="chip"><span class="flag">🇬🇧</span>UK</span>
  <span class="chip"><span class="flag">🇺🇸</span>USA</span>
  <span class="chip"><span class="flag">🇺🇿</span>Uzbekistan</span>
  <span class="chip"><span class="flag">🇻🇦</span>Vatican City</span>
  <span class="chip"><span class="flag">🇻🇳</span>Viet Nam</span>
</div>

<style>
  .country-chips {
    display: flex;
    flex-wrap: wrap;
    gap: .5rem;
    align-items: center;
  }
  .chip {
    display: inline-block;
    padding: .25rem .6rem;
    border: 1px solid #e5e7eb;
    border-radius: 9999px;
    background: #fff;
    color: #374151;
    font-size: .95rem;
    white-space: nowrap;
    transition: background .2s ease, color .2s ease, border-color .2s ease;
    cursor: default;
  }
  .chip:hover {
    background: #f9fafb;
    border-color: #d1d5db;
  }
  .chip .flag { margin-right: .35rem; }
  @media (max-width: 480px) {
    .chip { font-size: .9rem; }
  }
</style>

<script>
  document.addEventListener('DOMContentLoaded', function(){
    const chips = document.querySelectorAll('#country-chips .chip');
    const countEl = document.getElementById('country-count');
    if (countEl) countEl.textContent = chips.length + ' countries';
  });
</script>
