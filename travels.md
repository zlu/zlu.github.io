---
layout: default
title: My Travels
permalink: /travels/
---

### Map of Some Places I Visited

<div class="map-container">
  <div class="map-tabs">
    <button class="map-tab active" onclick="switchMap('google')">Google Maps</button>
    <button class="map-tab" onclick="switchMap('amap')">AMap (高德地图)</button>
  </div>
  
  <div class="map-content">
    <div id="google-map" class="map-frame active">
      <iframe src="https://www.google.com/maps/d/embed?mid=1Ki6wGR5Omnjm2EQ1DdwmGktgpPY-R4E&ehbc=2E312F" width="100%" height="480" frameborder="0" style="border:0;" allowfullscreen></iframe>
    </div>
    
    <div id="amap-map" class="map-frame">
      <iframe id="amap-iframe" src="https://guinness.autonavi.com/activity/2020CommonLanding/index.html?id=default&local=1&logId=&logParams=&gd_from=jinisi&schema=amapuri%3A%2F%2Fajx_favorites%2Ffolder%3Fdata%3D%257B%2522ugcId%2522%253A%252207805534120186948934%2522%252C%2522forceCustom%2522%253Atrue%252C%2522pathId%2522%253A6%252C%2522isCreatorShare%2522%253Atrue%257D&share_from=favorites_FavoriteFolder&share_from_type=AJX&share_type=image&share_lastClickSpm=" width="100%" height="480" frameborder="0" style="border:0;" allowfullscreen></iframe>
    </div>
  </div>
  
  <div class="map-instructions">
    <p class="map-note">
      <span id="google-instructions" class="map-instruction active">
        You're viewing the Google Maps version. This map is maintained using Google My Maps.
      </span>
      <span id="amap-instructions" class="map-instruction">
        您正在查看高德地图版本。如果您在中国，此地图可能加载更快。
        <br>
        <a href="https://www.amap.com/" target="_blank">点击这里访问高德地图网站</a>创建您自己的地图收藏。
      </span>
    </p>
  </div>
</div>

<style>
  .map-container {
    width: 100%;
    max-width: 100%;
    margin: 20px 0;
    border: 1px solid #eaeaea;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
  }
  
  .map-tabs {
    display: flex;
    background-color: #f5f5f5;
    border-bottom: 1px solid #eaeaea;
  }
  
  .map-tab {
    flex: 1;
    padding: 12px;
    background: none;
    border: none;
    cursor: pointer;
    font-size: 14px;
    font-weight: 500;
    color: #666;
    transition: all 0.3s ease;
  }
  
  .map-tab.active {
    background-color: #fff;
    color: #3b82f6;
    border-bottom: 2px solid #3b82f6;
  }
  
  .map-tab:hover:not(.active) {
    background-color: #e9e9e9;
  }
  
  .map-content {
    position: relative;
    width: 100%;
    height: 480px;
  }
  
  .map-frame {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    display: none;
  }
  
  .map-frame.active {
    display: block;
  }
  
  .map-instructions {
    padding: 10px 15px;
    background-color: #f9f9f9;
    border-top: 1px solid #eaeaea;
  }
  
  .map-note {
    margin: 0;
    font-size: 13px;
    color: #666;
  }
  
  .map-instruction {
    display: none;
  }
  
  .map-instruction.active {
    display: block;
  }
  
  @media (max-width: 768px) {
    .map-content {
      height: 400px;
    }
  }
  
  @media (max-width: 480px) {
    .map-content {
      height: 300px;
    }
    
    .map-tab {
      padding: 10px 5px;
      font-size: 13px;
    }
  }
</style>

<script>
  function switchMap(mapType) {
    // Update tabs
    document.querySelectorAll('.map-tab').forEach(function(tab) {
      tab.classList.remove('active');
    });
    document.querySelector(`.map-tab[onclick="switchMap('${mapType}')"]`).classList.add('active');
    
    // Update map frames
    document.querySelectorAll('.map-frame').forEach(function(frame) {
      frame.classList.remove('active');
    });
    document.getElementById(`${mapType}-map`).classList.add('active');
    
    // Update instructions
    document.querySelectorAll('.map-instruction').forEach(function(instruction) {
      instruction.classList.remove('active');
    });
    document.getElementById(`${mapType}-instructions`).classList.add('active');
    
    // Save preference
    if (window.localStorage) {
      localStorage.setItem('preferredMapType', mapType);
    }
    
    // Auto-detect language and set appropriate map
    if (!localStorage.getItem('mapManuallySelected')) {
      const language = navigator.language || navigator.userLanguage;
      if (language && language.toLowerCase().startsWith('zh')) {
        switchMap('amap');
      }
      localStorage.setItem('mapManuallySelected', 'true');
    }
  }
  
  // Initialize on page load
  document.addEventListener('DOMContentLoaded', function() {
    // Check for saved preference
    if (window.localStorage && localStorage.getItem('preferredMapType')) {
      switchMap(localStorage.getItem('preferredMapType'));
    } else {
      // Auto-detect language and set appropriate map
      const language = navigator.language || navigator.userLanguage;
      if (language && language.toLowerCase().startsWith('zh')) {
        switchMap('amap');
      }
    }
  });
</script>


### Countries Visited

<div class="country-stats" style="margin: 8px 0 12px;">
  <span id="country-count" class="region-badge" style="display:inline-block;background:#eef2ff;color:#3730a3;border:1px solid #c7d2fe;border-radius:9999px;padding:2px 8px;font-size:0.9rem;">—</span>
  <span style="margin-left:8px;color:#6b7280;font-size:0.95rem;">Tap a chip to explore the map.</span>
  
</div>

<div class="country-chips" id="country-chips">
  <span class="chip"><span class="flag">🇺🇸</span>USA</span>
  <span class="chip"><span class="flag">🇨🇦</span>Canada</span>
  <span class="chip"><span class="flag">🇩🇪</span>Germany</span>
  <span class="chip"><span class="flag">🇳🇱</span>Netherlands</span>
  <span class="chip"><span class="flag">🇪🇸</span>Spain</span>
  <span class="chip"><span class="flag">🇬🇧</span>UK</span>
  <span class="chip"><span class="flag">🇨🇭</span>Switzerland</span>
  <span class="chip"><span class="flag">🇫🇷</span>France</span>
  <span class="chip"><span class="flag">🇧🇪</span>Belgium</span>
  <span class="chip"><span class="flag">🇩🇰</span>Denmark</span>
  <span class="chip"><span class="flag">🇵🇹</span>Portugal</span>
  <span class="chip"><span class="flag">🇮🇹</span>Italy</span>
  <span class="chip"><span class="flag">🇻🇦</span>Vatican City</span>
  <span class="chip"><span class="flag">🇨🇿</span>Czech</span>
  <span class="chip"><span class="flag">🇦🇹</span>Austria</span>
  <span class="chip"><span class="flag">🇭🇺</span>Hungary</span>
  <span class="chip"><span class="flag">🇮🇱</span>Israel</span>
  <span class="chip"><span class="flag">🇨🇳</span>China</span>
  <span class="chip"><span class="flag">🇵🇦</span>Panama</span>
  <span class="chip"><span class="flag">🇲🇽</span>Mexico</span>
  <span class="chip"><span class="flag">🇹🇭</span>Thailand</span>
  <span class="chip"><span class="flag">🇰🇭</span>Cambodia</span>
  <span class="chip"><span class="flag">🇻🇳</span>Viet Nam</span>
  <span class="chip"><span class="flag">🇰🇷</span>Korea</span>
  <span class="chip"><span class="flag">🇲🇨</span>Monaco</span>
  <span class="chip"><span class="flag">🇯🇵</span>Japan</span>
  <span class="chip"><span class="flag">🇧🇷</span>Brazil</span>
  <span class="chip"><span class="flag">🇵🇪</span>Peru</span>
  <span class="chip"><span class="flag">🇨🇴</span>Colombia</span>
  <span class="chip"><span class="flag">🇦🇷</span>Argentina</span>
  <span class="chip"><span class="flag">🇨🇱</span>Chile</span>
  <span class="chip"><span class="flag">🇸🇪</span>Sweden</span>
  <span class="chip"><span class="flag">🇫🇮</span>Finland</span>
  <span class="chip"><span class="flag">🇪🇪</span>Estonia</span>
  <span class="chip"><span class="flag">🇱🇻</span>Latvia</span>
  <span class="chip"><span class="flag">🇱🇹</span>Lithuania</span>
  <span class="chip"><span class="flag">🇵🇱</span>Poland</span>
  <span class="chip"><span class="flag">🇸🇰</span>Slovakia</span>
  <span class="chip"><span class="flag">🇷🇸</span>Serbia</span>
  <span class="chip"><span class="flag">🇧🇬</span>Bulgaria</span>
  <span class="chip"><span class="flag">🇬🇷</span>Greece</span>
  <span class="chip"><span class="flag">🇲🇦</span>Morocco</span>
  <span class="chip"><span class="flag">🇷🇴</span>Romania</span>
  <span class="chip"><span class="flag">🇮🇪</span>Ireland</span>
  <span class="chip"><span class="flag">🇦🇱</span>Albania</span>
  <span class="chip"><span class="flag">🇲🇪</span>Montenegro</span>
  <span class="chip"><span class="flag">🇭🇷</span>Croatia</span>
  <span class="chip"><span class="flag">🇧🇦</span>Bosnia and Herzegovina</span>
  <span class="chip"><span class="flag">🇹🇷</span>Turkey</span>
  <span class="chip"><span class="flag">🇲🇹</span>Malta</span>
  <span class="chip"><span class="flag">🇱🇺</span>Luxembourg</span>
  <span class="chip"><span class="flag">🇲🇰</span>Macedonia</span>
  <span class="chip"><span class="flag">🇱🇦</span>Laos</span>
  <span class="chip"><span class="flag">🇬🇪</span>Georgia</span>
  <span class="chip"><span class="flag">🇦🇿</span>Azerbaijan</span>
  <span class="chip"><span class="flag">🇺🇿</span>Uzbekistan</span>
  <span class="chip"><span class="flag">🇰🇿</span>Kazakhstan</span>
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
