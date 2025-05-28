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

<details>
  <summary style="font-size:1.1em; cursor:pointer;">Show Countries Visited</summary>
  <ol style="margin-top:1em;">
    <li>USA</li>
    <li>Canada</li>
    <li>Germany</li>
    <li>Netherlands</li>
    <li>Spain</li>
    <li>UK</li>
    <li>Switzerland</li>
    <li>France</li>
    <li>Belgium</li>
    <li>Denmark</li>
    <li>Portugal</li>
    <li>Italy</li>
    <li>Vatican City</li>
    <li>Czech</li>
    <li>Austria</li>
    <li>Hungary</li>
    <li>Israel</li>
    <li>China</li>
    <li>Panama</li>
    <li>Mexico</li>
    <li>Thailand</li>
    <li>Cambodia</li>
    <li>Viet Nam</li>
    <li>Korea</li>
    <li>Monaco</li>
    <li>Japan</li>
    <li>Brazil</li>
    <li>Peru</li>
    <li>Colombia</li>
    <li>Argentina</li>
    <li>Chile</li>
    <li>Sweden</li>
    <li>Finland</li>
    <li>Estonia</li>
    <li>Latvia</li>
    <li>Lithuania</li>
    <li>Poland</li>
    <li>Slovakia</li>
    <li>Serbia</li>
    <li>Bulgaria</li>
    <li>Greece</li>
    <li>Morocco</li>
    <li>Romania</li>
    <li>Ireland</li>
    <li>Albania</li>
    <li>Montenegro</li>
    <li>Croatia</li>
    <li>Bosnia and Herzegovina</li>
    <li>Turkey</li>
    <li>Malta</li>
    <li>Luxembourg</li>
    <li>Macedonia</li>
    <li>Laos</li>
    <li>Georgia</li>
    <li>Azerbaijan</li>
    <li>Uzbekistan</li>
    <li>Kazakhstan</li>
  </ol>
</details>
