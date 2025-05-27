---
layout: default
title: My Travels
permalink: /travels/
---

# My Travels


## Map of Countries Visited

<div id="travel-map" style="height: 500px; margin-bottom: 2em;"></div>

<!-- Google Maps JS API -->
<script>
const visitedCountries = [
  { name: "USA", lat: 39.8283, lng: -98.5795 },
  { name: "Canada", lat: 56.1304, lng: -106.3468 },
  { name: "Germany", lat: 51.1657, lng: 10.4515 },
  { name: "Netherlands", lat: 52.1326, lng: 5.2913 },
  { name: "Spain", lat: 40.4637, lng: -3.7492 },
  { name: "UK", lat: 55.3781, lng: -3.4360 },
  { name: "Switzerland", lat: 46.8182, lng: 8.2275 },
  { name: "France", lat: 46.6034, lng: 1.8883 },
  { name: "Belgium", lat: 50.5039, lng: 4.4699 },
  { name: "Denmark", lat: 56.2639, lng: 9.5018 },
  { name: "Portugal", lat: 39.3999, lng: -8.2245 },
  { name: "Italy", lat: 41.8719, lng: 12.5674 },
  { name: "Vatican City", lat: 41.9029, lng: 12.4534 },
  { name: "Czech", lat: 49.8175, lng: 15.4730 },
  { name: "Austria", lat: 47.5162, lng: 14.5501 },
  { name: "Hungary", lat: 47.1625, lng: 19.5033 },
  { name: "Israel", lat: 31.0461, lng: 34.8516 },
  { name: "China", lat: 35.8617, lng: 104.1954 },
  { name: "Panama", lat: 8.5380, lng: -80.7821 },
  { name: "Mexico", lat: 23.6345, lng: -102.5528 },
  { name: "Thailand", lat: 15.8700, lng: 100.9925 },
  { name: "Cambodia", lat: 12.5657, lng: 104.9910 },
  { name: "Viet Nam", lat: 14.0583, lng: 108.2772 },
  { name: "Korea", lat: 35.9078, lng: 127.7669 },
  { name: "Monaco", lat: 43.7384, lng: 7.4246 },
  { name: "Japan", lat: 36.2048, lng: 138.2529 },
  { name: "Brazil", lat: -14.2350, lng: -51.9253 },
  { name: "Peru", lat: -9.1900, lng: -75.0152 },
  { name: "Colombia", lat: 4.5709, lng: -74.2973 },
  { name: "Argentina", lat: -38.4161, lng: -63.6167 },
  { name: "Chile", lat: -35.6751, lng: -71.5430 },
  { name: "Sweden", lat: 60.1282, lng: 18.6435 },
  { name: "Finland", lat: 61.9241, lng: 25.7482 },
  { name: "Estonia", lat: 58.5953, lng: 25.0136 },
  { name: "Latvia", lat: 56.8796, lng: 24.6032 },
  { name: "Lithuania", lat: 55.1694, lng: 23.8813 },
  { name: "Poland", lat: 51.9194, lng: 19.1451 },
  { name: "Slovakia", lat: 48.6690, lng: 19.6990 },
  { name: "Serbia", lat: 44.0165, lng: 21.0059 },
  { name: "Bulgaria", lat: 42.7339, lng: 25.4858 },
  { name: "Greece", lat: 39.0742, lng: 21.8243 },
  { name: "Morocco", lat: 31.7917, lng: -7.0926 },
  { name: "Romania", lat: 45.9432, lng: 24.9668 },
  { name: "Ireland", lat: 53.4129, lng: -8.2439 },
  { name: "Albania", lat: 41.1533, lng: 20.1683 },
  { name: "Montenegro", lat: 42.7087, lng: 19.3744 },
  { name: "Croatia", lat: 45.1000, lng: 15.2000 },
  { name: "Bosnia and Herzegovina", lat: 43.9159, lng: 17.6791 },
  { name: "Turkey", lat: 38.9637, lng: 35.2433 },
  { name: "Malta", lat: 35.9375, lng: 14.3754 },
  { name: "Luxembourg", lat: 49.8153, lng: 6.1296 },
  { name: "Macedonia", lat: 41.6086, lng: 21.7453 },
  { name: "Laos", lat: 19.8563, lng: 102.4955 },
  { name: "Georgia", lat: 42.3154, lng: 43.3569 },
  { name: "Azerbaijan", lat: 40.1431, lng: 47.5769 },
  { name: "Uzbekistan", lat: 41.3775, lng: 64.5853 },
  { name: "Kazakhstan", lat: 48.0196, lng: 66.9237 }
];

function initMap() {
  var map = new google.maps.Map(document.getElementById('travel-map'), {
    zoom: 2,
    center: {lat: 20, lng: 0},
    mapTypeId: 'terrain'
  });

  visitedCountries.forEach(function(country) {
    new google.maps.Marker({
      position: {lat: country.lat, lng: country.lng},
      map: map,
      title: country.name
    });
  });
}
window.initMap = initMap;
</script>
<script async defer src="https://maps.googleapis.com/maps/api/js?key=AIzaSyDAmbs-168EHpvMvoOP30CYWP3X3yCLJ38&callback=initMap"></script>


## Countries Visited

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
