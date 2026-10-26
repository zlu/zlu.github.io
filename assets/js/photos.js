(function () {
  var grid = document.getElementById('photos-grid');
  var filtersEl = document.getElementById('photos-filters');
  var dialog = document.getElementById('photos-lightbox');
  if (!grid || !dialog) return;

  var img = document.getElementById('photos-lightbox-img');
  var titleEl = document.getElementById('photos-lightbox-title');
  var metaEl = document.getElementById('photos-lightbox-meta');
  var items = Array.prototype.slice.call(grid.querySelectorAll('.photos-item'));

  // Build place filters from captions
  var places = [];
  var seen = {};
  items.forEach(function (item) {
    var place = (item.getAttribute('data-place') || '').trim();
    if (place && !seen[place]) {
      seen[place] = true;
      places.push(place);
    }
  });
  if (filtersEl && places.length > 1) {
    places.sort(function (a, b) {
      return a.localeCompare(b);
    });
    places.forEach(function (place) {
      var btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'photos-filter';
      btn.setAttribute('data-place', place);
      btn.textContent = place;
      filtersEl.appendChild(btn);
    });
    filtersEl.hidden = false;

    filtersEl.addEventListener('click', function (e) {
      var btn = e.target.closest('.photos-filter');
      if (!btn) return;
      var place = btn.getAttribute('data-place') || '';
      Array.prototype.forEach.call(filtersEl.querySelectorAll('.photos-filter'), function (b) {
        b.classList.toggle('is-active', b === btn);
      });
      items.forEach(function (item) {
        var match = !place || item.getAttribute('data-place') === place;
        item.classList.toggle('is-hidden', !match);
      });
    });
  }

  grid.addEventListener('click', function (e) {
    var thumb = e.target.closest('.photos-thumb');
    if (!thumb) return;
    var full = thumb.getAttribute('data-full');
    if (!full) return;
    img.src = full;
    img.alt = thumb.getAttribute('data-title') || '';
    titleEl.textContent = thumb.getAttribute('data-title') || '';
    metaEl.textContent = thumb.getAttribute('data-meta') || '';
    if (typeof dialog.showModal === 'function') {
      dialog.showModal();
    } else {
      dialog.setAttribute('open', '');
    }
  });

  dialog.addEventListener('click', function (e) {
    if (e.target === dialog) dialog.close();
  });

  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape' && dialog.open) dialog.close();
  });
})();
