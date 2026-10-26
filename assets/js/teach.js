document.addEventListener('DOMContentLoaded', function () {
  document.body.classList.add('teach-page-active');

  document.querySelectorAll('[data-teach-scroll-book]').forEach(function (btn) {
    btn.addEventListener('click', function () {
      var book = document.querySelector('.teach-hero-book');
      if (!book) return;
      book.scrollIntoView({ behavior: 'smooth', block: 'center' });
      book.classList.add('teach-book-pulse');
      setTimeout(function () {
        book.classList.remove('teach-book-pulse');
      }, 1200);
    });
  });

  initUniDirectory();
});

function initUniDirectory() {
  var root = document.getElementById('teach-uni-directory');
  if (!root) return;

  var search = root.querySelector('#teach-uni-search');
  var chips = root.querySelectorAll('[data-region-filter]');
  var emptyMsg = root.querySelector('.teach-uni-no-results');
  var searchMeta = root.querySelector('.teach-uni-search-meta');
  var activeRegion = 'all';

  function applyLangPlaceholder(langCode) {
    if (!search) return;
    var key = langCode === 'cn' ? 'data-cn-placeholder' : 'data-en-placeholder';
    var text = search.getAttribute(key);
    if (text) search.setAttribute('placeholder', text);
  }

  function applyFilter() {
    var query = ((search && search.value) || '').toLowerCase().trim();
    var visibleCount = 0;

    root.querySelectorAll('.teach-uni-item, .teach-uni-featured-item').forEach(function (item) {
      var region = item.getAttribute('data-region') || '';
      var hay = (item.getAttribute('data-filter-text') || '').toLowerCase();
      var regionOk = activeRegion === 'all' || region === activeRegion;
      var queryOk = !query || hay.indexOf(query) !== -1;
      var show = regionOk && queryOk;
      item.classList.toggle('is-hidden', !show);
      if (show && item.classList.contains('teach-uni-item')) visibleCount++;
    });

    root.querySelectorAll('[data-region-section]').forEach(function (section) {
      var key = section.getAttribute('data-region-section');
      var regionOk = activeRegion === 'all' || key === activeRegion;
      var anyVisible = Array.prototype.some.call(
        section.querySelectorAll('.teach-uni-item'),
        function (item) {
          return !item.classList.contains('is-hidden');
        }
      );
      section.classList.toggle('is-hidden', !regionOk || !anyVisible);
    });

    var featured = root.querySelector('.teach-uni-featured');
    if (featured) {
      var anyFeatured = Array.prototype.some.call(
        featured.querySelectorAll('.teach-uni-featured-item'),
        function (item) {
          return !item.classList.contains('is-hidden');
        }
      );
      featured.classList.toggle('is-hidden', !anyFeatured);
    }

    var jump = root.querySelector('.teach-uni-jump');
    if (jump) {
      jump.querySelectorAll('a').forEach(function (link) {
        var href = link.getAttribute('href') || '';
        var key = href.replace('#region-', '');
        var section = root.querySelector('[data-region-section="' + key + '"]');
        var hide =
          activeRegion !== 'all'
            ? key !== activeRegion
            : !!(section && section.classList.contains('is-hidden'));
        link.classList.toggle('is-hidden', hide);
        link.style.display = hide ? 'none' : '';
      });
      var anyJump = Array.prototype.some.call(jump.querySelectorAll('a'), function (a) {
        return a.style.display !== 'none';
      });
      jump.style.display = anyJump ? '' : 'none';
    }

    if (emptyMsg) emptyMsg.hidden = visibleCount !== 0;
    if (searchMeta) searchMeta.style.display = query || activeRegion !== 'all' ? 'none' : '';
  }

  if (search) {
    var handler = function () {
      applyFilter();
    };
    search.addEventListener('input', handler);
    search.addEventListener('keyup', handler);
  }

  chips.forEach(function (chip) {
    chip.addEventListener('click', function () {
      activeRegion = chip.getAttribute('data-region-filter') || 'all';
      chips.forEach(function (c) {
        c.classList.remove('is-active');
      });
      chip.classList.add('is-active');
      applyFilter();
      if (activeRegion !== 'all') {
        var section = root.querySelector('[data-region-section="' + activeRegion + '"]');
        if (section && !section.classList.contains('is-hidden')) {
          section.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      }
    });
  });

  document.addEventListener('languageChanged', function (e) {
    var langCode = e && e.detail && e.detail.language === 'cn' ? 'cn' : 'en';
    applyLangPlaceholder(langCode);
  });

  var htmlLang = document.documentElement.lang.toLowerCase();
  applyLangPlaceholder(htmlLang === 'zh-cn' ? 'cn' : 'en');
  applyFilter();
}
