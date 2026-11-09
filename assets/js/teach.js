document.addEventListener('DOMContentLoaded', function () {
  document.body.classList.add('teach-page-active');

  document.querySelectorAll('[data-teach-scroll-book]').forEach(function (btn) {
    btn.addEventListener('click', function () {
      if (window.ZlumeWebAnalytics && window.ZlumeWebAnalytics.trackFunnelMilestone) {
        window.ZlumeWebAnalytics.trackFunnelMilestone('wechat_cta_clicked');
      }
      var book = document.querySelector('.teach-hero-book');
      if (!book) return;
      book.scrollIntoView({ behavior: 'smooth', block: 'center' });
      book.classList.add('teach-book-pulse');
      setTimeout(function () {
        book.classList.remove('teach-book-pulse');
      }, 1200);
    });
  });

  initSubjectTabs();
  initHubSearch();
  initUniDirectory();
});

function initSubjectTabs() {
  var root = document.querySelector('[data-teach-subject-tabs]');
  if (!root) return;

  var tabs = Array.prototype.slice.call(root.querySelectorAll('[role="tab"]'));
  var panels = Array.prototype.slice.call(root.querySelectorAll('[role="tabpanel"]'));

  function activate(tab) {
    var key = tab.getAttribute('data-tab');
    tabs.forEach(function (t) {
      var on = t === tab;
      t.classList.toggle('is-active', on);
      t.setAttribute('aria-selected', on ? 'true' : 'false');
      t.tabIndex = on ? 0 : -1;
    });
    panels.forEach(function (panel) {
      var on = panel.getAttribute('data-panel') === key;
      panel.classList.toggle('is-active', on);
      if (on) {
        panel.removeAttribute('hidden');
      } else {
        panel.setAttribute('hidden', '');
      }
    });
  }

  tabs.forEach(function (tab, index) {
    tab.addEventListener('click', function () {
      activate(tab);
    });
    tab.addEventListener('keydown', function (e) {
      var next = null;
      if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
        next = tabs[(index + 1) % tabs.length];
      } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
        next = tabs[(index - 1 + tabs.length) % tabs.length];
      } else if (e.key === 'Home') {
        next = tabs[0];
      } else if (e.key === 'End') {
        next = tabs[tabs.length - 1];
      }
      if (!next) return;
      e.preventDefault();
      next.focus();
      activate(next);
    });
  });
}

function initHubSearch() {
  var root = document.querySelector('[data-teach-hub-search]');
  if (!root) return;

  var input = root.querySelector('#teach-hub-search-input');
  var results = root.querySelector('#teach-hub-search-results');
  var emptyMsg = root.querySelector('#teach-hub-search-empty');
  var hint = root.querySelector('.teach-hub-search-hint');
  var dataEl = document.getElementById('teach-hub-search-data');
  if (!input || !results || !dataEl) return;

  var catalog = [];
  try {
    catalog = JSON.parse(dataEl.textContent || '[]');
  } catch (err) {
    catalog = [];
  }

  function applyLangPlaceholder(langCode) {
    var key = langCode === 'cn' ? 'data-cn-placeholder' : 'data-en-placeholder';
    var text = input.getAttribute(key);
    if (text) input.setAttribute('placeholder', text);
  }

  function isCn() {
    return (document.documentElement.lang || '').toLowerCase() === 'zh-cn';
  }

  function normalize(s) {
    return String(s || '')
      .toLowerCase()
      .replace(/[^a-z0-9\u4e00-\u9fff]+/g, ' ')
      .trim();
  }

  function compact(s) {
    return String(s || '')
      .toLowerCase()
      .replace(/[^a-z0-9\u4e00-\u9fff]+/g, '');
  }

  function scoreText(hay, query, queryCompact) {
    if (!query) return 0;
    var n = normalize(hay);
    var c = compact(hay);
    if (n === query) return 100;
    if (c === queryCompact) return 95;
    if (n.indexOf(query) === 0 || c.indexOf(queryCompact) === 0) return 80;
    if (n.indexOf(query) !== -1 || c.indexOf(queryCompact) !== -1) return 60;
    return 0;
  }

  function search(queryRaw) {
    var query = normalize(queryRaw);
    var queryCompact = compact(queryRaw);
    if (!query || query.length < 2) return [];

    var hits = [];
    catalog.forEach(function (uni) {
      var uniScore = Math.max(
        scoreText(uni.n, query, queryCompact),
        scoreText(uni.s, query, queryCompact),
        scoreText(uni.ab, query, queryCompact),
        scoreText(uni.cn, query, queryCompact),
        scoreText(uni.lc, query, queryCompact)
      );
      if (uniScore > 0) {
        hits.push({
          kind: 'uni',
          score: uniScore + 5,
          href: '/teach/universities/' + uni.s + '/',
          title: uni.lc || uni.n,
          meta: uni.ab || uni.s
        });
      }

      (uni.c || []).forEach(function (course) {
        var aliasHay = Array.isArray(course.a) ? course.a.join(' ') : '';
        var courseScore = Math.max(
          scoreText(course.c, query, queryCompact),
          scoreText(course.t, query, queryCompact),
          scoreText(course.sl, query, queryCompact),
          scoreText(aliasHay, query, queryCompact),
          scoreText(uni.n + ' ' + course.c, query, queryCompact),
          scoreText((uni.ab || '') + ' ' + course.c, query, queryCompact),
          scoreText((uni.cn || '') + ' ' + course.c, query, queryCompact)
        );
        if (courseScore > 0) {
          hits.push({
            kind: 'course',
            score: courseScore,
            href: '/teach/courses/' + uni.s + '/' + course.sl + '/',
            title: course.c + (course.t ? ' · ' + course.t : ''),
            meta: uni.lc || uni.n
          });
        }
      });
    });

    hits.sort(function (a, b) {
      if (b.score !== a.score) return b.score - a.score;
      return a.title.localeCompare(b.title);
    });

    var seen = {};
    var out = [];
    hits.forEach(function (hit) {
      if (seen[hit.href]) return;
      seen[hit.href] = true;
      out.push(hit);
    });
    return out.slice(0, 12);
  }

  function render(query) {
    var hits = search(query);
    results.innerHTML = '';

    if (!normalize(query) || normalize(query).length < 2) {
      results.hidden = true;
      if (emptyMsg) emptyMsg.hidden = true;
      if (hint) hint.hidden = false;
      return;
    }

    if (hint) hint.hidden = true;

    if (!hits.length) {
      results.hidden = true;
      if (emptyMsg) emptyMsg.hidden = false;
      return;
    }

    if (emptyMsg) emptyMsg.hidden = true;
    results.hidden = false;

    var kindUni = isCn() ? '大学' : 'University';
    var kindCourse = isCn() ? '课程' : 'Course';

    hits.forEach(function (hit) {
      var li = document.createElement('li');
      var a = document.createElement('a');
      a.href = hit.href;
      a.setAttribute('role', 'option');

      var kind = document.createElement('span');
      kind.className = 'teach-hub-search-kind';
      kind.textContent = hit.kind === 'uni' ? kindUni : kindCourse;

      var title = document.createElement('strong');
      title.textContent = hit.title;

      var meta = document.createElement('span');
      meta.textContent = hit.meta;

      a.appendChild(kind);
      a.appendChild(title);
      a.appendChild(meta);
      li.appendChild(a);
      results.appendChild(li);
    });
  }

  var timer = null;
  function onInput() {
    clearTimeout(timer);
    timer = setTimeout(function () {
      render(input.value);
    }, 80);
  }

  input.addEventListener('input', onInput);
  input.addEventListener('keyup', onInput);
  input.addEventListener('search', onInput);

  document.addEventListener('languageChanged', function (e) {
    var langCode = e && e.detail && e.detail.language === 'cn' ? 'cn' : 'en';
    applyLangPlaceholder(langCode);
    if (input.value) render(input.value);
  });

  var htmlLang = document.documentElement.lang.toLowerCase();
  applyLangPlaceholder(htmlLang === 'zh-cn' ? 'cn' : 'en');
}

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
