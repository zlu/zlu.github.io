(function () {
  var root = document.getElementById('posts-index');
  if (!root) return;

  var perPage = parseInt(root.getAttribute('data-per-page'), 10) || 8;
  var tabs = root.querySelectorAll('.posts-lang-tab');
  var items = Array.prototype.slice.call(root.querySelectorAll('.post-index-item'));
  var pagination = document.getElementById('posts-pagination');
  var statusEl = document.getElementById('posts-page-status');
  var emptyEl = document.getElementById('posts-empty');
  var searchInput = document.getElementById('posts-search-input');
  var searchStatus = document.getElementById('posts-search-status');
  var prevBtn = pagination.querySelector('[data-action="prev"]');
  var nextBtn = pagination.querySelector('[data-action="next"]');

  var state = { lang: 'en', page: 1, query: '' };

  function normalize(s) {
    return String(s || '')
      .toLowerCase()
      .normalize('NFKD')
      .replace(/[\u0300-\u036f]/g, '')
      .trim();
  }

  // Subsequence fuzzy score (fzf-style). Higher is better; -1 = no match.
  function fuzzyScore(haystack, needle) {
    if (!needle) return 0;
    if (!haystack) return -1;

    var h = normalize(haystack);
    var n = normalize(needle);
    if (!n) return 0;

    // Prefer substring hits.
    var subIdx = h.indexOf(n);
    if (subIdx !== -1) {
      return 1000 - subIdx * 2 - (h.length - n.length);
    }

    var hi = 0;
    var score = 0;
    var consecutive = 0;
    var first = -1;

    for (var ni = 0; ni < n.length; ni++) {
      var ch = n.charAt(ni);
      var found = -1;
      for (; hi < h.length; hi++) {
        if (h.charAt(hi) === ch) {
          found = hi;
          hi++;
          break;
        }
      }
      if (found === -1) return -1;
      if (first === -1) first = found;
      if (ni > 0 && found === hi - 1 && consecutive > 0) {
        consecutive += 1;
        score += 8 * consecutive;
      } else {
        consecutive = 1;
        score += 2;
      }
      // Slight preference for earlier matches.
      score += Math.max(0, 12 - found);
    }

    score -= (h.length - n.length) * 0.1;
    score -= first;
    return score;
  }

  function fieldScore(title, tags, desc, token) {
    var t = fuzzyScore(title, token);
    var g = fuzzyScore(tags, token);
    var d = fuzzyScore(desc, token);
    var best = -1;
    if (t >= 0) best = Math.max(best, t * 3);
    if (g >= 0) best = Math.max(best, g * 2.2);
    if (d >= 0) best = Math.max(best, d);
    return best;
  }

  function scoreItem(el, query) {
    var q = normalize(query);
    if (!q) return 0;

    var title = el.getAttribute('data-title') || '';
    var tags = el.getAttribute('data-tags') || '';
    var desc = el.getAttribute('data-desc') || '';
    var tokens = q.split(/\s+/).filter(Boolean);
    var total = 0;

    for (var i = 0; i < tokens.length; i++) {
      var s = fieldScore(title, tags, desc, tokens[i]);
      if (s < 0) return -1;
      total += s;
    }
    return total;
  }

  function readParams() {
    var params = new URLSearchParams(window.location.search);
    var lang = params.get('lang');
    var page = parseInt(params.get('page'), 10);
    var q = params.get('q');
    if (lang === 'zh' || lang === 'en') state.lang = lang;
    if (page >= 1) state.page = page;
    if (q) {
      state.query = q;
      if (searchInput) searchInput.value = q;
    }
  }

  function writeParams() {
    var params = new URLSearchParams(window.location.search);
    params.set('lang', state.lang);
    if (state.page > 1) params.set('page', String(state.page));
    else params.delete('page');
    if (state.query) params.set('q', state.query);
    else params.delete('q');
    var qs = params.toString();
    var url = window.location.pathname + (qs ? '?' + qs : '') + window.location.hash;
    window.history.replaceState(null, '', url);
  }

  function filtered() {
    var scored = [];
    for (var i = 0; i < items.length; i++) {
      var el = items[i];
      if (el.getAttribute('data-lang') !== state.lang) continue;
      var score = scoreItem(el, state.query);
      if (score < 0) continue;
      scored.push({ el: el, score: score });
    }
    if (state.query) {
      scored.sort(function (a, b) {
        return b.score - a.score;
      });
    }
    return scored.map(function (row) {
      return row.el;
    });
  }

  function pageCount(n) {
    return Math.max(1, Math.ceil(n / perPage));
  }

  function render() {
    var list = filtered();
    var pages = pageCount(list.length);
    if (state.page > pages) state.page = pages;
    if (state.page < 1) state.page = 1;

    var start = (state.page - 1) * perPage;
    var end = start + perPage;

    items.forEach(function (el) {
      el.hidden = true;
    });
    list.forEach(function (el, i) {
      el.hidden = !(i >= start && i < end);
    });

    tabs.forEach(function (tab) {
      var active = tab.getAttribute('data-lang') === state.lang;
      tab.classList.toggle('is-active', active);
      tab.setAttribute('aria-selected', active ? 'true' : 'false');
    });

    if (emptyEl) {
      emptyEl.hidden = list.length > 0;
    }

    if (searchStatus) {
      if (state.query) {
        searchStatus.hidden = false;
        searchStatus.textContent =
          list.length === 1
            ? '1 match'
            : list.length + ' matches';
      } else {
        searchStatus.hidden = true;
        searchStatus.textContent = '';
      }
    }

    if (list.length === 0) {
      pagination.hidden = true;
      writeParams();
      return;
    }

    pagination.hidden = pages <= 1;
    statusEl.textContent = 'Page ' + state.page + ' of ' + pages;
    prevBtn.disabled = state.page <= 1;
    nextBtn.disabled = state.page >= pages;

    writeParams();
  }

  tabs.forEach(function (tab) {
    tab.addEventListener('click', function () {
      var lang = tab.getAttribute('data-lang');
      if (lang === state.lang) return;
      state.lang = lang;
      state.page = 1;
      render();
    });
  });

  prevBtn.addEventListener('click', function () {
    if (state.page <= 1) return;
    state.page -= 1;
    render();
    root.scrollIntoView({ behavior: 'smooth', block: 'start' });
  });

  nextBtn.addEventListener('click', function () {
    state.page += 1;
    render();
    root.scrollIntoView({ behavior: 'smooth', block: 'start' });
  });

  if (searchInput) {
    var debounceTimer = null;
    searchInput.addEventListener('input', function () {
      var value = searchInput.value;
      window.clearTimeout(debounceTimer);
      debounceTimer = window.setTimeout(function () {
        state.query = value.trim();
        state.page = 1;
        render();
      }, 120);
    });

    searchInput.addEventListener('keydown', function (e) {
      if (e.key === 'Escape' && searchInput.value) {
        searchInput.value = '';
        state.query = '';
        state.page = 1;
        render();
      }
    });
  }

  readParams();
  render();
})();
