(function () {
  var root = document.getElementById('posts-index');
  if (!root) return;

  var perPage = parseInt(root.getAttribute('data-per-page'), 10) || 8;
  var tabs = root.querySelectorAll('.posts-lang-tab');
  var listEl = document.getElementById('posts-list');
  var items = Array.prototype.slice.call(root.querySelectorAll('.post-index-item'));
  var pagination = document.getElementById('posts-pagination');
  var statusEl = document.getElementById('posts-page-status');
  var emptyEl = document.getElementById('posts-empty');
  var searchInput = document.getElementById('posts-search-input');
  var searchStatus = document.getElementById('posts-search-status');
  var clearBtn = document.getElementById('posts-search-clear');
  var prevBtn = pagination.querySelector('[data-action="prev"]');
  var nextBtn = pagination.querySelector('[data-action="next"]');

  var state = { lang: 'en', page: 1, query: '' };
  var originalOrder = items.slice();

  function normalize(s) {
    return String(s || '')
      .toLowerCase()
      .normalize('NFKD')
      .replace(/[\u0300-\u036f]/g, '')
      .trim();
  }

  function wordsOf(s) {
    return normalize(s)
      .split(/[^a-z0-9\u4e00-\u9fff]+/)
      .filter(Boolean);
  }

  // Typo-tolerant subsequence within one word (at most 2 skipped chars).
  // Exact whole-word / prefix matches are handled separately; mid-word
  // substrings like "lora" inside "exploration" must not rank.
  function fuzzyWordScore(word, needle) {
    if (!needle) return 0;
    if (!word || needle.length < 3) return -1;

    var h = word;
    var n = needle;
    if (h.indexOf(n) !== -1) return -1;

    var hi = 0;
    var first = -1;
    var last = -1;
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
      last = found;
    }

    var span = last - first + 1;
    if (span > n.length + 2) return -1;
    return 220 - (span - n.length) * 40 - (h.length - n.length);
  }

  function isWordChar(ch) {
    return /[a-z0-9\u4e00-\u9fff]/.test(ch);
  }

  function hasWordBoundedPhrase(haystack, phrase) {
    var h = normalize(haystack);
    var n = normalize(phrase);
    if (!n || !h) return -1;
    var idx = h.indexOf(n);
    while (idx !== -1) {
      var beforeOk = idx === 0 || !isWordChar(h.charAt(idx - 1));
      var afterOk =
        idx + n.length >= h.length || !isWordChar(h.charAt(idx + n.length));
      if (beforeOk && afterOk) return idx;
      idx = h.indexOf(n, idx + 1);
    }
    return -1;
  }

  function fieldTokenScore(haystack, token) {
    var n = normalize(token);
    if (!n) return 0;
    var words = wordsOf(haystack);
    if (!words.length) return -1;

    var best = -1;
    for (var i = 0; i < words.length; i++) {
      var w = words[i];
      if (w === n) {
        best = Math.max(best, 1500);
        continue;
      }
      // Prefix of a word: "pretrain" → "pretraining"
      if (w.indexOf(n) === 0) {
        best = Math.max(best, 1200 - (w.length - n.length));
        continue;
      }
      // Short tokens must be whole words (or prefixes above).
      if (n.length < 3) continue;
      var fz = fuzzyWordScore(w, n);
      if (fz > best) best = fz;
    }
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

    // Word-bounded phrase in title wins outright ("xiaomi tv", not "lora" in "exploration").
    var phraseIdx = hasWordBoundedPhrase(title, q);
    if (phraseIdx !== -1) {
      return (
        100000 -
        phraseIdx * 10 -
        Math.max(0, normalize(title).length - q.length)
      );
    }

    // All query tokens match title words in order → strong boost.
    var titleWords = wordsOf(title);
    var ordered = 0;
    var wi = 0;
    for (var ti = 0; ti < tokens.length; ti++) {
      var tok = tokens[ti];
      var hit = -1;
      for (; wi < titleWords.length; wi++) {
        if (
          titleWords[wi] === tok ||
          titleWords[wi].indexOf(tok) === 0 ||
          (tok.length >= 3 && fuzzyWordScore(titleWords[wi], tok) >= 0)
        ) {
          hit = wi;
          wi++;
          break;
        }
      }
      if (hit === -1) break;
      ordered += 1;
    }
    if (ordered === tokens.length) {
      total += 50000 - wi;
    }

    for (var i = 0; i < tokens.length; i++) {
      var t = fieldTokenScore(title, tokens[i]);
      var g = fieldTokenScore(tags, tokens[i]);
      var d = fieldTokenScore(desc, tokens[i]);
      var best = -1;
      if (t >= 0) best = Math.max(best, t * 4);
      if (g >= 0) best = Math.max(best, g * 2.5);
      if (d >= 0) best = Math.max(best, d * 0.55);
      if (best < 0) return -1;
      total += best;
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
        if (b.score !== a.score) return b.score - a.score;
        return originalOrder.indexOf(a.el) - originalOrder.indexOf(b.el);
      });
    }
    return scored.map(function (row) {
      return row.el;
    });
  }

  function pageCount(n) {
    return Math.max(1, Math.ceil(n / perPage));
  }

  function syncClearBtn() {
    if (!clearBtn) return;
    clearBtn.hidden = !searchInput || !searchInput.value;
  }

  function render() {
    var list = filtered();
    var pages = pageCount(list.length);
    if (state.page > pages) state.page = pages;
    if (state.page < 1) state.page = 1;

    var start = (state.page - 1) * perPage;
    var end = start + perPage;

    // Reorder DOM so score order is visual order (not just pagination order).
    var frag = document.createDocumentFragment();
    if (state.query) {
      list.forEach(function (el) {
        frag.appendChild(el);
      });
      originalOrder.forEach(function (el) {
        if (list.indexOf(el) === -1) frag.appendChild(el);
      });
    } else {
      originalOrder.forEach(function (el) {
        frag.appendChild(el);
      });
    }
    listEl.appendChild(frag);

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

    root.classList.toggle('is-searching', !!state.query);

    if (searchStatus) {
      if (state.query) {
        searchStatus.hidden = false;
        searchStatus.textContent =
          list.length === 0
            ? 'No matches'
            : list.length === 1
              ? '1 match'
              : list.length + ' matches';
      } else {
        searchStatus.hidden = true;
        searchStatus.textContent = '';
      }
    }

    syncClearBtn();

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

  function setQuery(value, fromInput) {
    state.query = String(value || '').trim();
    state.page = 1;
    if (!fromInput && searchInput) searchInput.value = value || '';
    render();
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
      syncClearBtn();
      window.clearTimeout(debounceTimer);
      debounceTimer = window.setTimeout(function () {
        setQuery(value, true);
      }, 80);
    });

    searchInput.addEventListener('keydown', function (e) {
      if (e.key === 'Escape' && searchInput.value) {
        e.preventDefault();
        setQuery('');
        searchInput.focus();
      }
    });
  }

  if (clearBtn) {
    clearBtn.addEventListener('click', function () {
      setQuery('');
      if (searchInput) searchInput.focus();
    });
  }

  readParams();
  render();
})();
