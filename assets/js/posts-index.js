(function () {
  var root = document.getElementById('posts-index');
  if (!root) return;

  var perPage = parseInt(root.getAttribute('data-per-page'), 10) || 8;
  var tabs = root.querySelectorAll('.posts-lang-tab');
  var items = Array.prototype.slice.call(root.querySelectorAll('.post-index-item'));
  var pagination = document.getElementById('posts-pagination');
  var statusEl = document.getElementById('posts-page-status');
  var prevBtn = pagination.querySelector('[data-action="prev"]');
  var nextBtn = pagination.querySelector('[data-action="next"]');

  var state = { lang: 'en', page: 1 };

  function readParams() {
    var params = new URLSearchParams(window.location.search);
    var lang = params.get('lang');
    var page = parseInt(params.get('page'), 10);
    if (lang === 'zh' || lang === 'en') state.lang = lang;
    if (page >= 1) state.page = page;
  }

  function writeParams() {
    var params = new URLSearchParams(window.location.search);
    params.set('lang', state.lang);
    if (state.page > 1) params.set('page', String(state.page));
    else params.delete('page');
    var qs = params.toString();
    var url = window.location.pathname + (qs ? '?' + qs : '') + window.location.hash;
    window.history.replaceState(null, '', url);
  }

  function filtered() {
    return items.filter(function (el) {
      return el.getAttribute('data-lang') === state.lang;
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

    if (list.length === 0) {
      pagination.hidden = true;
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

  readParams();
  render();
})();
