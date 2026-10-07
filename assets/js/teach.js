document.addEventListener('DOMContentLoaded', function() {
    document.body.classList.add('teach-page-active');

    function initializeTabs() {
        document.querySelectorAll('.lang-en .course-tabs, .lang-cn .course-tabs').forEach(tabContainer => {
            const tabBtns = tabContainer.querySelectorAll('.tab-btn');
            const tabContents = tabContainer.querySelectorAll('.tab-content');

            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            if (tabBtns.length > 0) {
                tabBtns[0].classList.add('active');
                const firstTabId = tabBtns[0].getAttribute('data-tab');
                let firstTabContent = tabContainer.querySelector(`.tab-content#${firstTabId}`);
                if (!firstTabContent) {
                    firstTabContent = document.getElementById(firstTabId);
                }
                if (firstTabContent) {
                    firstTabContent.classList.add('active');
                }
            }

            tabBtns.forEach(btn => {
                if (!btn.hasAttribute('data-tab-listener-added')) {
                    btn.addEventListener('click', () => {
                        const currentTabContainer = btn.closest('.course-tabs');
                        if (!currentTabContainer) return;

                        const siblingBtns = currentTabContainer.querySelectorAll('.tab-btn');
                        const contentContainer = currentTabContainer.querySelector('.tab-content-container') || currentTabContainer;

                        siblingBtns.forEach(b => b.classList.remove('active'));
                        contentContainer.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

                        btn.classList.add('active');
                        const tabId = btn.getAttribute('data-tab');
                        let tabContent = contentContainer.querySelector(`.tab-content#${tabId}`);
                        if (!tabContent) {
                            tabContent = document.getElementById(tabId);
                        }
                        if (tabContent) {
                            tabContent.classList.add('active');
                        }
                    });
                    btn.setAttribute('data-tab-listener-added', 'true');
                }
            });
        });
    }

    function updateTabLanguage(lang) {
        document.querySelectorAll('.lang-en .tab-btn, .lang-cn .tab-btn').forEach(btn => {
            const text = btn.getAttribute(`data-${lang}`);
            if (text) btn.textContent = text;
        });
    }

    document.addEventListener('languageChanged', function(e) {
        const langCode = (e && e.detail && e.detail.language === 'cn') ? 'cn' : 'en';
        updateTabLanguage(langCode);
    });

    initializeTabs();

    const currentHtmlLang = document.documentElement.lang.toLowerCase();
    const initialLangCode = currentHtmlLang === 'zh-cn' ? 'cn' : 'en';
    updateTabLanguage(initialLangCode);

    document.querySelectorAll('.course-search-input').forEach(input => {
        if (!input.hasAttribute('data-course-listener')) {
            const handler = function() { filterCourses(this); };
            input.addEventListener('input', handler);
            input.addEventListener('keyup', handler);
            input.setAttribute('data-course-listener', 'true');
        }
    });

    document.querySelectorAll('.teach-chip').forEach(chip => {
        chip.addEventListener('click', function() {
            const query = this.getAttribute('data-filter') || '';
            const container = this.closest('.course-list-container');
            if (!container) return;
            const input = container.querySelector('.course-search-input');
            if (!input) return;

            container.querySelectorAll('.teach-chip').forEach(c => c.classList.remove('is-active'));
            this.classList.add('is-active');

            input.value = query;
            filterCourses(input);
            input.focus();
            container.scrollIntoView({ behavior: 'smooth', block: 'start' });
        });
    });

    document.querySelectorAll('[data-teach-scroll-book]').forEach(btn => {
        btn.addEventListener('click', function() {
            const book = document.querySelector('.teach-hero-book');
            if (book) {
                book.scrollIntoView({ behavior: 'smooth', block: 'center' });
                book.classList.add('teach-book-pulse');
                setTimeout(() => book.classList.remove('teach-book-pulse'), 1200);
            }
        });
    });

    filterCourses();
});

function filterCourses(inputEl) {
  function updateMessages(container, hasQuery, matchCount) {
    const hint = container.querySelector('.course-search-hint');
    const emptyMsg = container.querySelector('.course-no-results');
    if (!hasQuery) {
      if (hint) hint.style.display = '';
      if (emptyMsg) emptyMsg.style.display = 'none';
      return;
    }
    if (hint) hint.style.display = 'none';
    if (emptyMsg) emptyMsg.style.display = matchCount === 0 ? '' : 'none';
  }

  if (!inputEl) {
    document.querySelectorAll('.course-list-container').forEach(container => {
      container.querySelectorAll('.university-entry').forEach(item => {
        item.style.display = 'none';
      });
      updateMessages(container, false, 0);
    });
    return;
  }

  const container = inputEl.closest('.course-list-container');
  if (!container) return;

  const filter = (inputEl.value || '').toLowerCase().trim();
  const items = container.querySelectorAll('.university-entry');

  if (!filter) {
    items.forEach(item => (item.style.display = 'none'));
    container.querySelectorAll('.teach-chip').forEach(c => c.classList.remove('is-active'));
    updateMessages(container, false, 0);
    return;
  }

  let matchCount = 0;
  items.forEach(item => {
    const filterText = (item.getAttribute('data-filter-text') || '').toLowerCase();
    const show = filterText.includes(filter);
    item.style.display = show ? '' : 'none';
    if (show) matchCount++;
  });
  updateMessages(container, true, matchCount);
}
