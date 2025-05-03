document.addEventListener('DOMContentLoaded', function() {
    // Initialize tabs
    function initializeTabs() {
        // Only initialize course-tabs inside the currently active language container
        document.querySelectorAll('.lang-en.active .course-tabs, .lang-cn.active .course-tabs').forEach(tabContainer => {
            let tabBtns = tabContainer.querySelectorAll('.tab-btn');
            const tabContents = tabContainer.querySelectorAll('.tab-content');

            // Remove all active classes first
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            // Show first tab by default
            if (tabBtns.length > 0 && tabContents.length > 0) {
                tabBtns[0].classList.add('active');
                const firstTabId = tabBtns[0].getAttribute('data-tab');
                const firstTab = document.getElementById(firstTabId);
                if (firstTab) firstTab.classList.add('active');
            }

            // Remove previous click handlers by cloning (to avoid stacking)
            tabBtns.forEach(btn => {
                const newBtn = btn.cloneNode(true);
                btn.parentNode.replaceChild(newBtn, btn);
            });

            // Re-select buttons after cloning
            tabBtns = tabContainer.querySelectorAll('.tab-btn');

            // Add click handlers
            tabBtns.forEach(btn => {
                btn.addEventListener('click', () => {
                    tabBtns.forEach(b => b.classList.remove('active'));
                    tabContents.forEach(c => c.classList.remove('active'));

                    btn.classList.add('active');
                    const tabId = btn.getAttribute('data-tab');
                    const tabContent = document.getElementById(tabId);
                    if (tabContent) tabContent.classList.add('active');
                });
            });
        });
    }

    // Update tab language
    function updateTabLanguage(lang) {
        const tabBtns = document.querySelectorAll('.tab-btn');
        tabBtns.forEach(btn => {
            const text = btn.getAttribute(`data-${lang}`);
            if (text) btn.textContent = text;
        });
    }

    // Listen for language change events
    document.addEventListener('languageChanged', function(e) {
        updateTabLanguage(e.detail.language);
        initializeTabs();
    });

    // Initialize everything
    initializeTabs();
    const currentLang = document.documentElement.lang.toLowerCase();
    updateTabLanguage(currentLang === 'zh-cn' ? 'cn' : 'en');
});

