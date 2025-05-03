document.addEventListener('DOMContentLoaded', function() {
    // Tab switching functionality
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            // Remove active class from all buttons and contents
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            // Add active class to clicked button and corresponding content
            btn.classList.add('active');
            const tabId = btn.getAttribute('data-tab');
            document.getElementById(tabId).classList.add('active');
        });
    });

    // Language switching for tabs
    function updateTabLanguage(lang) {
        tabBtns.forEach(btn => {
            const text = btn.getAttribute(`data-${lang}`);
            if (text) btn.textContent = text;
        });
    }

    // Listen for language change events
    document.addEventListener('languageChanged', function(e) {
        updateTabLanguage(e.detail.language);
    });

    // Initial language setup
    const currentLang = document.documentElement.lang.toLowerCase();
    updateTabLanguage(currentLang === 'zh-cn' ? 'cn' : 'en');
});
