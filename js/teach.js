document.addEventListener('DOMContentLoaded', function() {
    // Initialize tabs for ALL language containers
    function initializeTabs() {
        // Select tab containers within both language divs, regardless of 'active' class
        document.querySelectorAll('.lang-en .course-tabs, .lang-cn .course-tabs').forEach(tabContainer => {
            const tabBtns = tabContainer.querySelectorAll('.tab-btn');
            // Query for tab content potentially within the container or associated with it
            // Adjust this selector if your content structure is different
            const tabContents = tabContainer.querySelectorAll('.tab-content');

            // --- Start: Logic within each tabContainer ---
            // Remove existing active classes within this specific container
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            // Show first tab by default within this container
            if (tabBtns.length > 0) {
                tabBtns[0].classList.add('active');
                const firstTabId = tabBtns[0].getAttribute('data-tab');
                // Find the content associated *with this tab container*
                // Use querySelector on the container first for better scoping
                let firstTabContent = tabContainer.querySelector(`.tab-content#${firstTabId}`);
                 if (!firstTabContent) {
                    // Fallback: Search globally if not found within container (less ideal, depends on unique IDs)
                    firstTabContent = document.getElementById(firstTabId);
                 }
                 if (firstTabContent) {
                    firstTabContent.classList.add('active');
                 }
            }

            // Add click handlers (only once per button)
            tabBtns.forEach(btn => {
                // Check if listener already exists to prevent duplicates
                if (!btn.hasAttribute('data-tab-listener-added')) {
                    btn.addEventListener('click', () => {
                        // Find the closest tab container for context
                        const currentTabContainer = btn.closest('.course-tabs');
                        if (!currentTabContainer) return; // Should not happen

                        const siblingBtns = currentTabContainer.querySelectorAll('.tab-btn');
                        // Query contents relative to the container or globally if needed
                        const contentContainer = currentTabContainer.querySelector('.tab-content-container') || currentTabContainer; // Adjust if structure differs

                        // Deactivate all tabs and content within this container/scope
                        siblingBtns.forEach(b => b.classList.remove('active'));
                        const siblingContents = contentContainer.querySelectorAll('.tab-content'); // Find contents again relative to container
                        siblingContents.forEach(c => c.classList.remove('active'));


                        // Activate the clicked tab
                        btn.classList.add('active');
                        const tabId = btn.getAttribute('data-tab');

                        // Find and activate the corresponding content
                        let tabContent = contentContainer.querySelector(`.tab-content#${tabId}`);
                        if (!tabContent) {
                           // Fallback: Search globally
                           tabContent = document.getElementById(tabId);
                        }
                        if (tabContent) {
                           tabContent.classList.add('active');
                        }
                    });
                    // Mark the button so we don't add the listener again
                    btn.setAttribute('data-tab-listener-added', 'true');
                }
            });
             // --- End: Logic within each tabContainer ---
        });
    }

    // Update tab button text based on language
    function updateTabLanguage(lang) {
        // Select buttons within *all* language containers
        document.querySelectorAll('.lang-en .tab-btn, .lang-cn .tab-btn').forEach(btn => {
            const text = btn.getAttribute(`data-${lang}`);
            if (text) btn.textContent = text;
        });
    }

    // Listen for language change events - ONLY update text
    document.addEventListener('languageChanged', function(e) {
        // Ensure language code is consistent (e.g., 'en' or 'cn')
        const langCode = e.detail.language === 'zh-cn' ? 'cn' : 'en';
        updateTabLanguage(langCode);
        // DO NOT re-initialize tabs here. Visibility is handled by the main language switcher.
    });

    // --- Initial Setup ---
    // 1. Initialize tab functionality (event listeners, default active state)
    initializeTabs();

    // 2. Set initial tab text based on current language (assuming html lang attribute is set)
    const currentHtmlLang = document.documentElement.lang.toLowerCase();
    const initialLangCode = currentHtmlLang === 'zh-cn' ? 'cn' : 'en';
    updateTabLanguage(initialLangCode);

    // Note: Ensure your main language switching logic correctly handles the
    // visibility (e.g., display: none/block) of the '.lang-en' and '.lang-cn' divs.
    // This script now relies on that mechanism to show the correct set of tabs.
});

