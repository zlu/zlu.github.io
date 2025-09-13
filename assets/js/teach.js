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
        // Ensure language code is consistent ('en' or 'cn')
        const langCode = (e && e.detail && e.detail.language === 'cn') ? 'cn' : 'en';
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

    // 3. Wire up course search inputs and hide all courses initially
    document.querySelectorAll('.course-search-input, #courseSearchInput').forEach(input => {
        if (!input.hasAttribute('data-course-listener')) {
            const handler = function() { filterCourses(this); };
            input.addEventListener('input', handler);
            input.addEventListener('keyup', handler);
            input.setAttribute('data-course-listener', 'true');
        }
    });

    // Hide all until user starts typing in each list
    filterCourses();

    // Note: Ensure your main language switching logic correctly handles the
    // visibility (e.g., display: none/block) of the '.lang-en' and '.lang-cn' divs.
    // This script now relies on that mechanism to show the correct set of tabs.
});


// Function to filter the course list based on search input
function filterCourses(inputEl) {
  // Utility to toggle hint/no-results within a container
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

  // If called without an input element, hide all items and show hints
  if (!inputEl) {
    document.querySelectorAll('.course-list-container').forEach(container => {
      container.querySelectorAll('.university-entry').forEach(item => {
        item.style.display = 'none';
      });
      updateMessages(container, false, 0);
    });
    return;
  }

  // Scope filtering to the nearest course list container
  const container = inputEl.closest('.course-list-container');
  if (!container) return;

  const filter = (inputEl.value || '').toLowerCase();
  const items = container.querySelectorAll('.university-entry');

  // If there's no search query, hide all items and show hint
  if (!filter) {
    items.forEach(item => (item.style.display = 'none'));
    updateMessages(container, false, 0);
    return;
  }

  // Filter items in this container only and count matches
  let matchCount = 0;
  items.forEach(item => {
    const filterText = (item.getAttribute('data-filter-text') || '').toLowerCase();
    const show = filterText.includes(filter);
    item.style.display = show ? '' : 'none';
    if (show) matchCount++;
  });
  updateMessages(container, true, matchCount);
}
