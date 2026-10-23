(function () {
  var STORAGE_KEY = 'sidebar-collapsed';
  var root = document.documentElement;
  var toggle = document.getElementById('sidebar-toggle');
  var sidebar = document.getElementById('site-sidebar');
  if (!toggle) return;

  var mq = window.matchMedia('(min-width: 769px)');

  function isCollapsed() {
    return root.classList.contains('sidebar-collapsed');
  }

  function syncToggle() {
    var collapsed = isCollapsed();
    toggle.setAttribute('aria-expanded', collapsed ? 'false' : 'true');
    toggle.setAttribute(
      'aria-label',
      collapsed ? 'Expand sidebar' : 'Collapse sidebar'
    );
    toggle.setAttribute(
      'title',
      collapsed ? 'Expand sidebar' : 'Collapse sidebar'
    );
    if (sidebar) {
      if (collapsed && mq.matches) {
        sidebar.setAttribute('aria-hidden', 'true');
        sidebar.setAttribute('inert', '');
      } else {
        sidebar.removeAttribute('aria-hidden');
        sidebar.removeAttribute('inert');
      }
    }
  }

  function setCollapsed(collapsed) {
    root.classList.toggle('sidebar-collapsed', collapsed);
    try {
      localStorage.setItem(STORAGE_KEY, collapsed ? '1' : '0');
    } catch (e) {}
    syncToggle();
  }

  function applyForViewport() {
    if (!mq.matches) {
      root.classList.remove('sidebar-collapsed');
      syncToggle();
      return;
    }
    try {
      setCollapsed(localStorage.getItem(STORAGE_KEY) === '1');
    } catch (e) {
      syncToggle();
    }
  }

  toggle.addEventListener('click', function () {
    if (!mq.matches) return;
    setCollapsed(!isCollapsed());
  });

  if (mq.addEventListener) {
    mq.addEventListener('change', applyForViewport);
  } else if (mq.addListener) {
    mq.addListener(applyForViewport);
  }

  syncToggle();
})();
