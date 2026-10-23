/**
 * Hover / focus popovers for in-post links to /ai-glossary/… and /med-glossary/….
 * Definitions come from /assets/data/{ai,med}-glossary.json (built by Jekyll).
 */
(function () {
  const LINK_RE = /^\/(ai-glossary|med-glossary)\/([^/]+)\/?$/;
  const cache = { "ai-glossary": null, "med-glossary": null };
  const loading = { "ai-glossary": null, "med-glossary": null };

  let tip = null;
  let activeLink = null;
  let hideTimer = null;
  let showTimer = null;

  function ensureTip() {
    if (tip) return tip;
    tip = document.createElement("div");
    tip.className = "glossary-tooltip";
    tip.setAttribute("role", "tooltip");
    tip.hidden = true;
    tip.addEventListener("mouseenter", () => clearTimeout(hideTimer));
    tip.addEventListener("mouseleave", scheduleHide);
    document.body.appendChild(tip);
    return tip;
  }

  function loadGlossary(kind) {
    if (cache[kind]) return Promise.resolve(cache[kind]);
    if (loading[kind]) return loading[kind];
    loading[kind] = fetch("/assets/data/" + kind + ".json")
      .then((r) => (r.ok ? r.json() : {}))
      .then((data) => {
        cache[kind] = data || {};
        return cache[kind];
      })
      .catch(() => {
        cache[kind] = {};
        return cache[kind];
      });
    return loading[kind];
  }

  function parseGlossaryHref(href) {
    if (!href) return null;
    try {
      const u = new URL(href, window.location.origin);
      if (u.origin !== window.location.origin) return null;
      const m = u.pathname.match(LINK_RE);
      if (!m) return null;
      return { kind: m[1], slug: decodeURIComponent(m[2]) };
    } catch (_) {
      return null;
    }
  }

  function positionTip(anchor) {
    const el = ensureTip();
    const rect = anchor.getBoundingClientRect();
    const margin = 8;
    const tipRect = el.getBoundingClientRect();
    let top = rect.bottom + margin + window.scrollY;
    let left = rect.left + window.scrollX;

    if (left + tipRect.width > window.scrollX + window.innerWidth - 12) {
      left = window.scrollX + window.innerWidth - tipRect.width - 12;
    }
    if (left < window.scrollX + 12) left = window.scrollX + 12;

    const spaceBelow = window.innerHeight - rect.bottom;
    if (spaceBelow < tipRect.height + margin && rect.top > tipRect.height + margin) {
      top = rect.top + window.scrollY - tipRect.height - margin;
      el.classList.add("glossary-tooltip--above");
    } else {
      el.classList.remove("glossary-tooltip--above");
    }

    el.style.top = Math.round(top) + "px";
    el.style.left = Math.round(left) + "px";
  }

  function renderTip(entry) {
    const el = ensureTip();
    el.innerHTML =
      '<div class="glossary-tooltip-title"></div>' +
      '<div class="glossary-tooltip-body"></div>' +
      '<a class="glossary-tooltip-more" href="#"></a>';
    el.querySelector(".glossary-tooltip-title").textContent = entry.title;
    el.querySelector(".glossary-tooltip-body").textContent = entry.excerpt;
    const more = el.querySelector(".glossary-tooltip-more");
    more.href = entry.url;
    more.textContent = "Full definition →";
    el.hidden = false;
  }

  function hideTip() {
    clearTimeout(showTimer);
    clearTimeout(hideTimer);
    if (tip) tip.hidden = true;
    if (activeLink) activeLink.classList.remove("glossary-link--active");
    activeLink = null;
  }

  function scheduleHide() {
    clearTimeout(hideTimer);
    hideTimer = setTimeout(hideTip, 180);
  }

  function showForLink(link) {
    const parsed = parseGlossaryHref(link.getAttribute("href"));
    if (!parsed) return;

    clearTimeout(hideTimer);
    clearTimeout(showTimer);
    showTimer = setTimeout(() => {
      loadGlossary(parsed.kind).then((map) => {
        const entry = map[parsed.slug];
        if (!entry) return;
        if (activeLink) activeLink.classList.remove("glossary-link--active");
        activeLink = link;
        link.classList.add("glossary-link--active");
        renderTip(entry);
        positionTip(link);
        // Reposition after layout with real size
        requestAnimationFrame(() => positionTip(link));
      });
    }, 120);
  }

  function markGlossaryLinks(root) {
    const scope = root || document;
    scope.querySelectorAll('a[href*="/ai-glossary/"], a[href*="/med-glossary/"]').forEach((a) => {
      const parsed = parseGlossaryHref(a.getAttribute("href"));
      if (!parsed) return;
      if (a.dataset.glossaryBound) return;
      a.dataset.glossaryBound = "1";
      a.classList.add("glossary-link");
      a.setAttribute("aria-describedby", "glossary-tooltip-live");

      a.addEventListener("mouseenter", () => showForLink(a));
      a.addEventListener("mouseleave", scheduleHide);
      a.addEventListener("focus", () => showForLink(a));
      a.addEventListener("blur", scheduleHide);

      // Touch: first tap peeks; second tap (or Full definition) navigates.
      a.addEventListener(
        "click",
        (ev) => {
          if (!window.matchMedia("(hover: hover) and (pointer: fine)").matches) {
            if (activeLink !== a || !tip || tip.hidden) {
              ev.preventDefault();
              showForLink(a);
            }
          }
        },
        { passive: false }
      );
    });
  }

  document.addEventListener("DOMContentLoaded", () => {
    ensureTip().id = "glossary-tooltip-live";
    markGlossaryLinks(document.querySelector("article.post") || document.querySelector("main") || document);
    document.addEventListener("keydown", (ev) => {
      if (ev.key === "Escape") hideTip();
    });
    window.addEventListener(
      "scroll",
      () => {
        if (activeLink && tip && !tip.hidden) positionTip(activeLink);
      },
      { passive: true }
    );
  });
})();
