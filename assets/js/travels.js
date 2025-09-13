// Simple tab switcher and graceful fallback for map iframes
(function(){
  function qs(sel, root){ return (root||document).querySelector(sel); }
  function qsa(sel, root){ return Array.from((root||document).querySelectorAll(sel)); }

  function setActive(el, active){ if (!el) return; el.classList.toggle('active', !!active); }

  // Public API for inline onclick handlers
  window.switchMap = function(kind){
    const kinds = ['google','amap'];
    kinds.forEach(k => {
      setActive(qs(`#${k}-map`), k===kind);
      setActive(qs(`#${k}-instructions`), k===kind);
      // button state
      const btn = qs(`.map-tab[onclick*="${k}"]`);
      setActive(btn, k===kind);
    });
  };

  // Fallback overlays for iframes that fail to load in restrictive browsers
  function attachIframeFallback(wrapperSelector, iframeSelector, linkHref, linkText){
    const wrapper = qs(wrapperSelector);
    const iframe = qs(iframeSelector);
    if (!wrapper || !iframe) return;

    let loaded = false;
    let overlay = null;
    const onLoad = function(){
      loaded = true;
      if (overlay && overlay.parentNode) {
        overlay.parentNode.removeChild(overlay);
        overlay = null;
      }
    };
    iframe.addEventListener('load', onLoad, { once: true });

    // After 6s, if not loaded, show an overlay with a direct link
    setTimeout(function(){
      if (loaded) return;
      overlay = document.createElement('div');
      overlay.style.cssText = 'position:absolute;inset:0;display:flex;align-items:center;justify-content:center;background:#fafafa;border-top:1px solid #eee;color:#6b7280;font-size:14px;padding:12px;text-align:center;';
      const a = document.createElement('a');
      a.href = linkHref;
      a.target = '_blank';
      a.rel = 'noopener';
      a.textContent = linkText || 'Open map in a new tab';
      const msg = document.createElement('div');
      msg.innerHTML = 'Map failed to load in this browser. ';
      msg.appendChild(a);
      overlay.appendChild(msg);
      wrapper.appendChild(overlay);
    }, 6000);
  }

  document.addEventListener('DOMContentLoaded', function(){
    // Ensure the default visible tab is Google
    window.switchMap('google');

    // Add fallback overlays
    attachIframeFallback('#google-map', '#google-map iframe',
      'https://www.google.com/maps/d/viewer?mid=1Ki6wGR5Omnjm2EQ1DdwmGktgpPY-R4E',
      'Open Google My Maps');
    attachIframeFallback('#amap-map', '#amap-map iframe',
      'https://www.amap.com/',
      'Open AMap');
  });
})();
