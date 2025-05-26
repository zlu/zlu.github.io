// Wait for Decap CMS to be ready before registering preview templates
(function() {
  function registerMathJaxPreview() {
    // Helper to inject MathJax into the preview iframe
    function ensureMathJaxInIframe(iframeDoc) {
      if (!iframeDoc.getElementById('mathjax-script')) {
        const script = iframeDoc.createElement('script');
        script.id = 'mathjax-script';
        script.type = 'text/javascript';
        script.src = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js';
        script.async = true;
        iframeDoc.head.appendChild(script);
      }
    }

    // Helper to typeset math in the preview iframe
    function typesetMathInIframe(iframeWin) {
      if (iframeWin.MathJax && iframeWin.MathJax.typesetPromise) {
        iframeWin.MathJax.typesetPromise();
      }
    }

    // Custom preview component for markdown with MathJax
    const MathJaxPreview = ({ entry, widgetFor }) => {
      // Use widgetFor('body') for markdown fields
      const body = widgetFor && widgetFor('body') ? widgetFor('body') : '';
      // After rendering, inject MathJax and typeset
      setTimeout(() => {
        // Find the preview iframe
        const iframe = document.querySelector('.nc-previewPane-iframe, .cms-previewPane iframe');
        if (iframe && iframe.contentDocument) {
          ensureMathJaxInIframe(iframe.contentDocument);
          // Wait a bit for MathJax to load, then typeset
          setTimeout(() => typesetMathInIframe(iframe.contentWindow), 300);
        }
      }, 0);
      return body;
    };

    // Register for each collection with markdown
    window.CMS.registerPreviewTemplate('blog', MathJaxPreview);
    window.CMS.registerPreviewTemplate('drafts', MathJaxPreview);
    window.CMS.registerPreviewTemplate('ai-glossary', MathJaxPreview);
  }

  // Wait for CMS to be available
  function waitForCMS() {
    if (window.CMS && typeof window.CMS.registerPreviewTemplate === 'function') {
      registerMathJaxPreview();
    } else {
      setTimeout(waitForCMS, 100);
    }
  }
  waitForCMS();
})(); 