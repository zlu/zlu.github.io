// Wait for Decap CMS to be ready before registering preview templates
(function() {
  function normCourseKey(text) {
    return String(text || "")
      .replace(/\s+/g, "")
      .toUpperCase();
  }

  function courseKeys(course) {
    if (!course || typeof course.get !== "function") return [];
    const keys = [];
    const code = normCourseKey(course.get("code"));
    if (code) keys.push(code);

    const aliases = course.get("aliases");
    if (aliases && typeof aliases.forEach === "function") {
      aliases.forEach(function (alias) {
        const key = normCourseKey(
          alias && typeof alias.get === "function"
            ? alias.get("alias") || alias.get("code") || alias.get("name")
            : alias
        );
        if (key) keys.push(key);
      });
    }
    return keys;
  }

  function findDuplicateCourseKeys(courses) {
    const owners = Object.create(null);
    const duplicates = [];

    courses.forEach(function (course, index) {
      courseKeys(course).forEach(function (key) {
        if (owners[key] === undefined) {
          owners[key] = index;
          return;
        }
        if (duplicates.indexOf(key) === -1) duplicates.push(key);
      });
    });

    return duplicates;
  }

  function sortCourses(courses) {
    return courses.sort(function (a, b) {
      const codeA = normCourseKey(a && a.get ? a.get("code") : "");
      const codeB = normCourseKey(b && b.get ? b.get("code") : "");
      if (!codeA && !codeB) {
        const titleA = String((a && a.get && a.get("title")) || "");
        const titleB = String((b && b.get && b.get("title")) || "");
        return titleA.localeCompare(titleB);
      }
      if (!codeA) return 1;
      if (!codeB) return -1;
      const byCode = codeA.localeCompare(codeB, undefined, { numeric: true });
      if (byCode !== 0) return byCode;
      const titleA = String((a && a.get && a.get("title")) || "");
      const titleB = String((b && b.get && b.get("title")) || "");
      return titleA.localeCompare(titleB);
    });
  }

  function isCoursesEntry(entry) {
    const collection = entry.get("collection");
    if (collection === "courses") return true;
    const data = entry.get("data");
    return !!(
      data &&
      typeof data.get === "function" &&
      data.get("university") != null &&
      data.get("courses")
    );
  }

  function registerCoursesPreSave() {
    window.CMS.registerEventListener({
      name: "preSave",
      handler: function ({ entry }) {
        const data = entry.get("data");
        if (!isCoursesEntry(entry)) return data;

        const courses = data.get("courses");
        if (!courses || typeof courses.size !== "number" || courses.size === 0) {
          return data;
        }

        const duplicates = findDuplicateCourseKeys(courses);
        if (duplicates.length) {
          throw new Error(
            "Duplicate course code(s): " +
              duplicates.join(", ") +
              ". Each course needs a unique code (aliases must not collide either)."
          );
        }

        return data.set("courses", sortCourses(courses));
      },
    });
  }

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
    window.CMS.registerPreviewTemplate('med-glossary', MathJaxPreview);
  }

  // Wait for CMS to be available
  function waitForCMS() {
    if (window.CMS && typeof window.CMS.registerPreviewTemplate === 'function') {
      registerMathJaxPreview();
      registerCoursesPreSave();
    } else {
      setTimeout(waitForCMS, 100);
    }
  }
  waitForCMS();
})();
