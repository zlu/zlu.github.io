zlu.github.io
=============

[Live site](https://www.zlu.me)

## China Specificity

## 1. **How It Works**
- By default, load CDN versions (fast for most users).
- Use a GeoIP service (like ipapi.co) to detect if the user is in China.
- If in China, dynamically switch to local versions (which you must host in your `/assets` directory).

---

## 2. **Implementation**

### Download and host local versions of CSS/JS
  - Bootstrap CSS/JS (e.g. `/assets/bootstrap/css/bootstrap.min.css`, `/assets/bootstrap/js/bootstrap.bundle.min.js`)
  - MathJax
  - Any other CDN resources you want to serve locally

### **A. In `<head>`:**
```html
<!-- Bootstrap CSS: CDN by default, local fallback for China -->
<link id="bootstrap-cdn" rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css">
<link id="bootstrap-local" rel="stylesheet" href="/assets/bootstrap/css/bootstrap.min.css" disabled>

<!-- Font Awesome (if used) -->
<link id="fa-cdn" rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@fortawesome/fontawesome-free/css/all.min.css">
<link id="fa-local" rel="stylesheet" href="/assets/fontawesome/css/all.min.css" disabled>

<!-- MathJax (if used) -->
<script id="mathjax-cdn" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<script id="mathjax-local" async src="/assets/js/mathjax/es5/tex-mml-chtml.js" disabled></script>
```

### **B. Before `</body>`:**
```html
<!-- Bootstrap JS: CDN by default, local fallback for China -->
<script id="bootstrap-js-cdn" src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
<script id="bootstrap-js-local" src="/assets/bootstrap/js/bootstrap.bundle.min.js" disabled></script>

<script>
fetch('https://ipapi.co/json/')
  .then(response => response.json())
  .then(data => {
    if (data && data.country_code === 'CN') {
      // Bootstrap CSS
      document.getElementById('bootstrap-cdn').setAttribute('disabled', 'disabled');
      document.getElementById('bootstrap-local').removeAttribute('disabled');
      // Bootstrap JS
      document.getElementById('bootstrap-js-cdn').setAttribute('disabled', 'disabled');
      document.getElementById('bootstrap-js-local').removeAttribute('disabled');
      // Font Awesome
      document.getElementById('fa-cdn').setAttribute('disabled', 'disabled');
      document.getElementById('fa-local').removeAttribute('disabled');
      // MathJax
      document.getElementById('mathjax-cdn').setAttribute('disabled', 'disabled');
      document.getElementById('mathjax-local').removeAttribute('disabled');
      // Optionally, remove Google Analytics/Tag Manager and add Baidu Analytics here
    }
  });
</script>
```

---

## 3. **Updates**
- Place the above `<link>` and `<script>` tags in your layout.
- Download and host the local versions of the libraries in your `/assets` directory.
- Adjust the paths if your local files are in a different location.

---

## 4. **Optional: Baidu Analytics for China**
As Google Analytics does not work for China, use Baidu Analytics script in the same JS block if `country_code === 'CN'`.

---

## 5. **Summary**
- This approach is robust, works for any static site, and is easy to maintain.
- Users in China will get fast, reliable access to your site’s CSS/JS.
- All other users will continue to benefit from CDN speed.

