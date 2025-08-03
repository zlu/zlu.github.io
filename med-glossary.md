---
layout: page
title: Medicine Glossary
permalink: /med-glossary/
---

<div id="glossary-search">
  <input type="text" id="glossary-search-input" placeholder="Search terms...">
</div>

<ul id="glossary-list">
  {% for term in site.med-glossary %}
    <li class="glossary-item">
      <span class="glossary-term" style="cursor:pointer; color:blue; text-decoration:underline;">
        {{ term.title }}
      </span>
      <div class="glossary-definition" style="display:none; margin:0.5em 0 1em 1em;">
        {{ term.content }}
      </div>
    </li>
  {% endfor %}
</ul>

<script>
document.addEventListener('DOMContentLoaded', function() {
  // Search functionality
  const input = document.getElementById('glossary-search-input');
  const list = document.getElementById('glossary-list');
  const items = Array.from(list.getElementsByTagName('li'));

  input.addEventListener('input', function() {
    const query = input.value.toLowerCase();
    items.forEach(item => {
      const text = item.textContent.toLowerCase();
      item.style.display = text.includes(query) ? '' : 'none';
    });
  });

  // Inline definition toggle
  document.querySelectorAll('.glossary-term').forEach(function(term) {
    term.addEventListener('click', function() {
      const def = this.nextElementSibling;
      def.style.display = (def.style.display === 'none' || def.style.display === '') ? 'block' : 'none';
      if (def.style.display === 'block' && window.MathJax && window.MathJax.typesetPromise) {
        MathJax.typesetPromise([def]);
      }
    });
  });
});
</script> 