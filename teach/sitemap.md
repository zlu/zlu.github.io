---
layout: default
title: "辅导站点地图 | 地区 · 大学 · 服务"
title_en: "Teach sitemap | Regions, universities, services"
title_cn: "辅导站点地图 | 地区 · 大学 · 服务"
title_suffix: "zlu.me/teach"
description: "留学生计算机辅导站点地图：按地区和大学查找课号页，也可进入同步跟课、作业、考前突击与课程预习等服务入口。中英双语一对一，微信直约。"
keywords: 辅导站点地图, 留学生辅导目录, 大学计算机辅导, 课号辅导
permalink: /teach/sitemap/
lang: zh-CN
sitemap: true
---

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "BreadcrumbList",
  "itemListElement": [
    {
      "@type": "ListItem",
      "position": 1,
      "name": "留学辅导",
      "item": "https://zlu.me/teach/"
    },
    {
      "@type": "ListItem",
      "position": 2,
      "name": "辅导站点地图"
    }
  ]
}
</script>

<article class="teach-landing teach-html-sitemap">
  <div class="teach-topbar">
    <nav class="teach-crumbs" aria-label="Breadcrumb">
      <a href="/teach/" data-en="Tutoring" data-cn="留学辅导">留学辅导</a>
      <span>/</span>
      <span data-en="Sitemap" data-cn="站点地图">站点地图</span>
    </nav>
    {% include teach-lang-switcher.html %}
  </div>

  <header class="teach-landing-header">
    <h1 data-en="Teach sitemap"
        data-cn="辅导站点地图">辅导站点地图</h1>
    <p class="teach-landing-lead"
       data-en="Jump to region hubs, university pages with course codes, and service landings. Only indexable hubs are listed."
       data-cn="按地区进入大学页，再点进具体课号；也可直接看同步跟课、作业、考前与预习等服务入口。这里只列出可收录的辅导页，方便人和搜索引擎一起找到。">按地区进入大学页，再点进具体课号；也可直接看同步跟课、作业、考前与预习等服务入口。这里只列出可收录的辅导页，方便人和搜索引擎一起找到。</p>
  </header>

  <section class="teach-landing-section">
    <h2 data-en="Core pages"
        data-cn="核心入口">核心入口</h2>
    <ul class="teach-course-links">
      <li><a href="/teach/" data-en="Tutoring home" data-cn="辅导首页">辅导首页</a></li>
      <li><a href="/teach/universities/" data-en="University directory" data-cn="大学目录">大学目录</a></li>
      <li><a href="/teach/faq/" data-en="FAQ" data-cn="常见问题">常见问题</a></li>
    </ul>
  </section>

  <section class="teach-landing-section">
    <h2 data-en="Service hubs"
        data-cn="服务入口">服务入口</h2>
    <ul class="teach-course-links">
      <li><a href="/teach/services/sync/">同步课程辅导</a></li>
      <li><a href="/teach/services/assignment/">作业与 Project 辅导</a></li>
      <li><a href="/teach/services/exam/">考前突击辅导</a></li>
      <li><a href="/teach/services/preview/">课程预习辅导</a></li>
    </ul>
  </section>

  <section class="teach-landing-section">
    <h2 data-en="Regions"
        data-cn="按地区">按地区</h2>
    <ul class="teach-course-links teach-course-links--grid">
      {% assign region_pages = site.pages | where_exp: "p", "p.layout == 'teach_region'" %}
      {% assign region_pages = region_pages | sort: "region_cn" %}
      {% for rp in region_pages %}
      <li>
        <a href="{{ rp.url }}">
          <strong>{{ rp.region_cn }}</strong>
          <span class="teach-course-title">{{ rp.region_en }}</span>
        </a>
      </li>
      {% endfor %}
    </ul>
  </section>

  <section class="teach-landing-section">
    <h2 data-en="Universities with course pages"
        data-cn="已有课号页的大学">已有课号页的大学</h2>
    <p class="teach-section-note"
       data-en="Hubs that are in the XML sitemap (at least one linked course)."
       data-cn="仅列出已进入 XML 站点地图的大学页（至少有一门可链课程）。目录没有的课号，仍可微信直接问。">仅列出已进入 XML 站点地图的大学页（至少有一门可链课程）。目录没有的课号，仍可微信直接问。</p>
    <ul class="teach-course-links teach-course-links--grid">
      {% assign hub_unis = site.data.teach_universities | default: empty %}
      {% for uni in hub_unis %}
        {% if uni.courses and uni.courses.size > 0 %}
        <li>
          <a href="/teach/universities/{{ uni.slug }}/">
            <strong data-en="{{ uni.label_en | default: uni.name }}"
                    data-cn="{{ uni.label_cn | default: uni.name }}">{{ uni.label_cn | default: uni.name }}</strong>
            <span class="teach-course-title">{{ uni.courses.size }} 门课</span>
          </a>
        </li>
        {% endif %}
      {% endfor %}
    </ul>
  </section>

  <p class="teach-landing-back">
    <a href="/teach/" data-en="← Tutoring home" data-cn="← 辅导首页">← 辅导首页</a>
  </p>
</article>
