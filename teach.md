---
layout: default
title: "Computer Science Tutoring for Chinese Students — AI & Data Science Expert"
title_en: "Computer Science Tutoring for Chinese Students | AI & Data Science Expert"
title_cn: "留学生计算机一对一辅导 | 人工智能与数据科学"
title_suffix: "Professional CS Tutor for International Students | zlu.me/teach"
description: Direct 1-1 CS tutoring for Chinese students abroad — no middleman fee. Bilingual help with lectures, homework, exams, and project write-ups. Silicon Valley engineer, AI & data science.
keywords: computer science tutor, no middleman, 无中介, 留学生辅导, 作业辅导, 考前突击, AI tutoring, 中英双语, WeChat tutoring
lang: zh-CN
alternate_lang: en
hreflang:
  en: https://zlu.me/teach/?lang=en
  zh-CN: https://zlu.me/teach/
permalink: /teach/
region_pages: []
---

{% include tutoring-schema.html %}
{% include teach-hero.html %}
{% include teach-scenes.html %}
{% include teach-needs.html %}
{% include teach-method.html %}

<section class="teach-directory" aria-labelledby="uni-dir-title">
  <h2 id="uni-dir-title"
      data-en="University courses"
      data-cn="大学课程目录">大学课程目录</h2>
  <p class="teach-directory-lead"
     data-en="Search detailed course pages below, or open the full <a href='/teach/universities/'>1000+ university directory</a> (AU/UK/US/CA/HK/SG). Message WeChat with school + course code."
     data-cn="下方可搜已整理课号；完整 <a href='/teach/universities/'>1000+ 所大学目录</a>（澳英美加港新）按地区浏览。微信发送学校 + 课号即可约课。">下方可搜已整理课号；完整 <a href="/teach/universities/">1000+ 所大学目录</a>（澳英美加港新）按地区浏览。微信发送学校 + 课号即可约课。</p>
  <div class="teach-region-links teach-region-links-inline">
    <a href="/teach/services/sync/">同步辅导</a>
    <a href="/teach/services/assignment/">作业辅导</a>
    <a href="/teach/services/exam/">考前突击</a>
    <a href="/teach/services/preview/">课程预习</a>
  </div>
  <div class="teach-region-links teach-region-links-inline">
    <a href="/teach/australia/">澳大利亚</a>
    <a href="/teach/uk/">英国</a>
    <a href="/teach/usa/">美国</a>
    <a href="/teach/hong-kong/">香港</a>
    <a href="/teach/singapore/">新加坡</a>
    <a href="/teach/canada/">加拿大</a>
  </div>
  {% include teach-course-list.html %}
</section>

{% capture en_content %}
## Proof, not a sales script {#why-me}

- **Direct booking** — WeChat me; consult and learn with the same person
- **No middleman fee** — your tuition goes to the person teaching
- **Silicon Valley engineer** — 15+ years, five patents, bilingual CS/AI/data science
- **Integrity** — explain, debug, structure reports; never ghostwrite or sit exams

### How a session runs {#how-it-works}

1. Send school, course code, deadline, and files (syllabus / errors / draft).
2. Class on **ClassIn** or **Tencent Meeting** (install first). Also available: Zoom, Microsoft Teams, Slack, Google Meet, WeChat video.
3. Live coding in your environment; bilingual explanation.
4. You leave with next steps you can execute alone.

### Course categories {#course-categories}

{% include teach-course-tabs.html %}

### Student testimonials {#student-testimonials}

{% include teach-student-testimonies.html %}

### Frequently asked questions {#faqs}

{% include teach-faq-en.md %}

{% endcapture %}

{% capture cn_content %}
## 实力证明，不是话术 {#为什么找我}

- **直接约课** —— 微信找我；咨询和上课是同一个人
- **无中介抽成** —— 学费付给真正上课的人
- **硅谷工程师** —— 15+ 年、五项专利、中英双语 CS / AI / 数据科学
- **学术诚信** —— 讲思路、带调试、梳报告结构；**不代写、不代考**

### 一节课怎么进行 {#上课方式}

1. 发送学校、课号、截止日与资料（大纲 / 报错 / 草稿）。
2. 主用 **ClassIn** 或 **腾讯会议**（请提前下载）。也可用 Zoom、Microsoft Teams、Slack、Google Meet、微信视频。
3. 在你的环境里直播写代码；中英双语讲解。
4. 课后给你可独立推进的下一步。

### 课程分类 {#课程分类}

{% include teach-course-tabs.html %}

### 学生评价 {#学生评价或成功案例}

{% include teach-student-testimonies.html %}

### 常见问题 {#常见问题解答-faq}

{% include teach-faq-cn.md %}

{% endcapture %}

<div class="lang-cn" id="cn-content">{{ cn_content | markdownify }}</div>
<div class="lang-en" id="en-content">{{ en_content | markdownify }}</div>

{% include teach-sticky-cta.html %}
