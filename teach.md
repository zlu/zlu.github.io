---
layout: default
title: "Computer Science Tutoring for Chinese Students — AI & Data Science Expert"
title_en: "Computer Science Tutoring for Chinese Students | AI & Data Science Expert"
title_cn: "留学生计算机一对一辅导 | 人工智能与数据科学"
title_suffix: "Professional CS Tutor for International Students | zlu.me/teach"
description: Expert computer science tutoring for Chinese students in Australia, USA, UK, Canada & NZ. Specialized in AI, Data Science, Python & University coursework. Bilingual instruction (中英双语) available. 15+ years Silicon Valley experience.
keywords: computer science tutor, CS tuition, AI tutoring, data science help, Python programming, Chinese students abroad, 计算机科学辅导, 留学生辅导, 编程家教, 人工智能课程辅导
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

<section class="teach-directory" aria-labelledby="uni-dir-title">
  <h2 id="uni-dir-title"
      data-en="University courses"
      data-cn="大学课程目录">大学课程目录</h2>
  <p class="teach-directory-lead"
     data-en="Search by school or course code. Message on WeChat with your code and deadline."
     data-cn="按学校或课号搜索，微信发我课号与截止日期即可约课。">按学校或课号搜索，微信发我课号与截止日期即可约课。</p>
  {% include teach-course-list.html %}
</section>

{% capture en_content %}
## Why study with me {#why-me}

- **Silicon Valley engineer** with 15+ years across storage, telecom, and product systems
- **Five patents** in storage systems and telephony
- **Bilingual teaching** (中 / EN) for AI, ML, data science, and core CS
- **Integrity first**: explain, debug, and review — never ghostwrite assignments or exams

### How sessions work {#how-it-works}

Send the syllabus, assignment, or error log ahead of time. We meet on ClassIn, Tencent Meeting, Zoom, or WeChat video. Lessons focus on understanding and independent completion — not shortcuts that risk academic integrity.

### Course categories {#course-categories}

{% include teach-course-tabs.html %}

### Student testimonials {#student-testimonials}

{% include teach-student-testimonies.html %}

### Frequently asked questions {#faqs}

{% include teach-faq-en.md %}

{% endcapture %}

{% capture cn_content %}
## 为什么找我 {#为什么找我}

- **硅谷工程师**，软件与系统工程 15+ 年（存储、通信、产品）
- **五项专利**（存储系统与电话通信）
- **中英双语**讲授 AI / 机器学习 / 数据科学 / 编程基础
- **学术诚信**：讲思路、带调试、做复习 —— **不代写、不代考**

### 上课方式 {#上课方式}

课前发送大纲、作业或报错日志。通过 ClassIn、腾讯会议、Zoom 或微信视频上课。目标是你能独立完成，而不是代做导致学术风险。

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
