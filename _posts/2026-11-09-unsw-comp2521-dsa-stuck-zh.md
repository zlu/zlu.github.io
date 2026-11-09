---
layout: post
title: "UNSW COMP2521 为什么特别卡：数据结构作业怎么推进"
date: 2026-11-09
comments: true
tags:
  - 留学辅导
  - UNSW
  - COMP2521
  - 数据结构
  - 算法
lang: zh
description: "UNSW COMP2521（Data Structures and Algorithms）常让留学生卡在实现细节和复杂度表述之间。本文说明常见卡点、作业推进顺序，以及同步跟课和一对一辅导如何配合。"
keywords: UNSW COMP2521, COMP2521 作业, 数据结构 留学生, Data Structures and Algorithms, 新南威尔士大学
teach_course: /teach/courses/unsw/comp2521/
---

[COMP2521](/teach/courses/unsw/comp2521/) 比入门编程更折磨人的地方，在于它同时考三件事：结构选得对不对、实现细不细、你能不能用复杂度语言说清楚取舍。很多同学本地样例能过，一到大数据或边界就崩；或者代码勉强对，报告/口头解释却说不圆。

## 常见卡点

- 链表、树、图的操作顺序在纸上清楚，一写进 C 就被指针和服务函数缠住。
- 知道要用哪种结构，却说不清为什么不用另一种，复杂度一问就空。
- 作业测试专卡空表、重复键、内存释放这类边界，修一个冒出两个。

这些问题很少是「你不够聪明」，更多是缺少可重复的排查顺序。

## 建议的推进顺序

1. 先用纸笔画清状态变化，再碰代码。
2. 为每个操作写最小可复现例子，包括空结构和极端输入。
3. 最后才优化和谈复杂度。顺序反了，容易在错误实现上讨论大 O。

一对一时，我通常会让你在自己的环境里改，我负责指出「现在到底卡在结构、实现还是表述」。需要约课可看：[COMP2521 课程辅导](/teach/courses/unsw/comp2521/)。

前序如果是 [COMP1511](/teach/courses/unsw/comp1511/) 还没稳，也可以先把拆题和调试习惯补上。更多课号见 [UNSW 辅导目录](/teach/universities/unsw/)。
