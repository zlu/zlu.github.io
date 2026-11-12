---
title: Vue
synonyms:
  - Vue.js
  - VueJS
related:
  - JavaScript
  - TypeScript
  - React
  - Angular
---
**Vue** is a [JavaScript](/cs-glossary/javascript/) UI framework with a reactive component model and single-file components (template + script + style). Competes with [React](/cs-glossary/react/) and [Angular](/cs-glossary/angular/) for SPA / hybrid apps.

```vue
<script setup>
import { ref } from "vue";
const count = ref(0);
</script>

<template>
  <button @click="count++">{{ count }}</button>
</template>
```
