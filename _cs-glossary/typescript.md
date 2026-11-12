---
title: TypeScript
synonyms:
  - TS
related:
  - JavaScript
  - React
  - Node.js
  - Next.js
---
**TypeScript** is [JavaScript](/cs-glossary/javascript/) plus an optional static type system. It compiles (erases) to JS. Used heavily with [React](/cs-glossary/react/), [Node.js](/cs-glossary/nodejs/), and [Next.js](/cs-glossary/nextjs/).

```typescript
function greet(name: string): string {
  return `hello, ${name}`;
}

const n: number = greet("world").length;
```
