---
title: React
synonyms:
  - React.js
  - ReactJS
related:
  - JavaScript
  - TypeScript
  - Next.js
  - Node.js
---
**React** is a [JavaScript](/cs-glossary/javascript/) / [TypeScript](/cs-glossary/typescript/) UI library for composing components. State updates schedule re-renders of components that read that state. Hooks such as `useState` and `useEffect` cover local state and subscribe / cleanup. [Next.js](/cs-glossary/nextjs/) is a common full-stack React framework.

```jsx
import { useState } from "react";

function ClickCounter() {
  const [count, setCount] = useState(0);
  return (
    <button onClick={() => setCount((c) => c + 1)}>
      {count}
    </button>
  );
}
```
