---
title: Node.js
synonyms:
  - Node
  - NodeJS
related:
  - JavaScript
  - TypeScript
  - Express
  - Next.js
---
**Node.js** is a [JavaScript](/cs-glossary/javascript/) runtime built on V8, outside the browser. Event-loop I/O, npm packages, and common use for APIs ([Express](/cs-glossary/express/)) and tooling ([Next.js](/cs-glossary/nextjs/), bundlers, CLIs).

```javascript
import http from "node:http";

const server = http.createServer((req, res) => {
  res.writeHead(200, { "content-type": "text/plain" });
  res.end("hello, world");
});

server.listen(3000);
```
