---
title: Express
synonyms:
  - Express.js
  - ExpressJS
related:
  - Node.js
  - JavaScript
  - TypeScript
  - Next.js
---
**Express** is a minimal HTTP framework for [Node.js](/cs-glossary/nodejs/). Middleware chain, routing, and JSON APIs. Many higher-level Node frameworks sit on top of the same request / response model.

```javascript
import express from "express";

const app = express();

app.get("/hello", (req, res) => {
  res.json({ message: `hello, ${req.query.name ?? "world"}` });
});

app.listen(3000);
```
