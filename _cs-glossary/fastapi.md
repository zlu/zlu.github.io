---
title: FastAPI
synonyms: []
related:
  - Python
  - Flask
  - Django
  - TypeScript
---
**FastAPI** is a [Python](/cs-glossary/python/) API framework built on type hints and ASGI. Request/response models via annotations; automatic OpenAPI docs. Common for JSON services that want less boilerplate than raw [Flask](/cs-glossary/flask/).

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/hello/{name}")
def hello(name: str) -> dict[str, str]:
    return {"message": f"hello, {name}"}
```
