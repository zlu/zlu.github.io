---
title: Flask
synonyms: []
related:
  - Python
  - Django
  - Express
  - FastAPI
---
**Flask** is a lightweight [Python](/cs-glossary/python/) web microframework. You wire routes and extensions explicitly; less built-in than [Django](/cs-glossary/django/).

```python
from flask import Flask

app = Flask(__name__)

@app.get("/hello/<name>")
def hello(name: str):
    return f"hello, {name}"
```
