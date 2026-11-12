---
title: Django
synonyms: []
related:
  - Python
  - Ruby on Rails
  - SQL
  - Flask
---
**Django** is a batteries-included web framework for [Python](/cs-glossary/python/). ORM, admin, auth, and URL routing ship in the core project. Follows an MTV (model–template–view) layout close to classic MVC.

```python
# urls.py
from django.urls import path
from . import views

urlpatterns = [
    path("posts/<int:pk>/", views.post_detail),
]

# views.py
from django.shortcuts import get_object_or_404, render
from .models import Post

def post_detail(request, pk):
    post = get_object_or_404(Post, pk=pk)
    return render(request, "post_detail.html", {"post": post})
```
