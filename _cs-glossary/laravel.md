---
title: Laravel
synonyms: []
related:
  - PHP
  - Ruby on Rails
  - Django
  - SQL
---
**Laravel** is a full-stack web framework for [PHP](/cs-glossary/php/). Routing, Eloquent ORM, Blade templates, and artisan CLI. Role is similar to [Rails](/cs-glossary/ruby-on-rails/) or [Django](/cs-glossary/django/) in those ecosystems.

```php
// routes/web.php
use App\Models\Post;
use Illuminate\Support\Facades\Route;

Route::get('/posts/{post}', function (Post $post) {
    return view('posts.show', ['post' => $post]);
});
```
