---
title: Ruby on Rails
synonyms:
  - Rails
  - RoR
  - ROR
related:
  - Ruby
  - Django
  - Laravel
  - SQL
---
**Ruby on Rails** (Rails, RoR) is a full-stack web framework for [Ruby](/cs-glossary/ruby/). Convention over configuration, Active Record ORM, MVC routing, and generators for CRUD apps.

```ruby
# config/routes.rb
resources :posts

# app/controllers/posts_controller.rb
class PostsController < ApplicationController
  def show
    @post = Post.find(params[:id])
  end
end

# app/models/post.rb
class Post < ApplicationRecord
  validates :title, presence: true
end
```
