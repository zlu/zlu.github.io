---
title: SQL
synonyms:
  - Structured Query Language
related:
  - Python
  - Java
  - Ruby on Rails
  - Django
---
**SQL** is the standard language for relational databases: declare tables, insert rows, and query with `SELECT` / joins / aggregates. Dialects differ (PostgreSQL, MySQL, SQLite, SQL Server) but the core grammar is shared.

```sql
CREATE TABLE users (
  id   SERIAL PRIMARY KEY,
  name TEXT NOT NULL
);

INSERT INTO users (name) VALUES ('world');

SELECT name FROM users WHERE id = 1;
```
