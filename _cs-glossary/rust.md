---
title: Rust
synonyms: []
related:
  - C
  - C++
  - Go
  - Python
---
**Rust** is a statically typed systems language with ownership and borrowing instead of a tracing GC. Aims for memory safety without data races, at C / C++ performance levels.

```rust
fn greet(name: &str) -> String {
    format!("hello, {name}")
}

fn main() {
    println!("{}", greet("world"));
}
```
