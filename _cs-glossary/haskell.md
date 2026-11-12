---
title: Haskell
synonyms: []
related:
  - Smalltalk
  - Rust
  - Python
  - JavaScript
---
**Haskell** is a purely functional, statically typed, lazy language. Fran (Elliott and Hudak) implemented early FRP in Haskell. Common in compilers, formal tooling, and teaching type systems.

```haskell
greet :: String -> String
greet name = "hello, " ++ name

main :: IO ()
main = putStrLn (greet "world")
```
