---
title: Haskell
synonyms: []
related:
  - FRP
  - Smalltalk
  - Rust
  - Python
  - JavaScript
---
**Haskell** is a purely functional, statically typed, lazy language. Fran (Elliott and Hudak) implemented early [FRP](/cs-glossary/frp/) here; Reactive Banana and Yampa are later teaching / research libraries in the same lineage. Common in compilers, formal tooling, and teaching type systems.

```haskell
greet :: String -> String
greet name = "hello, " ++ name

main :: IO ()
main = putStrLn (greet "world")
```
