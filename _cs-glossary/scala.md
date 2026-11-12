---
title: Scala
synonyms: []
related:
  - Java
  - Kotlin
  - Haskell
  - Python
---
**Scala** is a statically typed JVM language that mixes OOP and functional features. Used in data infrastructure (e.g. Apache Spark) and services that want [Java](/cs-glossary/java/) interop with stronger FP tools.

```scala
def greet(name: String): String = s"hello, $name"

@main def run(): Unit =
  println(greet("world"))
```
