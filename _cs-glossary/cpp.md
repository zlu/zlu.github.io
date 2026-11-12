---
title: C++
synonyms:
  - Cpp
  - C plus plus
related:
  - C
  - Rust
  - Java
  - C#
---
**C++** extends [C](/cs-glossary/c/) with classes, templates, RAII, and a large standard library. Used for performance-critical applications, games, and systems that still want zero-cost abstractions.

```cpp
#include <iostream>
#include <string>

std::string greet(const std::string& name) {
  return "hello, " + name;
}

int main() {
  std::cout << greet("world") << "\n";
}
```
