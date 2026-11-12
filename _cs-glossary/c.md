---
title: C
synonyms:
  - C language
related:
  - C++
  - Rust
  - Go
  - Java
---
**C** is a statically typed systems language with manual memory management, pointers, and a thin abstraction over the machine. Still the lingua franca for OS kernels, embedded code, and language runtimes.

```c
#include <stdio.h>

const char *greet(const char *name) {
  static char buf[64];
  snprintf(buf, sizeof buf, "hello, %s", name);
  return buf;
}

int main(void) {
  puts(greet("world"));
  return 0;
}
```
