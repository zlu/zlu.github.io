---
title: Go
synonyms:
  - Golang
related:
  - Rust
  - C
  - Java
  - Python
---
**Go** is a statically typed language from Google with garbage collection, structural interfaces, and built-in goroutines / channels for concurrency. Common for CLIs, network services, and cloud tooling.

```go
package main

import "fmt"

func greet(name string) string {
	return "hello, " + name
}

func main() {
	fmt.Println(greet("world"))
}
```
