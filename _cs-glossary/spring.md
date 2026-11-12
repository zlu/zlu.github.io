---
title: Spring
synonyms:
  - Spring Framework
  - Spring Boot
related:
  - Java
  - Kotlin
  - Django
  - Dotnet
---
**Spring** is the dominant application framework on the JVM ([Java](/cs-glossary/java/), [Kotlin](/cs-glossary/kotlin/)). Dependency injection, web MVC / WebFlux, and data access. **Spring Boot** adds opinionated defaults and embedded servers for services.

```java
@RestController
public class HelloController {
  @GetMapping("/hello")
  public String hello(@RequestParam String name) {
    return "hello, " + name;
  }
}
```
