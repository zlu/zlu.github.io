---
title: .NET
synonyms:
  - Dotnet
  - DotNet
  - CLR
related:
  - C#
  - F#
  - Java
  - Spring
---
**.NET** is Microsoft’s runtime and library stack for [C#](/cs-glossary/csharp/), [F#](/cs-glossary/fsharp/), and related languages (modern .NET / previously .NET Core and the .NET Framework). ASP.NET is the usual web stack on top.

```csharp
// Minimal ASP.NET endpoint (.NET 6+)
var app = WebApplication.CreateBuilder(args).Build();
app.MapGet("/hello/{name}", (string name) => $"hello, {name}");
app.Run();
```
