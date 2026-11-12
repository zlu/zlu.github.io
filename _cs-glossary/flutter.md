---
title: Flutter
synonyms: []
related:
  - Dart
  - React
  - Swift
  - Kotlin
---
**Flutter** is a UI toolkit from Google for mobile, desktop, and web, written in [Dart](/cs-glossary/dart/). Widgets are the composition unit; a rebuild propagates when state changes (similar job to [React](/cs-glossary/react/) components).

```dart
class Counter extends StatefulWidget {
  @override
  State<Counter> createState() => _CounterState();
}

class _CounterState extends State<Counter> {
  int count = 0;

  @override
  Widget build(BuildContext context) {
    return TextButton(
      onPressed: () => setState(() => count++),
      child: Text("$count"),
    );
  }
}
```
