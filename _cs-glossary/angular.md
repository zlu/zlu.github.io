---
title: Angular
synonyms:
  - AngularJS
related:
  - TypeScript
  - JavaScript
  - React
  - Vue
---
**Angular** is a [TypeScript](/cs-glossary/typescript/)-first SPA framework from Google. Components, dependency injection, and RxJS-heavy async. (The older “AngularJS” 1.x line is separate; current Angular is 2+.)

```typescript
import { Component, signal } from "@angular/core";

@Component({
  selector: "app-counter",
  template: `<button (click)="inc()">{{ count() }}</button>`,
})
export class CounterComponent {
  count = signal(0);
  inc() {
    this.count.update((n) => n + 1);
  }
}
```
