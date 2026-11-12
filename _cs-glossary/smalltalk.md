---
title: Smalltalk
synonyms:
  - Smalltalk-80
related:
  - Ruby
  - Java
  - Objective-C
  - React
---
**Smalltalk** is a dynamically typed, purely object-oriented language (Xerox PARC; Smalltalk-80 is the usual reference dialect). Everything is an object; control flow is message sends. It pioneered live IDEs, MVC, and the **dependents** change-notification mechanism that the Gang of Four later catalogued as Observer.

Dependents (subject broadcasts change; views update):

```smalltalk
"Model"
anAccount changed: #balance.

"Dependent / view"
MyView>>update: aSymbol
  aSymbol = #balance ifTrue: [self refreshBalance].
```

Modern OOP and many UI toolkits still use the same subscribe / notify shape. See Krasner and Pope (1988) for the Smalltalk-80 MVC write-up.
