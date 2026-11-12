---
title: Objective-C
synonyms:
  - ObjC
related:
  - Smalltalk
  - Swift
  - C
  - Java
---
**Objective-C** is [C](/cs-glossary/c/) plus a [Smalltalk](/cs-glossary/smalltalk/)-style dynamic message-send object model. Long the main language for macOS / iOS; largely superseded for new code by [Swift](/cs-glossary/swift/).

```objc
#import <Foundation/Foundation.h>

NSString *greet(NSString *name) {
  return [NSString stringWithFormat:@"hello, %@", name];
}

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    NSLog(@"%@", greet(@"world"));
  }
  return 0;
}
```
