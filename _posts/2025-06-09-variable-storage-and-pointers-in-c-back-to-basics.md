---
layout: post
title: 'Variable Storage and Pointers in C: Back to Basics'
date: 2025-06-09 06:04 +0800
tags:
  - c
  - variable storage
  - pointer
  - data segment
  - unix
  - linux
description: "Variable Storage and Pointers in C: Back to Basics"
comments: true
---

Understanding C Pointers and Memory Layout.

A C program's memroy is divided into several segments.  Text/Code Segment stores executable instructions.  Data Segment stores global and static variables, initialized or uninitialized.  Uninitialized are stored in BBS (Block Started by Symbol).  Heaps are dynamically allocated memory using malloc, calloc etc. Stack stores local variables, function parameters, and return addresses.

A C pointer is a variable that stores a memory address.  For example:

```c
int x = 10;     // An integer variable
int *p = &x;    // p is a pointer to an integer, storing the address of x
```

Now there's such a thing called pointer to a pointer, meaning a pointer that stores the address of another pointer.  For example:

```c
int x = 10;
int *p = &x;     // p points to x
int **pp = &p;   // pp points to p
```

There are three type of variable storages in C: stack, heap, and data segment.  Stack stores local variables and function parameters.  Heap stores dynamically allocated memory.  Data segment stores global and static variables.


### Stack
The stack is a region of memory used for **function calls**. It stores **local (non-static) variables**, **function parameters**, and **return addresses**.    Variables on the stack exist **only while the function is running**. When the function returns, the stack space is reclaimed.  Local variables on the stack are **not automatically initialized**.
- **Example:**  
  ```c
  void foo() {
      int local_var = 2; // stack
  }
  ```

### Heap

The heap is a region of memory used for **dynamic memory allocation**. Unlike stack memory, heap memory must be **manually managed** by the programmer using functions like `malloc()`, `calloc()`, `realloc()`, and `free()`. Memory on the heap persists until it is explicitly deallocated or the program terminates.  It is also limited by avaiable system meory with a slower access speed than stack due to dynamic allocation.  It can thus become fragmented over time.  There is **no automatic initialization** unless using `calloc()`.

**Example:**
```c
void dynamic_allocation() {
    // Allocate memory for an integer on the heap
    int *heap_var = (int *)malloc(sizeof(int));
    *heap_var = 42;  // Store value 42 in the allocated memory
    
    // Allocate array of 10 integers
    int *array = (int *)malloc(10 * sizeof(int));
    for (int i = 0; i < 10; i++) {
        array[i] = i * 2;  // Initialize array elements
    }
    
    // Always free allocated memory when done
    free(heap_var);
    free(array);
}
```

### Data Segment

The data segment is a portion of a program's static memory where **global variables**, **static variables** (both inside and outside functions), and their initial values are stored.  Variables in the data segment exist for the **entire duration of the program**.  They `maintain` their values between function calls.  The **initialized data segment** stores variables with explicit initial values (e.g., `int x = 5;`).  The **uninitialized data segment** (BSS - Block Started by Symbol) stores variables without explicit initial values (e.g., `static int y;`).  They are however initialized to zero by default.

- **Example:**  
  ```c
  int global_var = 10;      // data segment
  static int static_var = 5; // data segment
  void foo() {
      static int y = 3;     // data segment
  }
  ```

  The key differences from other memory segments are:
  - v.s. Stack: Stack stores local variables and function call information.  It is automatically managed (pushed/popped with function calls).  They are temporary and exist on within their scope.
  - v.s. Heap: Heap is used for dynamic memory allocation (mallo, calloc, etc.).  Heap memory must be manually managed (allocated/freed by the programmer).  In addition, heap memory persists until explicitly freed or until program termination.

  We use data segment for global/static variables throughout the program's lifetime.  And when we want their states/values to be maintained between function calls.  Usually we use them for global configuration values or constants.  Though we should not overuse global variables as it makes debugging harder.  There is also the thread safety isue to be considered.

### Key Differences

| Feature         | Data Segment                | Stack                        |
|-----------------|----------------------------|------------------------------|
| Scope           | Global/static variables     | Local (automatic) variables  |
| Lifetime        | Whole program               | During function call         |
| Initialization  | At program start            | At function entry            |
| Memory location | Fixed (not per function)    | Grows/shrinks with calls     |
| Example         | `static int x;`            | `int y;` in a function       |

### Exercises
Let's take a look at these exercises:

1. **What is the size of an `int *` type, where the size of `int` is 32 bits and the CPU uses 64-bit addressable memory?**

**Answer**: 8 bytes  
**Explanation**: Pointer size depends on 64-bit addressable memory (8 bytes), not `int` size.


2. **What is the expected output of the following C code?**

```c
void foo(int x) {
    if (x < 5) {
        static int y = 5;
        x = y + x;
        printf("%d, ", x);
        y += 1;
    }
}
void main() {
    for (int i = 7; i >= 0; i--)
        foo(i);
}
```

**Answer**: 9, 9, 9, 9, 9 (tricky: 9, 8, 7, 6, 5)
**Explanation**: 

When static int y = 5 is declared inside a function, `y` is intialized only once, the first time the function is called.  After that, `y` remembers its value between calls, because it is stored in the program's data segment, not on the stack.


3. **Memory contents after line 10:**

```c
static char val = -47;  // Static variable in data segment
int main(int argc, char **argv) {
    char *str = "XYZA";          // String literal in text segment
    char tr = str[2];            // 'Z' (ASCII 90)
    char *sp = &val;             // Pointer to val
    char x = *sp + 2;            // Dereference sp (-47) + 2 = -45
    char **holder = malloc(sizeof(char *));  // Allocate memory for a char pointer
    *holder = &val;              // Make the allocated pointer point to val
    tr = tr + **holder + 2;      // 90 + (-47) + 2 = 45
    ; // snapshot
}
``` 

**What is the Memory Layout After Line 10?**

- Static/Global Segment (0x2000)
  - val is -47 (which is 0xD1 in two's complement 8-bit)
  - Stored at address 0x2000
- Text Segment (0x3000)
  - String literal "XYZA" is stored here
  - 'X' (88), 'Y' (89), 'Z' (90), 'A' (65), '\0' (0)
- Heap (0x4000)
  - malloc(sizeof(char *)) allocates memory for one pointer
  - holder points to 0x4000
  - The memory at 0x4000 contains 0x2000 (address of val)
- Stack (grows downward from 0x7000)
  - str (4 bytes): 0x3000 (points to "XYZA")
  - tr (1 byte): 0 (result of 90 + (-47) + 2)
  - sp (4 bytes): 0x2000 (points to val)
  - x (1 byte): -45 (from *sp + 2 which is -47 + 2)
  - holder (4 bytes): 0x4000 (points to the allocated memory)

**Answer**:  
- **Static/Global (0x2000)**: `val = -47 (0xD1)` at `0x2000`  
- **Heap (0x4000)**: `holder` points to `0x4000`, `*holder = 0x2000`  
- **Stack (0x7000)**:  
  - `str = 0x3000` at `0x7000`  
  - `tr = 0` at `0x7004`  
  - `sp = 0x2000` at `0x7005`  
  - `x = -45` at `0x7009`  
  - `holder = 0x4000` at `0x700A`  
- **Program code (0x3000)**: `"XYZA"` (`'X'=88`, `'Y'=89`, `'Z'=90`, `'A'=65`, `'\0'=0`)  

**Explanation**:  
- `val` at `0x2000` = `-47`.  
- `str` points to `"XYZA"` at `0x3000`, `tr = 'Z' = 90`.  
- `sp = &val = 0x2000`, `x = -47 + 2 = -45`.  
- `holder` at `0x4000`, `*holder = 0x2000`.  
- `tr = 90 + (-47) + 2 = 0`.  
- Stack: `str` (4 bytes), `tr` (1 byte), `sp` (4 bytes), `x` (1 byte), `holder` (4 bytes).
