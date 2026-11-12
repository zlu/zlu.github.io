---
title: Time complexity
synonyms:
  - Running time
  - Time cost
related:
  - Big-O
  - Space complexity
  - Amortized analysis
  - Polynomial time
---
**Time complexity** describes how an algorithm’s running time grows with input size, usually in [Big-O](/cs-glossary/big-o/) / [$\Theta$](/cs-glossary/big-theta/) of elementary steps (comparisons, arithmetic, memory words), not wall-clock seconds.

State the case: best, average, or worst. [Quicksort](/cs-glossary/quicksort/) is $\Theta(n \log n)$ average and $\Theta(n^2)$ worst; [hash table](/cs-glossary/hash-table/) lookup is expected $O(1)$ and $O(n)$ worst with bad hashing.

[Polynomial time](/cs-glossary/polynomial-time/) is the theory cutoff for “tractable” decision procedures. In engineering, cache behavior and constants still decide which $O(n \log n)$ routine wins.
