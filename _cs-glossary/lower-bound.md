---
title: Lower bound
synonyms:
  - Information-theoretic lower bound
related:
  - Big-Omega
  - Comparison sort
  - Time complexity
---
A **lower bound** argues that *any* algorithm in a model needs at least a certain cost on some inputs. Example: any [comparison sort](/cs-glossary/comparison-sort/) requires $\Omega(n \log n)$ comparisons in the worst case (decision-tree / information argument).

Lower bounds are about problems, not one implementation. Beating an $\Omega(n \log n)$ comparison lower bound means leaving the comparison model (integer counting, radix, hashing tricks), not finding a cleverer `if (a < b)`.

Write lower bounds with [$\Omega$](/cs-glossary/big-omega/); write algorithm upper bounds with [$O$](/cs-glossary/big-o/) or [$\Theta$](/cs-glossary/big-theta/).
