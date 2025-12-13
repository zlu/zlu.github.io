---
title: Zero-Shot
synonyms: []
related:
  - N-Shot
---
In the context of **machine learning (especially few-shot learning)** and **prompt engineering for large language models (LLMs)**, **zero-shot**, **1-shot (one-shot)**, and their variants refer to different ways of providing examples to a model to guide its output. These paradigms are critical for adapting pre-trained models to new tasks *without extensive fine-tuning*.

### 1. **Zero-Shot Learning**
**Definition**: The model is tasked to perform a task **without any explicit examples** of the desired input-output pairs. It relies solely on its pre-trained knowledge and a natural language description of the task.

#### Key Characteristics
- No demonstration examples are provided in the prompt.
- The model uses its understanding of language, logic, and world knowledge learned during pre-training.
- Commonly used for tasks that are intuitive or align with the model’s training data.

#### Example (Text Classification)
**Prompt**:
> Classify the following sentence into "positive" or "negative": "The new smartphone has a terrible battery life."

**Model Output**:
> negative

### 2. **1-Shot (One-Shot) Learning**
**Definition**: The model is given **exactly one example** of the target task’s input-output pair to learn the pattern before being asked to perform the task on new data.

#### Key Characteristics
- One demonstration example is included in the prompt to clarify the task’s requirements.
- Useful when the task is specific or the model might misinterpret the zero-shot instruction.

#### Example (Text Classification)
**Prompt**:
> Example: Sentence: "I love this movie!" Label: positive
> Now classify the following sentence into "positive" or "negative": "The new smartphone has a terrible battery life."

**Model Output**:
> negative

### 3. **N-Shot Learning (Few-Shot Learning)**
**Definition**: A generalization where the model is given **N examples** (typically $N \geq 2$ and small, e.g., 2–5) of the task to learn the pattern. "Few-shot" is an umbrella term that includes 1-shot as a special case.

#### Key Characteristics
- More examples help the model grasp complex patterns (e.g., nuanced classification, named entity recognition).
- The number of examples \(N\) is small compared to full fine-tuning (which uses thousands/millions of samples).

#### Example (Named Entity Recognition, 2-shot)
**Prompt**:
> Example 1: Sentence: "Apple was founded by Steve Jobs in Cupertino." Entities: Apple (Company), Steve Jobs (Person), Cupertino (City)
> Example 2: Sentence: "Tesla’s factory is located in Austin." Entities: Tesla (Company), Austin (City)
> Now extract entities from the sentence: "Microsoft was established by Bill Gates in Seattle."

**Model Output**:
> Microsoft (Company), Bill Gates (Person), Seattle (City)

### 4. **Comparison Table**
| Paradigm      | Number of Examples | Core Idea                                  | Use Case                                  |
|---------------|--------------------|--------------------------------------------|-------------------------------------------|
| Zero-Shot     | 0                  | Rely on pre-trained knowledge + task description | Simple, intuitive tasks (e.g., sentiment, summarization) |
| 1-Shot        | 1                  | One example to clarify task rules          | Tasks with ambiguous instructions         |
| N-Shot (Few-Shot) | N (2–5)        | Multiple examples to learn complex patterns | Specialized tasks (e.g., entity extraction, code generation) |

### Key Takeaway
These paradigms enable pre-trained models to adapt to new tasks **efficiently**—without the need for costly fine-tuning on large datasets. The choice of zero/1/N-shot depends on the task complexity and the model’s familiarity with the target domain.
