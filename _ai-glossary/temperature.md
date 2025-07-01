---
title: Temperature
---
Temperature is a scalar value (usually between 0 and 2) used during the sampling process from a probability distribution to control the level of randomness in the output.

When a model generates text, it computes a probability distribution over the possible next tokens (words, characters, etc.). The temperature modifies this distribution before sampling:

{% raw %}
$$
P_i^{(\text{adjusted})} = \frac{\exp\left(\frac{\log P_i}{T}\right)}{\sum_j \exp\left(\frac{\log P_j}{T}\right)}
$$
{% endraw %}

Where:
- $P_i$ is the original probability of token i,
- $T$ is the temperature.

#### Example
 Example:

Prompt: “Once upon a time, in a land far away,”
- T = 0.2 → “there lived a wise old king who ruled with kindness and wisdom.”
- T = 1.0 → “a dragon taught poetry to wandering clouds.”
- T = 1.8 → “the moon whispered jellyfish secrets through laser bananas.”

####  Use Cases:
- Low temperature (0–0.5): Factual answers, programming help, summarization.
- Medium temperature (0.7–1.0): Creative writing, marketing copy, storytelling.
- High temperature (1.2+): Brainstorming ideas, surreal or poetic content.
