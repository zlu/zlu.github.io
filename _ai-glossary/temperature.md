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


