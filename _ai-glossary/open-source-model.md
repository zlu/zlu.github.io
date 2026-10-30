---
title: Open-Source Model
synonyms:
  - Open Source Model
  - Open-Source Models
  - Open-Weight Model
  - OSS LLM
related:
  - Base Model
  - Instruct Model
  - Pretraining
  - Fine-Tuning
  - Continued Pretraining
---
An **open-source model** (often **open-weight**) is a language or multimodal model whose weights are publicly downloadable under a stated license, so you can run, inspect, and further train it yourself. Licenses differ: some are permissive for commercial use; others restrict it. Always read the model card.

These families finished large-scale [pretraining](/ai-glossary/pretraining/) (and often instruction tuning) before release. Product work usually starts from one of them, then applies [continued pretraining](/ai-glossary/continued-pretraining/) or [fine-tuning](/ai-glossary/fine-tuning/) on your data.

### Common families (entry points)

| Family | Org | Links | Notes |
|:-------|:----|:------|:------|
| Qwen | Alibaba Cloud | [qwen.ai](https://qwen.ai/){:target="_blank" rel="noopener noreferrer"}, [HF](https://huggingface.co/Qwen){:target="_blank" rel="noopener noreferrer"}, [GitHub](https://github.com/QwenLM){:target="_blank" rel="noopener noreferrer"} | Dense and MoE lines; strong multilingual (including Chinese). Common starting point for domain CPT and fine-tuning. |
| gpt-oss | OpenAI | [openai.com/open-models](https://openai.com/open-models/){:target="_blank" rel="noopener noreferrer"}, [HF](https://huggingface.co/openai){:target="_blank" rel="noopener noreferrer"}, [GitHub](https://github.com/openai/gpt-oss){:target="_blank" rel="noopener noreferrer"} | Apache 2.0 MoE reasoning models (20B / 120B). Not ChatGPT API weights. Expect OpenAI’s harmony chat format; strong on reasoning and tool use. |
| Llama | Meta | [llama.com](https://www.llama.com/){:target="_blank" rel="noopener noreferrer"}, [HF](https://huggingface.co/meta-llama){:target="_blank" rel="noopener noreferrer"} | Ecosystem default: most tutorials, serving stacks, and fine-tune recipes assume Llama-shaped models. Community license; check commercial terms per release. |
| Mistral | Mistral AI | [mistral.ai](https://mistral.ai/){:target="_blank" rel="noopener noreferrer"}, [HF](https://huggingface.co/mistralai){:target="_blank" rel="noopener noreferrer"} | Efficient open weights (including Mixtral MoE). Many releases are Apache 2.0. Company also sells closed API models under the same brand. |
| Gemma | Google | [Docs](https://ai.google.dev/gemma){:target="_blank" rel="noopener noreferrer"}, [HF](https://huggingface.co/google){:target="_blank" rel="noopener noreferrer"} | Open weights under Google’s Gemma license (not Apache). Competitive small and mid sizes; later versions add multimodal. |
| DeepSeek | DeepSeek | [deepseek.com](https://www.deepseek.com/){:target="_blank" rel="noopener noreferrer"}, [HF](https://huggingface.co/deepseek-ai){:target="_blank" rel="noopener noreferrer"} | Often MoE; strong on math, code, and reasoning benchmarks relative to size. Popular when you want open weights close to frontier open scores. |
| Phi | Microsoft | [HF](https://huggingface.co/microsoft){:target="_blank" rel="noopener noreferrer"} | Small models trained for quality at low parameter count. Good for local / edge boxes where Llama-70B-class hardware is not an option. |
| OLMo | Allen Institute for AI | [allenai.org/olmo](https://allenai.org/olmo){:target="_blank" rel="noopener noreferrer"}, [HF](https://huggingface.co/allenai){:target="_blank" rel="noopener noreferrer"} | Research-first openness: weights plus training data and code. Usually chosen for reproducibility, not as the strongest chat default. |

