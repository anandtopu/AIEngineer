# Module 13 -- LLM Inference, Fine-tuning, and Compression

Goal: be the person on the team who can take a base LLM, adapt it,
shrink it, and serve it.

## What to read

- `src/advanced/bpe_tokenizer.py` -- how text becomes ids.
- `src/advanced/llm_sampling.py` -- how logits become tokens.
- `src/advanced/quantization_int8.py` -- how big models become small.
- `src/deep_learning/lora_from_scratch.py` -- how big models become task-specific.
- `projects/06_mini_llm_eval/run.py` -- how you measure whether any of it actually worked.

## Concepts you must be able to explain

### Tokenization
- Why subword? Character-level is too long, word-level has OOV problems.
- BPE (used by GPT) vs WordPiece (used by BERT) vs SentencePiece (used by T5/LLaMA).
- The "</w>" end-of-word marker and why it matters for detokenization.

### Decoding / sampling
- Greedy = deterministic, but often boring and prone to loops.
- Temperature scales logits before softmax. T<1 sharpens, T>1 flattens.
- Top-k truncates to the k most likely tokens. Simple but can clip fat tails.
- Top-p (nucleus) keeps the smallest set whose cumulative probability >= p.
  Adapts to the local entropy of the distribution.
- Repetition penalty, frequency penalty, presence penalty -- ad hoc but useful.
- Beam search is mostly used for translation, not open-ended generation.

### Fine-tuning
- Full fine-tuning: update every parameter. Expensive but maximally expressive.
- LoRA: freeze the base, add a low-rank update (B @ A) per linear layer.
  Trainable params drop by 100-1000x; quality stays close to full FT.
- QLoRA: LoRA on top of a 4-bit quantized base. Lets you fine-tune 65B models
  on a single consumer GPU.
- Prompt tuning / prefix tuning: even cheaper, but less robust.
- RLHF / DPO: teach the model human preferences via paired comparisons.

### Quantization
- Symmetric vs asymmetric: where the zero point sits.
- Per-tensor vs per-channel: per-channel preserves accuracy in linear layers.
- Post-training quantization (PTQ) vs quantization-aware training (QAT).
- INT8 is mostly free; INT4 requires care; FP8 is becoming standard for training.

### Inference optimization
- KV cache: avoid recomputing K and V for previous tokens at each step.
- Continuous batching: pack many users' requests into one GPU batch.
- Flash Attention: tile attention so it fits in SRAM and becomes IO-bound.
- Speculative decoding: a small draft model proposes tokens, the big model
  verifies them in parallel.

### Evaluation
- Reference-based: exact match, F1, BLEU, ROUGE, BERTScore.
- Reference-free: perplexity, coherence, factuality classifiers.
- LLM-as-judge: cheap and scalable, but biased toward verbose / similar models.
- Always report per-slice scores, not just an overall number.

## Common interview questions

- "We need to fine-tune Llama-2-70B on customer support tickets, but we only have one A100. Walk me through your plan." (QLoRA + 4-bit base + PEFT.)
- "Inference latency is 800ms p99. Where do we start?" (Profile -> KV cache -> batching -> speculative decoding -> quantization.)
- "Our LLM judge says model B is better but human eval says model A is better. What is going on?" (Judge bias -- length, similarity to judge, formatting.)
- "Greedy decoding is producing repetitive text. Why and what would you change?"

## Drills

1. Run `llm_sampling.py` and add a `min_p` sampling function (variant of top-p
   that uses a probability floor instead of cumulative mass).
2. Modify `lora_from_scratch.py` to also train a LoRA adapter on a *second*
   task and demonstrate that you can swap A, B in and out without touching
   the base weight.
3. Extend `projects/06_mini_llm_eval/run.py` to compute bootstrap 95% CIs on
   the per-slice scores. Real harnesses always report uncertainty.
