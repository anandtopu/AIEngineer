# Module 17 -- Modern Training, Alignment, and Compression

Goal: understand the techniques actually used to build 2026 open-weight
models (LLaMA 3, Qwen 2.5, Mistral, Gemma 2, Phi-4) and align them for
end-user use. This module covers the non-obvious training choices that
interviewers probe for "do you know the current state of the art?"

## What to read / run

- `src/deep_learning/rope_positional.py` -- rotary positional embeddings.
- `src/deep_learning/kv_cache.py` -- the key speedup behind fast inference.
- `src/deep_learning/speculative_decoding.py` -- lossless decoding speedup.
- `src/deep_learning/lora_from_scratch.py` -- parameter-efficient fine-tuning.
- `src/deep_learning/dpo_from_scratch.py` -- preference learning without PPO.
- `src/deep_learning/knowledge_distillation.py` -- teacher-student compression.
- `src/deep_learning/clip_dual_encoder.py` -- contrastive training for embeddings.
- `src/advanced/quantization_int8.py` -- INT8 weight quantization.

## Modern Transformer choices

| Choice                   | Older default       | 2026 default                          |
|--------------------------|---------------------|---------------------------------------|
| Positional encoding      | Sinusoid / learned  | **RoPE**, often with NTK / YARN scaling |
| Normalization            | LayerNorm           | **RMSNorm** (cheaper, same accuracy)    |
| Activation               | ReLU / GELU         | **SwiGLU**                              |
| Attention variant        | Full MHA            | **Grouped-query attention (GQA)** or MQA |
| Attention kernel         | Standard            | **FlashAttention-2 / 3**                |
| Precision                | FP32                | BF16 train, FP8 where supported         |
| Tokenizer                | WordPiece           | SentencePiece BPE (GPT / LLaMA)         |
| Context length           | 2K                  | 32K-1M+ with position scaling           |
| Pretraining scale law    | Kaplan              | **Chinchilla** (more data, less params) |

## Alignment pipelines

**Old pipeline** (InstructGPT, 2022):
1. SFT -- supervised fine-tune on instruction-following demos.
2. Reward model -- train a classifier on ranked pairs.
3. PPO -- RL against the reward model with a KL penalty.

**Modern pipeline** (2024-2026):
1. SFT on high-quality instruction data.
2. **DPO** (Direct Preference Optimization) on pairs of
   (chosen, rejected). Closed-form replacement for PPO that needs
   only a frozen reference + a policy.
3. Optional **rejection-sampling fine-tuning**: sample N outputs,
   keep the best under a reward model / LLM judge, SFT on those.
4. Optional **identity preference optimization** variants (IPO, KTO)
   for specific failure modes of DPO.

## Parameter-efficient fine-tuning

| Method      | Trainable params | When to pick it                      |
|-------------|------------------|--------------------------------------|
| Full FT     | 100%             | Max quality, max compute             |
| **LoRA**    | 0.1-1%           | Default PEFT, works almost everywhere |
| **QLoRA**   | 0.1-1%           | LoRA on a 4-bit base. Fits huge models on one GPU. |
| Adapters    | 1-3%             | Legacy; LoRA has mostly replaced it  |
| Prefix tuning | <0.1%          | Very cheap, less robust              |
| BitFit      | 0.01%            | Only tune biases. Baseline.          |

## Compression

- **Post-training quantization**: FP16 -> INT8 is roughly free.
  INT4 requires care (AWQ, GPTQ, NF4).
- **Quantization-aware training** when PTQ drops accuracy too much.
- **Pruning**: structured (channel, head) + unstructured (magnitude).
- **Distillation**: the dominant ship-a-smaller-model technique.
  See `knowledge_distillation.py` for the Hinton-style loss.

## Inference optimization

1. **KV cache** -- mandatory for autoregressive decoding.
2. **Grouped / multi-query attention** -- shrink KV cache by sharing
   K, V across heads.
3. **FlashAttention** -- IO-aware attention kernel; massive speedup on GPU.
4. **Continuous batching** (vLLM) -- pack requests of different lengths.
5. **Paged attention** -- treat KV cache like virtual memory.
6. **Speculative decoding** -- small draft + big target, lossless.
7. **Prompt / prefix caching** -- reuse the KV cache for a fixed prefix.
8. **Distillation + quantization** -- ship the smallest model that hits
   your quality bar.

## Common interview questions

- Why did the field move from sinusoidal to RoPE?
- Explain DPO vs PPO-RLHF. Why did DPO take over?
- How do GQA / MQA save memory? What do they trade off?
- Your 70B model's first-token latency is 3s. What would you try?
- When would you distill vs fine-tune vs prompt-engineer?
- What does "Chinchilla-optimal" mean and why does it matter?

## Drills

1. Modify `kv_cache.py` to use grouped-query attention (one K, V pair
   shared across all heads). Compare cache size.
2. Train the `dpo_from_scratch.py` example with different `beta` values
   and plot the win-rate. Smaller beta -> closer to reference, larger
   beta -> more drift.
3. Distill the teacher in `knowledge_distillation.py` into an even
   smaller student (hidden=16). Does more temperature help?
