# Module 12 -- Transformer Internals

Goal: be able to derive the Transformer block on a whiteboard and answer
"why does this design choice exist?" questions.

## What to read

- `src/deep_learning/attention_from_scratch.py` -- scaled dot-product and multi-head attention in numpy.
- `src/deep_learning/transformer_encoder_torch.py` -- minimal pre-norm encoder block in PyTorch.
- `src/deep_learning/autograd_micro.py` -- backprop from first principles. Helps you trust the Transformer's gradient flow.

## Concepts you must be able to explain

1. **Q, K, V intuition** -- queries ask, keys advertise, values are the payload returned in a soft-lookup.
2. **The sqrt(d_k) scaling** -- without it, dot products grow with dimension and softmax saturates, killing gradients.
3. **Multi-head attention** -- different heads attend to different relations (syntax, coreference, position). Concatenating then projecting recombines them.
4. **Causal mask** -- decoder-only models must zero out attention to future positions, otherwise the loss is trivially solvable by copying the answer.
5. **Positional encoding** -- attention is permutation-invariant; positions must be injected via sinusoids, learned embeddings, or RoPE.
6. **Pre-norm vs post-norm** -- pre-norm (LayerNorm before the residual branch) is more stable for deep stacks; the original paper used post-norm.
7. **FFN block** -- the feed-forward layer is where most of the parameters live; attention is the mixer, FFN is the per-token transformer.
8. **Residual connections** -- without them, deep stacks fail to train at all.

## Common interview questions

- Walk through a single forward pass of a Transformer block, tensor shapes included.
- Why is attention O(T^2) in sequence length? What are the standard mitigations? (Flash Attention, sparse attention, sliding window, linear attention, KV caching for inference.)
- What is the difference between encoder-only (BERT), decoder-only (GPT), and encoder-decoder (T5) variants?
- Where does dropout typically go inside a Transformer block?
- How would you adapt a Transformer for very long context?

## Drills

1. Run `attention_from_scratch.py` and modify it to print the attention matrix
   for a *non*-causal forward pass. Check the rows still sum to 1.
2. Modify `transformer_encoder_torch.py` to switch from pre-norm to post-norm and
   compare loss curves on a tiny synthetic task.
3. Add a `KVCache` to a single attention head (track K and V across timesteps so
   you only compute one new query per step). This is the core trick behind
   fast LLM inference.
