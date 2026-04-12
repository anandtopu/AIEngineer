# Project 06 -- Mini LLM Evaluation Harness

A self-contained evaluation pipeline for LLMs that runs offline (no API key,
no model download). Mirrors the architecture of real harnesses such as
`lm-eval-harness`, OpenAI evals, and Anthropic evals.

## Pipeline

```
suite of Examples  ->  generator (mock LLM)  ->  graders  ->  per-slice aggregator
```

## What you'll learn

- How reference-based metrics differ in behavior:
  - **exact_match** is unforgiving (`"Nine"` != `"9"`).
  - **token_f1** rewards partial overlap.
  - **rouge_l** rewards longest common subsequence.
  - **llm_judge** (stub) tolerates paraphrase.
- Why per-slice scoring matters: a model can be great on factoids and
  awful on math, and the average will hide it.
- The shape of a real eval harness, so you can plug a real LLM into the
  `mock_llm` slot and run it against MMLU / TriviaQA / your own dataset.

## Run

```bash
python projects/06_mini_llm_eval/run.py
```

## Extension ideas

- Add a `bootstrap_ci(scores, alpha=0.05)` helper and report 95% CIs.
- Replace `mock_llm` with a real API call (Anthropic, OpenAI, local llama.cpp).
- Add a contamination check that flags examples whose answer appears
  verbatim in the prompt.
- Add a "judge" that calls a second LLM to grade open-ended answers.
