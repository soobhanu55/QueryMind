# Cost- and latency-aware routing (`LLM_PROVIDER=routed`)

The rule-based parser answers first (free, under a millisecond). The model is called only when the rules are not confident
(`ROUTER_MIN_CONFIDENCE`, default 0.55) or fail. Each tier has a timeout and a circuit breaker; if the model is down, over its
daily call budget (`ROUTER_LLM_DAILY_CALLS`), or not allowed for a private question, the rules answer is returned and marked
`rules_degraded`. Questions that mention e-mail addresses, phone numbers or addresses (`ROUTER_PRIVATE_PATTERN`) are never
sent to a hosted API. Every `/query` response says which tier answered (`route`), the model, tokens, and the list-price cost;
`GET /llm/stats` shows calls per tier, escalations, fallbacks and breaker states.

## Measured on the 74-question benchmark

Same questions, same schema prompt, same guardrail and result check. The model is Qwen2.5-1.5B-Instruct run locally (RTX 4050).
`python tests/accuracy/run_accuracy.py --provider routed` with `ROUTER_LLM=local ROUTER_MIN_CONFIDENCE=<t>`.

| Setup | Accuracy | Model calls | Mean generation time |
|---|---|---|---|
| Model for every question | 68.9% (51/74) | 74 | about 3.7 s |
| Rules only | 91.9% (68/74) | 0 | 0.5 ms |
| **Routed, threshold 0.55 (default)** | **93.2% (69/74)** | **1** | 286 ms (one cold model load of 21 s dominates) |
| Routed, threshold 0.65 | 81.1% (60/74) | 34 | 1.7 s |
| Routed, threshold 0.80 | 75.7% (56/74) | 60 | 3.3 s |

Answers by tier at threshold 0.65: the 40 questions the rules were confident about were 40/40 correct; the 34 escalated to
the model were 20/34 correct (9 of the 34 produced SQL that failed on execution). Mean generation time is over the answers that
executed.

## What this shows, and what it does not

* **Routing cuts model calls by 99% here (1 of 74) at no loss in accuracy**, because the rules are right whenever they are
  confident. At 74 questions the gain over rules-only (one question) is not statistically meaningful; the point is the saving.
* **Escalating more hurts**: this small model is worse than the rules, so a higher threshold lowers accuracy (81.1%, 75.7%).
  Routing only helps when the escalation tier is better than the cheap tier on the questions it receives. A stronger model
  (`ROUTER_LLM=groq`, Llama 3.3 70B on Groq's free tier) should change that trade-off; it has not been measured.
* **The rules were written against this question set**, so their confidence is better calibrated here than it would be on
  questions nobody has seen. On unseen traffic expect more escalations and a lower rules-only accuracy.
* **Cost is not yet in euros**: the local model has no price, so `cost_usd` is `null` for it (reported as unpriced, not
  guessed); a Groq call is priced at list price (Llama 3.3 70B, $0.59 / $0.79 per 1M tokens) even though the free tier bills nothing.
* One failure mode is not covered: when the rules produce confident but invalid SQL (benchmark question 30), no tier escalates
  because the error only appears at execution time. Retrying with the model on an execution error is the natural next step.
