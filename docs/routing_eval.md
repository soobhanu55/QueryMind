# Cost- and latency-aware routing (`LLM_PROVIDER=routed`)

The rule-based parser answers first (free, under a millisecond). The model is called only when the rules are not confident
(`ROUTER_MIN_CONFIDENCE`, default 0.55) or fail. Each tier has a timeout and a circuit breaker; if the model is down, over its
daily call budget (`ROUTER_LLM_DAILY_CALLS`), or not allowed for a private question, the rules answer is returned and marked
`rules_degraded`. Questions that mention e-mail addresses, phone numbers or addresses (`ROUTER_PRIVATE_PATTERN`) are never
sent to a hosted API. Every `/query` response says which tier answered (`route`), the model, tokens, and the list-price cost;
`GET /llm/stats` shows calls per tier, escalations, fallbacks and breaker states.

## Does routing improve accuracy? Not with the model I could test, and the first numbers were optimistic

The model is Qwen2.5-1.5B-Instruct run locally (RTX 4050); `ACCURACY_SET=<file> ROUTER_LLM=local ROUTER_MIN_CONFIDENCE=<t> python tests/accuracy/run_accuracy.py --provider routed`.

**Original 74 questions** (the rules were written against these):

| Setup | Accuracy | Model calls |
|---|---|---|
| Model only | 68.9% (51/74) | 74 |
| Rules only | 91.9% (68/74) | 0 |
| Routed, threshold 0.55 | 93.2% (69/74) | 1 |
| Routed, threshold 0.65 | 81.1% (60/74) | 34 |
| Routed, threshold 0.80 | 75.7% (56/74) | 60 |

**35 new questions written afterwards** (`tests/accuracy/heldout_set.jsonl`, different phrasings, same schema; never used to
build the rules):

| Setup | Accuracy | Model calls |
|---|---|---|
| Model only | 51.4% (18/35) | 35 |
| **Rules only** | **60.0% (21/35)** | 0 |
| Routed, threshold 0.55 | 60.0% (21/35) | 0 (the rules' confidence is never below 0.55) |
| Routed, threshold 0.75 | 51.4% (18/35) | 27 |
| Routed, threshold 0.80 | 51.4% (18/35) | 31 |

What the second table shows:

* **The 91.9% rule-parser figure does not carry over: it is 60% on unseen phrasings.** The earlier "93.2% with 1 model call
  instead of 74" was true on the questions the rules were built on and says nothing about new traffic.
* **The rules' confidence is a weak signal on new questions.** Answers at confidence 0.75 or above were 7/8 correct, but at 0.70 only
  2/4 and at 0.55 only 8/15 were. So a low-confidence threshold escalates almost nothing, and a high one escalates most questions.
* **With a 1.5B model as the escalation tier, routing cannot help.** The model is worse than the rules on both sets
  (51.4% vs 60.0%; 68.9% vs 91.9%), so every escalation costs accuracy. Routing pays only when the escalation model is better than
  the cheap tier on the questions it receives. A stronger model (`ROUTER_LLM=groq`, Llama 3.3 70B on Groq's free tier) might
  change that; I could not measure it without your key, and the Groq provider is tested only against a mocked HTTP transport
  (request shape, 429 back-off, token usage, error path), never against Groq itself.

## What routing is still good for

The accuracy claim is withdrawn, but the mechanisms are tested and independent of it: per-tier timeouts and circuit breakers, graceful
degradation to the rules answer when the model is down or over budget, a daily call budget for free-tier quotas, the private-question
rule, and per-response tier, token and list-price cost. Cost is not in euros yet: the local model has no price, so `cost_usd` is `null`
for it (reported as unpriced, never guessed), while a Groq call is priced at list price ($0.59 / $0.79 per 1M tokens for Llama 3.3 70B)
even though the free tier bills nothing.

Not covered: when the rules produce confident but invalid SQL (benchmark question 30), no tier escalates because the error only
shows at execution time; retrying with the model on an execution error is the natural next step.

## Repair after an execution error

Most of the small model's failures on the original set were queries the database rejected (a column that does not exist, a
wrong alias). Model providers now get one repair attempt: the model is shown its SQL and the error and the new query goes through
the same guardrail and executor (API route and benchmark; `REPAIR=0` turns it off in the benchmark). Local model, original 74
questions: 68.9% -> 73.0% (5 queries repaired). On the 35 held-out questions: 51.4% -> 51.4%, 0 repairs, because there the
failures are valid queries that answer the wrong question, which a repair prompt cannot see.
