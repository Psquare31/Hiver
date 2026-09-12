# AI Support Agent — AppleSupport

An inbound-triage agent over real Twitter customer-support conversations. For
each incoming customer message it (1) classifies intent, (2) drafts a reply
grounded in how the brand historically resolved similar issues, and (3) decides
auto-handle vs escalate **with a stated reason**.

The brief says the proof matters more than the system. This repo is built that
way: the evaluation machinery is the deliverable, and a good deal of it exists
specifically to attack its own headline number.

---

## Reproduce the headline results (no API keys, under 15 minutes)

```bash
make setup       # install dependencies
make data        # download corpus + build artefacts   (~5 min, one-off, 516MB)
make reproduce   # regenerate every table, offline     (~2 min)
```

`make reproduce` runs with `LLM_OFFLINE=1`, replaying a **committed response
cache**. Any cache miss is a hard error, not a silent network call — so you get
either the exact published numbers or a clear explanation, never a half-live run
that quietly differs from this README.

To confirm the cache is genuine rather than something you have to take on trust:

```bash
make verify-cache SAMPLE=20   # re-issues 20 cached prompts live and diffs
```

Prompts are stored alongside every cached response, so you can also just read
what each model was asked.

**To run against the live APIs** (both free tiers, no card required):

```bash
cp .env.example .env    # add GEMINI_API_KEY and GROQ_API_KEY
make live               # ~100 min under free-tier rate limits
```

---

## Results

### Task metrics

<!-- RESULTS:TASK -->
| system | intent macro-F1 | intent acc | auto rate | harmful auto | within budget | safe auto rate |
|---|---|---|---|---|---|---|
| trivial | 0.026 [0.02, 0.03] | 0.148 | 1.000 | 0.319 | ✗ | 0.0 |
| always_dm | 0.026 [0.02, 0.03] | 0.148 | 1.000 | 0.319 | ✗ | 0.0 |
| simple | 0.558 [0.49, 0.62] | 0.581 | 0.705 | 0.150 | ✗ | 0.0 |
| **agent** | _pending API keys_ | | | | | |
<!-- /RESULTS:TASK -->

**Headline metric — "% of traffic safely auto-handled":** the share of messages
auto-handled *while* the harmful-auto rate stays within a 5% budget. A system
that auto-handles everything scores **0**, not 100. All three baselines
currently score 0: `simple` auto-handles 70.5% of traffic but wrongly
auto-handles 15.0% of cases that needed a human, three times the budget.

### Judged reply quality

_Pending API keys._ Populated by `make live`.

### Judge–human agreement

_Pending._ Requires `golden/human_labels.csv` (see **Human labelling** below).

---

## How it works

```
message
   │
   ├─► classify.py    intent, 1 of 10, with confidence      (prompt built from taxonomy.yaml)
   │
   ├─► retrieve.py    BM25 over 39,436 past resolutions     (time-filtered per query)
   │
   ├─► draft.py       reply grounded in retrieved exemplars (or a safe holding reply)
   │
   └─► route.py       auto | escalate + reason              (rules → confidence floor → prior)
```

### Data

- **Source:** [`SunidhiSriram/twcs`](https://huggingface.co/datasets/SunidhiSriram/twcs)
  on HuggingFace — a byte-faithful mirror of the Kaggle
  `thoughtvector/customer-support-on-twitter` file. **2,811,774 rows**, exact
  match to the original. Avoids the Kaggle credential dance; schema is identical.
- **Brand:** AppleSupport — 106,623 (customer → first agent reply) pairs,
  **39,510 (37.1%) with substantive replies** — the most groundable brand in
  the corpus.
- **Unit of work:** the customer's opening message → the brand's first reply.
  Not the whole thread: that is the decision an inbound triage agent actually
  faces, and using the full thread would leak the resolution into the input.

### Why AppleSupport

Measured across the top 15 brands (`results/brand_selection.csv`):

| brand | agent tweets | substantive | handoff | groundable replies |
|---|---|---|---|---|
| **AppleSupport** | 106,860 | **37.1%** | 47.2% | **39,603** |
| SpotifyCares | 43,265 | 18.4% | 50.3% | 7,962 |
| AmazonHelp | 169,840 | 5.9% | 40.4% | 9,941 |
| Uber_Support | 56,270 | 2.2% | **75.7%** | 1,251 |
| Ask_Spectrum | 25,860 | 3.0% | 64.0% | 772 |

Volume and usability are inversely correlated. AmazonHelp is the largest support
account in the corpus and one of the least usable — its replies are almost
entirely *"Kindly share your details here: `<link>`"*, because resolution happens
in DMs the dataset never captured. AppleSupport has by far the most groundable
material. Brand is a config parameter (`BRAND` env var); switching costs no code
change, but it does cost a new taxonomy and a fresh labelling pass.

### Intent taxonomy

10 intents, derived bottom-up. TF-IDF + KMeans over 4,000 customer messages (k
swept 6–14) produced the *reading material*, not the taxonomy — silhouette
peaked at 0.023 and one cluster absorbed 52% of messages. I read cluster
exemplars and authored `config/taxonomy.yaml`, tracing each intent to observed
structure.

`battery_power` · `os_update` · `device_performance` · `connectivity` ·
`app_software` · `account_appleid` · `hardware_repair` · `feedback_complaint` ·
`acknowledgement` · `other`

`acknowledgement` was **added after labelling, not before** — clustering never
surfaced it. `other` sits at **7.6%**, under the 15% threshold the taxonomy sets
for itself and a test enforces.

---

## Golden evaluation set

**210 examples, hand-labelled** (`golden/labels_author.py` — recorded as source
so the reasoning is diffable, including the boundary calls a second annotator
could reasonably dispute).

**Sampling.** Two strata:
- `random` (80) — uniform sample of real traffic. Representative.
- `stratified` (130) — balanced across intents, with hard cases (multi-intent,
  shouting, escalation-adjacent) deliberately oversampled so per-class metrics
  are measurable at all.

Stratification uses a **model-independent keyword heuristic**. Sampling by
classifier predictions would bias the set toward what the classifier already
handles and under-represent exactly where it fails.

**The set is deliberately not representative**, so every row carries a `weight`
and every headline is reported raw *and* reweighted to the natural distribution.

| intent | count |
|---|---|
| app_software | 55 |
| device_performance | 38 |
| battery_power | 23 |
| hardware_repair | 19 |
| connectivity | 17 |
| other | 16 |
| account_appleid | 15 |
| feedback_complaint | 14 |
| os_update | 10 |
| acknowledgement | 3 |
| **route: auto / escalate** | **156 / 54** |

**14.8% of the set is a single 2017 bug** — the iOS 11 fault rendering "I" as a
boxed question mark. That concentration is a headline limitation, not trivia;
see the report.

### Human labelling

```bash
make sheet    # → golden/labelling_sheet.html
```

Open in a browser (~1 hour, progress saved automatically), then Export CSV to
`golden/human_labels.csv`.

- **Part A (40 items)** — intent + route, labelled **blind**. My labels are
  never shown; if the annotator can see the answer, agreement measures
  suggestibility rather than judgement.
- **Part B (63 presentations)** — reply quality on 4 ordinal dimensions plus the
  binary *"would you send this as-is?"*, blind to which system wrote each reply,
  order randomised, systems interleaved.
- **15 items are silently duplicated**, ≥20 positions apart. This measures
  annotator self-consistency, which is the **ceiling** on achievable judge–human
  agreement. Without it, κ = 0.6 cannot be interpreted at all.

---

## Evaluation harness

**Automated metrics** (`src/eval/metrics.py`)
- Macro-F1 for intent, not accuracy — classes range from 31 to 10 members.
- Every headline number carries a **bootstrap CI** (n=210 ⇒ roughly ±7 points).
- Routing scored **asymmetrically**: `harmful_auto_rate` and
  `unnecessary_escalation_rate` are reported separately. There is deliberately
  no combined "routing accuracy" anywhere in the codebase.

**LLM judge** (`src/eval/judge.py`, rubric in `config/rubric.yaml`)
- 4 ordinal dimensions (grounded / actionable / safe / tone) as diagnostics,
  plus the binary `send_as_is` as the headline.
- Blind to system identity.
- Explicitly told the historical reply is **not** an answer key, because ~half
  this brand's replies resolve nothing — without that instruction the judge
  measures mimicry of a non-answer.
- **Cross-family judging by default**: Gemini drafts, so Groq `gpt-oss-120b`
  scores the same replies and the delta is reported as self-preference bias.

**Agreement** (`src/eval/agreement.py`)
- Cohen's κ on the binary; Krippendorff's ordinal α on the 1–5 scales.
- Reported against the annotator's test–retest ceiling, with Landis–Koch bands
  stated so the report cannot quietly inflate them.

---

## Model arena

| role | model | provider | why |
|---|---|---|---|
| drafter | `gemini-3.6-flash` | Gemini | had free-tier quota left (see below) |
| classifier | `gemini-3.5-flash-lite` | Gemini | cheapest adequate |
| **primary judge** | `openai/gpt-oss-120b` | Groq | **different family from the drafter** |
| cross-family judge | `gemini-3.6-flash` | Gemini | independent check on the judge |
| arena | `openai/gpt-oss-20b` | Groq | speed/quality tradeoff |
| arena | `qwen/qwen3.6-27b` | Groq | third model family |
| ceiling | `gemini-3.1-pro-preview` | Gemini | **excluded — not on the free tier** |

Two things worth knowing before you trust the roster:

**Free-tier quota is per model, and far tighter than documented.** Gemini
publishes 1,500 requests/day; this key exhausted `gemini-3.8-flash` and then
`gemini-3.7-flash` after **fewer than 40 total requests**, while 3.6-flash and
the flash-lite models kept serving. The client therefore implements failover
down a configured chain, and records which model actually answered each row so
results are never attributed to a model that did not produce them.

**Pro models are not on the free tier at all** — the API reports
`limit: 0, model: gemini-3.1-pro`. A Google AI Pro subscription does not enable
this; its Cloud credit must be attached as billing on the API project. The
ceiling row is excluded rather than silently failing.

Everything runs on free tiers, so **cost is reported as tokens consumed and
equivalent price at published paid rates**, plus latency. Actual spend is zero,
and reporting zero would say nothing; token efficiency transfers to anyone
deciding what to run in production.

Rate limiting is enforced on **tokens and requests at the correct scope** —
Groq's free tier is token-bound (8K/min, 200K/day) and published per model;
Gemini's daily cap is account-wide. Daily counters persist to disk, because a
process restart that silently resets them causes a real lockout.

---

## Layout

```
config/     taxonomy.yaml · rubric.yaml · models.yaml
src/data/   download · brand selection · thread reconstruction · text
src/llm/    provider adapters · cache · rate limiting · client
src/agent/  classify · retrieve · draft · route
src/eval/   golden · metrics · judge · agreement · report · build_sheet
golden/     golden set, my labels, labelling sheet
cache/llm/  committed response cache (what makes `make reproduce` work)
results/    generated tables
```

`make test` — 21 tests covering metric arithmetic against hand-computed values,
leakage guards, cache-key behaviour, rate-limit accounting, and the regex
boundary bugs that bit twice during development.

---

## Reading order

1. **[DECISIONS.md](DECISIONS.md)** — 20 non-obvious decisions and why.
2. **[REPORT.md](REPORT.md)** — problem framing, results, failure analysis, and
   the mandatory *"what is misleading about my headline number"* section.

## Credits

- Dataset: Thought Vector, *Customer Support on Twitter* (Kaggle), via the
  `SunidhiSriram/twcs` HuggingFace mirror.
- `rank_bm25` (Okapi BM25), scikit-learn (TF-IDF, LogisticRegression, metrics),
  `krippendorff` (ordinal α).
- Landis & Koch (1977) for the κ interpretation bands.
