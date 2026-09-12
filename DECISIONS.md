# Decision log

Non-obvious choices, and why. Roughly ordered by how much each moves the
headline numbers.

---

## Data and framing

**1. Brand chosen by measuring reply substance, not tweet volume.**
`src/data/brand.py` measures, across the top 15 brands, the share of agent
replies containing real troubleshooting content versus pure channel handoffs.
Volume and usability turned out to be inversely correlated: Uber is 75.7%
handoff, Ask_Spectrum 64.0%, AmazonHelp only 5.9% substantive despite being the
largest account in the corpus. **AppleSupport has the most groundable material
of any brand — 39,510 substantive replies at 37.1%** — which is why it was
chosen. Brand is a config parameter (`BRAND` env var); switching costs no code
change, but it does cost a new taxonomy and a fresh labelling pass.

**2. The substance metric was wrong twice, and both failures were instructive.**
v1 matched redirect phrases ("dm us", "click here") and ranked AmazonHelp as
*most* groundable at 0.6% handoff. Inspection showed the opposite — Amazon says
*"Kindly share your details here: <url>"*, which no phrase list anticipates.
Enumerating ways to say "go away" is unbounded; every brand invents its own.
v2 inverted to detect substance but over-credited, because a credential request
(*"Can you DM us your account's email address?"*) is phrased as a question about
a product noun and scored as troubleshooting. v3 requires an instruction or a
diagnostic probe and explicitly demotes credential asks. Pinned by 20
hand-labelled fixtures in `tests/fixtures_replies.py`. **Trusting v1 would have
built the entire project on the least groundable brand in the dataset.**

**3. The corpus is one product event, and that limits every number.**
96% of AppleSupport traffic falls in Oct–Dec 2017, the iOS 11 release window.
More sharply: **14.8% of the 210-example golden set is a single bug** — the
iOS 11 autocorrect fault rendering "I" as a boxed question mark. A classifier
scoring well here is partly scoring well on one 2017 defect. This is stated in
the report rather than left for a reviewer to find.

**4. Unit of work is (first customer message → first brand reply).**
Not the whole thread. That is the decision an inbound triage agent actually
faces, and using the full thread would leak the resolution into the input.

**5. Thread reconstruction is a join, not a loop.**
The row-wise version with per-row `.loc` lookups was fine on Spotify (41k pairs)
and did not finish in 10 minutes on Apple (2.5× the volume). Verified by
reproducing Spotify's 41,585 pairs exactly before switching.

## Taxonomy and labelling

**6. Taxonomy derived bottom-up, with clustering as a reading aid only.**
TF-IDF + KMeans over 4,000 messages, k swept 6–14. Silhouette peaked at 0.023
and one cluster absorbed 52% of messages, so the clusters are unusable as
labels — but readable as structure. Each of the 10 intents traces to observed
clusters via `derived_from` in `config/taxonomy.yaml`.

**7. `acknowledgement` was added after labelling, not before.**
Clustering never surfaced it. It only became visible when labelling put a
coherent group of thanks/praise/closing messages under one reason code. On the
Spotify pass this dropped `other` from 15.2% to 9.5%; the class carried over to
Apple. The taxonomy file commits to `other` < 15%, and a test enforces it.

**8. `device_performance` vs `app_software` is the deliberate hard boundary.**
A named app misbehaving is `app_software`; the whole device freezing or
slowing is `device_performance`. These are separated because the *remedy*
differs, and the boundary is documented in `golden/labels_author.py` as the
largest expected source of annotator disagreement.

**9. Labels are recorded as source, including the calls I might have got wrong.**
`golden/labels_author.py` documents every boundary rule and lists the specific
rows where I overrode the escalation rules, so a reviewer can disagree with a
specific decision rather than with an opaque CSV.

**10. Escalation rule gaps are recorded as measured failures, not fixed quietly.**
While labelling I found messages the rules miss: *"practically on fire"* (the
list has "caught fire"), *"electric shock"* (absent entirely), *"not supposed to
inflate"* (the list has "swelling"). And one false positive: *"Third time today
my phone freezes"* fires `repeat_contact`, which counts crashes, not support
contacts. I labelled what a careful human would decide and left the rules
untouched, so the gap shows up as a number in the failure analysis instead of
disappearing into a regex.

## Evaluation design

**11. Headline metric is "% of traffic safely auto-handled".**
Auto-handled rate, creditable only while harmful-auto stays within a 5% budget.
A system that auto-handles everything scores **0**, not 100.

**12. Routing is scored asymmetrically and never averaged.**
`harmful_auto_rate` and `unnecessary_escalation_rate` are reported separately.
There is deliberately no combined "routing accuracy" anywhere in the codebase.

**13. Three baselines, not the two the brief asks for.**
The third is `always_dm`, which replies with a channel handoff to everything —
using AppleSupport's own most common phrasing, because **47.3% of Apple's real
replies are handoffs**. A clumsy strawman would be easy for the judge to mark
down; the point is that this baseline is genuinely hard to distinguish from what
the brand really does. Any judge scoring it well is telling us the metric
rewards non-answers.

**14. Golden set is deliberately unrepresentative, and carries weights.**
80 uniform-random rows plus 130 stratified/hard-oversampled. Every row has a
`weight`; every headline is reported raw *and* reweighted to the natural
distribution. Stratification uses a model-independent keyword heuristic —
sampling by classifier predictions would bias the set toward what the classifier
already handles.

**15. 15 silent duplicates in the labelling sheet, ≥20 items apart.**
Test–retest self-consistency is the *ceiling* on achievable judge–human
agreement; without it κ = 0.6 is uninterpretable. Separation matters: an item
repeated a few positions later is remembered, which measures recall rather than
judgement. An earlier version allowed a gap of 3.

**16. The headline agreement metric is a binary, not the 1–5 scales.**
Humans agree poorly on 3-vs-4 on any subjective scale. `send_as_is` is the real
operational decision and is what κ is computed on; the ordinal dimensions stay
as diagnostics with Krippendorff's ordinal α.

**17. The judge is told the historical reply is NOT an answer key.**
It sees what the brand actually said, with the 47% handoff rate stated
explicitly. Without that, the judge measures mimicry of a non-answer.

**18. Cross-family judging by default.**
Gemini drafts, so a Groq model judges the same replies and the delta is reported
as self-preference bias.

## Retrieval

**19. Retrieval is time-filtered per query, not by a global cutoff.**
A global cutoff at the earliest golden timestamp left **90 usable pairs out of
thousands** — the corpus spans ~2 months and the golden set covers all of it.
Per-query filtering keeps the full 39,436-pair index while preserving the
guarantee exactly, verified empirically over all 210 rows rather than asserted.

**20. Only substantive replies enter the index.**
Indexing handoffs would teach the drafter that the answer to everything is
"send us a DM" — and because handoffs dominate the ground truth, the judge would
reward it.

**21. BM25 is the default, but dense retrieval was built and measured.**
"We didn't try embeddings" is weaker than "we tried them and here is what they
bought". `src/agent/dense.py` embeds all 39,436 pairs on the local GPU
(MiniLM-L6-v2, ~8 seconds) and adds an RRF hybrid. **The two retrievers overlap
on only 5% of retrieved evidence (Jaccard 0.053)** — they surface almost
entirely different threads, which makes retriever choice a far larger decision
than it appears. BM25 remains the default because its evidence is inspectable:
a reviewer can see exactly which historical reply drove a draft.

## Infrastructure — bugs that would have silently corrupted results

**22. Gemini thinking tokens were destroying the output.**
Gemini 3.x Flash draws thinking tokens from the *same* `max_output_tokens`
budget as the answer. Measured on the classifier prompt: at `max_output_tokens=100`,
92 tokens went to thinking and 4 to the answer, returning `"format?\n    "`; at
300, it was 287 vs 9. Both come back as HTTP 200 with `finish_reason=MAX_TOKENS`
and truncated garbage — so **every classification would have fallen back to
`other` at zero confidence, and the run would have looked like a model-quality
problem rather than a config bug**. Fixed with `thinking_budget=0`, which
returns clean JSON in 13–27 tokens. Support is not uniform (3.7-flash requires
the parameter, 3.5-flash-lite rejects it with a 400), so it is probed per model
and memoised.

**23. gpt-oss returns its answer in a separate field.**
Groq's gpt-oss models emit chain-of-thought into `message.reasoning` and only
fill `content` afterwards. With a small `max_tokens` they return
`finish_reason=length` and an **empty** `content` — which would have emptied
every cross-family judge verdict without raising. Fixed with
`reasoning_effort="low"`, which also cut completion tokens ~3× (29 vs 95).
Qwen has the same problem in a different shape: inline `<think>` blocks,
stripped in `_extract_json`.

**24. Truncation is a loud error — but escalated, not retried.**
An empty or truncated answer that cost tokens is a failure, and caching it
silently would corrupt every downstream metric. The first fix made truncation a
*retryable* error, which was wrong in an instructive way: truncation is
**deterministic**, so the same prompt at the same ceiling truncates every time.
Each draft then burned five identical retries with exponential backoff before
failing over, and **the full run appeared to hang while making no progress** —
13 cached calls in 45 seconds of wall clock. The retry that can actually succeed
is a larger ceiling, so the provider now escalates the budget (×4, up to three
attempts) and only then declares a non-retryable failure so the client fails
over to a different model.

**24a. Thinking is ~95% of billed output tokens on this workload.**
Measured on the drafting prompt: ~880-970 completion tokens for a ~50-token
reply. A single fixed reserve was not enough either — gemini-3.7-flash spent
~300 thinking tokens but gemini-3.6-flash blew through 812 on the same prompt.
This is why the cost table counts `thoughts_token_count`: reporting only the
visible answer would understate Gemini's real output cost by roughly 20x.

**25. The Gemini SDK hides real errors inside a retry wrapper.**
It retries internally via tenacity and raises `RetryError[<Future ... raised
ServerError>]`, which is *not* an `APIError` subclass — so an `except APIError`
handler misses it and a transient 5xx escapes as an uncaught exception mid-run
(it killed the first full run). Errors are now unwrapped to the root cause
before being classified retryable vs fatal.

**26. Free-tier quota is per model, and newest ≠ most available.**
`gemini-3.8-flash` returns 429 RESOURCE_EXHAUSTED on this key while 3.7, 3.6 and
3.5 serve normally. The drafter/judge therefore runs on `gemini-3.7-flash`.

**27. Pro models are not on the free tier at all.**
The API reports `limit: 0, model: gemini-3.1-pro` for free-tier input tokens. A
Google AI Pro *subscription* does not enable this — its Cloud credit has to be
attached as billing on the API project. The ceiling row is therefore excluded
from the default arena rather than silently failing.

**28. Rate limits are enforced on tokens and requests, at the right scope.**
Groq's free tier is token-bound (8K/min, 200K/day) and published per model;
Gemini's request cap is account-wide. Limiter scope is configured per provider.
Daily counters persist to disk, because a process restart that resets them
causes a real lockout rather than a soft backoff.

**29. The response cache is committed, and stores prompts.**
Free-tier limits make a cold run far longer than the brief's 15-minute
reproduce budget. `make reproduce` replays offline with `LLM_OFFLINE=1`, where
any cache miss is a hard error rather than a silent network call. Prompts are
stored (~4KB/entry) so `make verify-cache` can re-issue calls and diff — an
unauditable cache behind a headline number is worthless.

**30. Cost is reported as tokens and equivalent paid rates, not spend.**
We pay nothing, so reporting spend reports zero and says nothing. Token
efficiency and published-rate equivalents transfer to someone choosing what to
run in production.

**31. Classifier prompt compressed to ~580 tokens, with the cost measured.**
The full taxonomy with examples runs ~1,220 tokens, which at Groq's 200K/day
allows ~110 classifications — not enough to score 210 rows.
`verbose_taxonomy=True` restores the examples so the ablation can measure what
the compression costs rather than assuming it is free.
