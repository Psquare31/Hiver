# Decision log

Non-obvious choices, and why. Ordered roughly by how much each one moves the
headline numbers.

---

**1. Brand chosen by measuring reply substance, not tweet volume.**
Volume and usability are inversely correlated in this corpus. `src/data/brand.py`
measures, for the top 15 brands, the share of agent replies containing real
troubleshooting content versus pure channel handoffs. Uber is 75.7% handoff,
T-Mobile 65.8%. SpotifyCares has 7,962 groundable replies with a bounded intent
space. AppleSupport actually has more groundable replies (39,603) — this
contradicted my prior, and is recorded rather than buried — but its intent space
spans the entire Apple product line, which makes "a small set of intents"
indefensible. Brand is a config parameter; switching costs one line.

**2. The substance metric was wrong twice, and the failures were instructive.**
v1 matched redirect phrases ("dm us", "click here") and scored AmazonHelp at
0.6% handoff — apparently the most groundable brand in the corpus. Inspection
showed the opposite: Amazon says *"Kindly share your details here: <url>"*,
which no phrase list anticipates. Enumerating ways to say "go away" is
unbounded. v2 inverted to detect substance, but over-credited: Spotify's
canonical DM wall *"Can you DM us your account's email address?"* is a question
about a product noun and scored as troubleshooting. v3 requires an instruction
or a diagnostic probe and explicitly demotes credential requests. Pinned by 20
hand-labelled fixtures in `tests/fixtures_replies.py`. **Had I trusted v1, the
entire project would have been built on AmazonHelp — the worst possible choice.**

**3. Three baselines, not the two the brief asks for.**
The third is `always_dm`: replies "please DM us" to everything. Since ~50% of
Spotify's real replies are handoffs, this baseline imitates modal brand
behaviour. Any judge that scores it well is revealing that the metric rewards
non-answers. It exists to attack my own headline number, and it is the sharpest
evidence in the misleading-number section.

**4. Headline metric is "% of traffic safely auto-handled", not accuracy.**
Auto-handled rate, creditable only while harmful-auto stays ≤5%. A system that
auto-handles everything scores 0, not 100. Task-level metrics sit underneath as
diagnostics. One number invites attack, which is the point.

**5. Routing is scored asymmetrically and never averaged.**
Auto-handling a compromised account is not the same kind of error as escalating
a question a bot could answer. `harmful_auto_rate` and
`unnecessary_escalation_rate` are reported separately; there is deliberately no
combined "routing accuracy" anywhere in the codebase.

**6. Escalation rules are one-directional.**
Rules can force an escalation, never force an auto-handle. A confident model
cannot override the security or refund rules. Confidence below 0.55 escalates
regardless of intent — an unreliable classification means the reply is being
drafted for the wrong problem.

**7. Golden-set stratification uses a model-independent heuristic.**
Sampling by classifier predictions would bias the set toward cases the
classifier already handles confidently and under-represent exactly where it
fails. Stratification uses keyword rules derived from the taxonomy; the system
under test never touches sampling.

**8. The golden set is deliberately unrepresentative, and carries weights.**
80 uniform-random rows plus 130 stratified/hard-oversampled rows. Raw numbers
overstate difficulty. Every row has a `weight`, and every headline is reported
raw *and* reweighted to the natural distribution. The gap is a finding, not an
embarrassment.

**9. The taxonomy changed after labelling, which is the process working.**
KMeans (silhouette peaked at 0.02, one cluster absorbed 55% of messages) never
surfaced an `acknowledgement` class — thanks/praise/closing messages were buried
in the catch-all. Labelling made it visible: 12 of 32 `other` rows shared one
reason code. Splitting it out dropped `other` from 15.2% to 9.5%, back under the
coverage threshold the taxonomy file sets for itself.

**10. Retrieval is time-filtered per query, not by a global cutoff.**
A global cutoff at the earliest golden timestamp left **90 usable pairs out of
7,652** — the corpus spans ~2 months and the golden set covers all of it.
Filtering per query keeps the full 7,607-pair index while preserving the
guarantee exactly. Verified empirically over all 210 rows, not just asserted.
Honest consequence: early-period messages retrieve less precedent, because less
history existed.

**11. Only substantive replies enter the retrieval index.**
Indexing handoffs would teach the drafter that the answer to everything is
"please DM us" — and because handoffs dominate the ground truth, the judge would
reward it. This single filter is what makes grounding meaningful.

**12. BM25, not embeddings.**
No embedding API keeps the pipeline free and offline-reproducible, but the real
reason is inspectability: a reviewer can see exactly which historical reply drove
a draft. Dense retrieval is in "what I'd do next", not smuggled in.

**13. The judge is told the historical reply is NOT an answer key.**
It sees what the brand actually said, with an explicit instruction that ~half of
this brand's replies resolve nothing. Without that, the judge measures mimicry
of a non-answer rather than quality.

**14. Cross-family judging by default.**
Gemini drafts and Gemini judges is self-preference bias. Groq `gpt-oss-120b`
scores the same replies and the delta is reported as a result.

**15. 15 silent duplicates in the human labelling sheet, ≥20 items apart.**
Test-retest self-consistency is the *ceiling* on achievable judge-human
agreement. Without it, κ=0.6 is uninterpretable. Separation matters: an item
repeated a few positions later is remembered, which measures recall rather than
judgement and inflates the ceiling.

**16. The headline agreement metric is a binary, not the 1-5 scales.**
Humans agree poorly on 3-vs-4 on any subjective scale; building the headline on
that manufactures disagreement about the scale rather than the reply.
`send_as_is` is the actual operational decision and is what κ is computed on.
The ordinal dimensions remain as diagnostics.

**17. The response cache is committed, and stores prompts.**
Free-tier limits make a cold run ~100 minutes against a 15-minute reproduce
requirement. `make reproduce` replays offline with `LLM_OFFLINE=1`, where any
cache miss is a hard error rather than a silent network call. Prompts are stored
(~4KB/entry) so `make verify-cache` can re-issue calls and diff — an unauditable
cache behind a headline number is worthless.

**18. Rate limits are enforced on tokens and requests, per correct scope.**
Groq's free tier is token-bound (8K/min, 200K/day), not request-bound; an
RPM-only limiter passes the check then collects 429s all day. Groq publishes
limits per model, Gemini's daily cap is account-wide — so limiter scope is
configured per provider. Daily counters persist to disk, because a process
restart that resets them causes a real lockout.

**19. Cost is reported as tokens and equivalent paid rates, not spend.**
We pay nothing, so reporting actual spend reports zero and says nothing. Tokens
and published-rate equivalents are what transfer to someone choosing what to run
in production.

**20. Classifier prompt is compressed to ~580 tokens, and the cost is measured.**
The full taxonomy with examples runs ~1,220 tokens, which at Groq's 200K/day
allows ~110 classifications — not enough to score 210 rows. The compact
rendering fits. `verbose_taxonomy=True` restores the examples so the ablation
can measure what compression costs rather than assuming it is free.
