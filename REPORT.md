# AI Support Agent for AppleSupport — Report

*Companion documents: [DECISIONS.md](DECISIONS.md) (31 decisions), [README.md](README.md) (reproduce in <15 min).*

---

## 1. Problem framing

### What "good" means for this brand

AppleSupport's Twitter channel is **triage, not resolution**. 47.3% of its
replies are channel handoffs — *"Send us a DM and we'll take a closer look"* —
because the actual fix needs account access, a serial number, or a repair
booking, none of which happen in public. Of the rest, the dominant pattern is a
**diagnostic probe**: *"Can you tell us which version of iOS is installed in
Settings > General?"*

So a good agent here is not one that resolves issues. It is one that:

1. **Routes correctly**, with the cost asymmetry respected. Auto-handling a
   swollen battery or a compromised Apple ID is a serious failure; escalating a
   question a bot could answer costs an agent two minutes. These are not
   symmetric errors and are never averaged into one number in this repo.
2. **Asks the right next question**, grounded in what this brand has actually
   asked before in the same situation.
3. **Says why**, on every routing decision. An escalation a human cannot act on
   is not triage.

The headline metric follows from that: **% of traffic safely auto-handled** —
the auto-handled rate, creditable only while the harmful-auto rate stays within
a 5% budget. A system that auto-handles everything scores **0**, not 100.

### What I chose not to build

- **Multi-turn dialogue.** The unit is (first customer message → first brand
  reply). That is the decision inbound triage actually faces; using the full
  thread leaks the resolution into the input.
- **Actual resolution.** No account lookups, no order status, no device
  history. Without those systems the honest ceiling is a good first reply, and
  pretending otherwise would make every metric optimistic.
- **A fine-tuned classifier.** 210 labelled examples cannot support it, and it
  would compete for the same budget as the evaluation, which is the deliverable.
- **Dense retrieval as the default** — though I built and measured it (§6).
- **Banking77.** Different domain, different label space; transfer claims from
  it would not survive scrutiny.

---

## 2. The system

```
message → classify (10 intents + confidence)
        → retrieve (BM25 over 39,436 past resolutions, time-filtered per query)
        → draft    (grounded in retrieved exemplars, or a safe holding reply)
        → route    (rules → confidence floor → intent prior, always with a reason)
```

Escalation rules are **one-directional**: they can force an escalation, never
force an auto-handle. A confident model cannot override the safety or refund
rules. Confidence below 0.55 escalates regardless of intent, because an
unreliable classification means the reply is being drafted for the wrong problem.

---

## 3. Data and taxonomy

**Corpus.** `SunidhiSriram/twcs` on HuggingFace — a byte-faithful mirror of the
Kaggle file, 2,811,774 rows, verified exactly. AppleSupport contributes 106,623
(customer → first reply) pairs, of which **39,510 (37.1%) contain substantive
troubleshooting**.

**Why AppleSupport.** Chosen by measurement, not volume — the two are inversely
correlated in this dataset:

| brand | agent tweets | substantive | handoff | groundable replies |
|---|---|---|---|---|
| **AppleSupport** | 106,860 | **37.1%** | 47.2% | **39,603** |
| SpotifyCares | 43,265 | 18.4% | 50.3% | 7,962 |
| AmazonHelp | 169,840 | 5.9% | 40.4% | 9,941 |
| Uber_Support | 56,270 | 2.2% | **75.7%** | 1,251 |
| Ask_Spectrum | 25,860 | 3.0% | 64.0% | 772 |

AmazonHelp is the largest support account in the corpus and one of the least
usable: its replies are almost entirely *"Kindly share your details here:
`<link>`"*. Full table in `results/brand_selection.csv`.

**Taxonomy.** 10 intents, derived bottom-up. TF-IDF + KMeans over 4,000 messages
(k swept 6–14) produced *reading material*, not labels — silhouette peaked at
0.023 and one cluster absorbed 52% of messages. I read cluster exemplars and
authored `config/taxonomy.yaml`, tracing each intent to observed structure.

`battery_power` · `os_update` · `device_performance` · `connectivity` ·
`app_software` · `account_appleid` · `hardware_repair` · `feedback_complaint` ·
`acknowledgement` · `other`

`acknowledgement` was **added after labelling**, not before — clustering never
surfaced it. `other` sits at 7.6%, under the 15% threshold the taxonomy file
sets for itself and a test enforces.

---

## 4. Golden set

**210 hand-labelled examples**, recorded as source in
`golden/labels_author.py` so every boundary call is reviewable and diffable.

| | |
|---|---|
| `random` stratum | 80 (uniform, representative) |
| `stratified` stratum | 130 (class-balanced, hard cases oversampled) |
| hard cases | 73 (35%) |
| route split | 156 auto / 54 escalate |
| `other` rate | 7.6% |

Stratification uses a **model-independent keyword heuristic** — sampling by
classifier predictions would bias the set toward what the classifier already
handles and under-represent exactly where it fails. Because the set is
deliberately unrepresentative, every row carries a `weight` and every headline
is reported raw *and* reweighted to the natural distribution.

**Escalation reasons** (54 rows): service_booking 17, repeat_contact 10,
account_lookup 6, safety_hazard 5, churn_risk 5, data_loss 4, money_dispute 3,
account_security 3, human_requested 1.

---

## 5. Results

<!-- RESULTS:START -->
*Populated by `make live` / `make reproduce`. See `results/summary_table.csv`.*
<!-- RESULTS:END -->

---

## 6. Retrieval: BM25 vs dense vs hybrid

Rather than assert that lexical retrieval was sufficient, I built the dense
alternative and measured it. All 39,436 substantive pairs embedded on the local
GPU (MiniLM-L6-v2, ~8 seconds), plus a reciprocal-rank-fusion hybrid.

| | exemplars/query | zero-evidence queries |
|---|---|---|
| BM25 | 3.94 | 0.5% |
| Dense | 3.90 | 1.0% |
| Hybrid (RRF) | 3.98 | — |
| **Overlap** | **0.34 of ~4** | **Jaccard 0.053** |

**The two retrievers agree on 5% of the evidence they surface.** They are
finding almost entirely different threads for the same query. That makes
retriever choice a much larger decision than it appears — and it means any
grounding claim is conditional on a choice that was never tuned. BM25 remains
the default because its evidence is inspectable: a reviewer can see exactly
which historical reply drove a draft.

---

## 7. Failure analysis

<!-- FAILURES:START -->
*Populated after the judged run.*
<!-- FAILURES:END -->

### Failure mode 0 — escalation rules miss real hazards (found during labelling)

Independent of any model, the hand-written escalation regexes miss phrasings a
human escalates immediately:

| message | why it slips through |
|---|---|
| *"Any reason our laptop should be practically **on fire** after 20 mins?"* | trigger list has `caught fire` |
| *"I got an **electric shock** from your headphones"* | absent from the safety list entirely |
| *"it's not supposed to **inflate** like this just sitting on my desk"* | list has `swelling`/`bulging` |
| *"**WHERE ARE MY PHOTOS**"* | data-loss list expects `photos are gone` |

And one false positive in the other direction: *"**Third time today** my phone
freezes"* fires `repeat_contact`, which is meant to count *support contacts*,
not crashes.

I left the rules untouched and labelled what a careful human would decide, so
the gap appears as a measured number rather than disappearing into a regex.
`tests/test_pipeline.py::test_known_rule_gaps_are_still_gaps` pins it, so the
gap cannot close silently without this section changing too.

---

## 8. What is misleading about my headline number

**This section is mandatory in the brief, and it is the most useful thing here.**

**1. 14.8% of the golden set is a single 2017 bug.** The iOS 11 autocorrect
fault that rendered "I" as a boxed question mark accounts for 31 of 210
examples. It has one canonical answer. A classifier that learns it looks
competent on a seventh of the benchmark while having learned one string.

**2. The corpus is one product event.** 96% of AppleSupport traffic here falls
in Oct–Dec 2017, the iOS 11 release window. Intent priors, retrieval evidence,
and the taxonomy itself are all fitted to a crisis period. A normal support
month looks different, and nothing here measures that.

**3. The golden set is deliberately unrepresentative.** Hard cases are
oversampled on purpose. The raw number understates live performance and the
weighted number is an estimate built on a keyword-heuristic prior, not a true
traffic distribution. Both are reported; neither is "the" answer.

**4. n = 210 means roughly ±7 points on any proportion.** Every headline
carries a bootstrap CI. Differences smaller than the interval are not results.

**5. The judge is an LLM, and its ceiling is a single annotator.** Judge–human
agreement is bounded by that annotator's own test–retest consistency. One
annotator means no inter-human agreement is measurable at all, so "how hard is
this task for humans" is genuinely unknown.

**6. The ground truth is not a gold standard.** 47.3% of Apple's real replies
are handoffs. Where the reference resolves nothing, a metric anchored on it
partly rewards non-answers — which is exactly why the `always_dm` baseline
exists, and why its judge score should be read as a measurement of the metric,
not of that baseline.

**7. Grounding is conditional on an untuned retriever choice.** BM25 and dense
retrieval overlap on 5% of evidence (§6). "Grounded in brand precedent" means
"grounded in whatever BM25 surfaced".

**8. Auto-handle rates are measured on messages that reached a human.** Every
thread in this corpus got a reply. Traffic that was ignored, resolved in DMs, or
never tweeted is invisible. The denominator is not live traffic.

**9. Free-tier constraints shaped the system, not just the schedule.** The
classifier prompt is compressed to ~580 tokens to fit Groq's daily budget; the
ceiling model is absent because Pro is not on the free tier. Some of what is
measured here is the free tier, not the models.

---

## 9. What I'd do next, with one more week

1. **A second annotator on the same 210 rows.** The single biggest weakness is
   that judge quality is bounded by one person's consistency with no
   inter-human agreement to calibrate against. This is a day's work and would
   upgrade every agreement claim in the report.
2. **De-duplicate the iOS 11 "I" bug and re-measure.** Report the headline with
   that cluster capped at its natural rate. If performance falls sharply, the
   number was mostly one string.
3. **Tune the retriever against the drafts it produces.** §6 shows the choice
   matters and was never optimised. Sweep BM25 vs dense vs hybrid *by judge
   score on the drafts*, not by retrieval overlap.
4. **A second brand as a transfer test.** The pipeline is brand-parameterised
   and the Spotify artefacts still exist. Running it unchanged would show
   whether the design generalises or is fitted to Apple's iOS 11 window.
5. **Calibrate the confidence floor.** 0.55 was chosen a priori. With the
   golden set it can be set where harmful-auto actually crosses the budget.
6. **Cost the escalation asymmetry properly.** Replace the fixed 5% budget with
   an explicit cost ratio agreed with the support team, and report the
   auto-handle rate as a curve over that ratio rather than at one point.
