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

All numbers from `make reproduce`; raw tables in `results/`.

### Task metrics (n = 210, bootstrap 95% CIs)

| system | intent macro-F1 | intent acc | auto rate | harmful auto | safe auto rate |
|---|---|---|---|---|---|
| **agent** | **0.652** [0.55, 0.71] | 0.671 | 0.776 | **0.073** | **0.0** |
| simple | 0.312 [0.25, 0.37] | 0.395 | 0.788 | 0.126 | 0.0 |
| always_dm | 0.042 [0.03, 0.05] | 0.262 | 1.000 | 0.221 | 0.0 |
| trivial | 0.042 [0.03, 0.05] | 0.262 | 1.000 | 0.221 | 0.0 |

The agent roughly **doubles macro-F1 over the simple baseline** (0.652 vs 0.312,
non-overlapping CIs) and **halves the harmful-auto rate** (7.3% vs 12.6%).

**And the headline metric is 0 for every system, including the agent.** Nothing
meets the 5% harm budget at the default operating point. That is the honest
result: on this evidence the system is not deployable as an autonomous
responder, and reporting a flattering number here would require quietly
loosening the budget.

### The operating point matters more than the model

The confidence floor was set a priori at 0.55. Sweeping it
(`results/threshold_sweep.csv`) shows what is actually purchasable:

| confidence floor | auto rate | harmful auto | harmful CI upper | within budget |
|---|---|---|---|---|
| 0.00 - 0.40 | 0.781 | 0.073 | 0.111 | no |
| 0.55 (default) | 0.776 | 0.073 | 0.111 | no |
| 0.90 | 0.667 | 0.069 | 0.106 | no |
| **0.95** | **0.466** | **0.037** | 0.063 | **point estimate only** |
| 1.00 | 0.000 | 0.000 | 0.000 | trivially |

So the defensible claim is: **the agent can auto-handle roughly 47% of traffic
at an estimated 3.7% harmful-auto rate** - but at n=210 the confidence interval
reaches 6.3%, so the 5% budget **cannot be demonstrated, only estimated**. No
floor short of "escalate everything" is safe once the CI upper bound is used.

Two things this curve exposes:

**The floor is nearly inert below 0.90.** Harmful-auto sits flat at 7.3% from
0.0 to 0.90, because the classifier is badly calibrated: 82% of predictions come
back at confidence >= 0.92. The signal is not absent - accuracy does rise across
bands (0.48 at <=0.8, 0.79 at >0.94) - but it is compressed into the top of the
range, so a threshold has almost nothing to cut on.

**93% of the agent's drafts came from the weakest model.** Free-tier quota
exhaustion pushed 195 of 210 drafts onto `gemini-3.5-flash-lite`, with only 15
on `gemini-3.7-flash`. The "agent" row is therefore not a single system, and the
serving model is recorded per row precisely so this cannot be glossed over.

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

Top five modes, with real examples from `results/predictions.parquet`.

### 1. Conversational escalation signals are invisible to the agent (11 of 20 harmful autos)

The single biggest source of *dangerous* errors. Of 20 cases where the agent
auto-handled something a human needed to take, **7 were `repeat_contact` and 4
were `churn_risk`** - customers signalling that support has already failed them,
or that they are leaving. The rules catch CATEGORY escalations (Apple ID,
hardware, safety) and miss CONVERSATIONAL ones.

| customer | agent reply |
|---|---|
| *"worst support ever. Expected more from #apple #EpicFail"* | *"We'd like to help. Send us a DM and let us know which iOS version you're running currently."* |
| *"I was chatting with a customer service representative for 81 minutes before she could transfer me to a senior rep who then accidentally ended the chat"* | *"We're glad to help you. DM us using the link below, and detail what issues you're having."* |
| *"you replaced my airpods under warranty (sincerely, thanks) but I am still having daily issues"* | *"We'd love to help you get your iPhone working in the way you expect. DM us what issues you're seeing."* |

**Hypothesis.** The classifier is asked what the customer *wants*, and answers
from the stated topic. "Worst support ever" has no topic, so it gets routed on
whatever product noun is nearby and receives a diagnostic question - which is
the worst possible reply to someone complaining about support itself. The fix is
not a better classifier: it is a separate conversational-state signal
(has-been-failed-before, is-threatening-to-leave) feeding the router
independently of intent.

### 2. Fault vs trigger: `device_performance` -> `os_update` (6 cases)

*"I installed ios 11.1 and since, my phone froze and kept endless reboot"*
*"Since updating to iOS 11.1.1, I've had phone switch off, app crashing"*

The taxonomy says `os_update` is for when the update **is** the subject, and
`device_performance` for the fault. These messages name the update as the
trigger and the freeze as the problem. I labelled the fault; the model labelled
the trigger. **This is a genuine taxonomy weakness, not purely a model error** -
a second annotator could reasonably side with the model, which is exactly why
the boundary is documented in `golden/labels_author.py`.

### 3. `app_software` leaks into `other` and `feedback_complaint` (12 cases)

The largest intent class (55 of 210) bleeds in both directions. Messages about
the iOS 11 "I" bug phrased as venting (*"yall got my phone thinkin a letter I is
an emoji, ios 11 is dick"*) get read as complaints rather than as the specific
known bug they are. The consequence is operational: `feedback_complaint` routes
to a generic acknowledgement, while `app_software` would route to a documented
workaround.

### 4. Escalation rules miss real hazards (found during labelling, model-independent)

| message | why it slips through |
|---|---|
| *"Any reason our laptop should be practically **on fire** after 20 mins?"* | trigger list has `caught fire` |
| *"I got an **electric shock** from your headphones"* | absent from the safety list entirely |
| *"it's not supposed to **inflate** like this just sitting on my desk"* | list has `swelling`/`bulging` |
| *"**WHERE ARE MY PHOTOS**"* | data-loss list expects `photos are gone` |

And one false positive the other way: *"**Third time today** my phone freezes"*
fires `repeat_contact`, which is meant to count support contacts, not crashes.

The rules were left untouched and I labelled what a careful human would decide,
so the gap appears as a number rather than disappearing into a regex.
`tests/test_pipeline.py::test_known_rule_gaps_are_still_gaps` pins it so it
cannot close silently without this section changing too.

### 5. Confidence is compressed, so the safety threshold has nothing to cut on

82% of predictions come back at confidence >= 0.92, and the harmful-auto rate is
flat from floor 0.0 to 0.90. The signal is real but crushed into the top of the
range. **Hypothesis:** asking a model for "the probability a careful annotator
would agree" invites a fluent-sounding number, not a calibrated one - it has no
feedback signal to calibrate against. A margin-based proxy (asking for a
ranked top-2 and using the gap) or an ensemble disagreement rate would likely
discriminate far better, and costs one extra field in the same call.

---

## 8. What is misleading about my headline number

**This section is mandatory in the brief, and it is the most useful thing here.**

**0. The headline number is 0, and even that is generous.** No system meets the
5% harm budget. The best defensible operating point (floor 0.95) auto-handles
47% at an estimated 3.7% harmful-auto - but its CI reaches 6.3%, so **at n=210
the budget cannot be demonstrated, only estimated**. If I reported "47% safely
auto-handled" as the headline, the word doing the most work would be "safely",
and it would not be supported.

**1. The "agent" is not one system.** Free-tier quota exhaustion pushed 195 of
210 drafts onto `gemini-3.5-flash-lite` and only 15 onto `gemini-3.7-flash`. The
row labelled "agent" is mostly the weakest model in the roster. Whether that
helps or hurts the numbers is unmeasured; it is recorded per row so a reader can
check rather than trust.

**2. 14.8% of the golden set is a single 2017 bug.** The iOS 11 fault rendering
"I" as a boxed question mark is 31 of 210 examples and has one canonical answer.
A classifier that learns that one string looks competent on a seventh of the
benchmark.

**3. The corpus is one product event.** 96% of AppleSupport traffic here falls
in Oct-Dec 2017, the iOS 11 release window. Intent priors, retrieval evidence
and the taxonomy are all fitted to a crisis period. Nothing here measures a
normal support month.

**4. The golden set is deliberately unrepresentative.** Hard cases are
oversampled on purpose, so the raw number understates live performance - and the
weighted number is built on a keyword-heuristic prior, not a measured traffic
distribution. Both are reported; neither is "the" answer.

**5. n = 210 means roughly +/-7 points on any proportion.** The agent-vs-simple
macro-F1 gap (0.652 vs 0.312) survives that comfortably. The safety claim does
not. Differences smaller than the interval are not results.

**6. Intent accuracy and routing safety are not the same achievement.** The
agent doubles macro-F1 over the simple baseline, and that improvement does NOT
translate into passing the harm budget. Leading with 0.652 would imply a
competence the routing numbers do not support.

**7. The ground truth is not a gold standard.** 47.3% of Apple's real replies
are channel handoffs. Where the reference resolves nothing, a metric anchored on
it partly rewards non-answers - which is why `always_dm` exists, and why its
judge score should be read as a measurement of the metric rather than of that
baseline.

**8. Grounding is conditional on an untuned retriever choice.** BM25 and dense
retrieval overlap on 5% of retrieved evidence (section 6). "Grounded in brand
precedent" means "grounded in whatever BM25 surfaced", and that choice was never
optimised against reply quality.

**9. Auto-handle rates are measured on messages that reached a human.** Every
thread in this corpus got a reply. Traffic that was ignored, resolved in DMs, or
never tweeted is invisible, so the denominator is not live traffic.

**10. The judge is an LLM and, on this run, from the same family as the
drafter.** Groq's free tier allows ~171 judge calls/day against the 840 needed,
so Gemini had to judge Gemini's drafts. The cross-family delta is reported
rather than asserted away, but the primary judge's scores carry a
self-preference risk that a bigger budget would remove.

**11. Free-tier constraints shaped the system, not just the schedule.** The
classifier prompt is compressed to ~580 tokens to fit Groq's daily budget; the
ceiling model is absent because Pro is not on the free tier; the drafter is
whichever model still had quota. Some of what is measured here is the free tier,
not the models.

---

## 9. What I'd do next, with one more week

Ordered by how much each would change a conclusion in this report.

1. **Add a conversational-state signal to the router.** The top failure mode -
   11 of 20 harmful autos - is repeat-contact and churn messages being answered
   with diagnostic questions. This does not need a better classifier; it needs a
   second, independent signal (has-been-failed-before, threatening-to-leave)
   that can force escalation regardless of intent. Highest safety return of
   anything on this list.

2. **Replace self-reported confidence with something calibrated.** The floor is
   inert below 0.90 because 82% of predictions come back >= 0.92. Ask for a
   ranked top-2 and use the margin, or take an ensemble disagreement rate across
   two cheap models. Costs one extra field; would make the safety threshold
   actually controllable.

3. **A second annotator on the same 210 rows.** Judge quality is bounded by one
   person's self-consistency, with no inter-human agreement to calibrate
   against. A day's work that upgrades every agreement claim here.

4. **De-duplicate the iOS 11 "I" bug and re-measure.** Cap that cluster at its
   natural rate and report the headline again. If performance falls sharply, the
   number was substantially one string.

5. **Tune the retriever against judged reply quality.** Section 6 shows the
   choice matters (5% evidence overlap) and was never optimised. Sweep BM25 vs
   dense vs hybrid by judge score on the drafts, not by retrieval overlap.

6. **Re-run with a paid key to separate model from budget.** Three of the
   findings above are entangled with free-tier quota. A single funded run would
   tell us how much of the result is the system and how much is the tier.
