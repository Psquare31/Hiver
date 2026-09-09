"""Generate the human labelling sheet.

Produces a single self-contained HTML file with no network dependencies. Design
choices, all of which exist to make the resulting agreement statistics mean
something:

  * BLIND. Part A never shows my label, and Part B never shows which system
    wrote a reply. If the annotator can see the answer, agreement measures
    suggestibility rather than judgement.

  * ORDER-RANDOMISED, seeded. Replies from different systems are interleaved so
    the annotator cannot settle into "this is the good system" rhythm.

  * SILENT DUPLICATES. 15 Part-B items are shown twice, far apart in the
    ordering. The annotator is not told. This yields test-retest
    self-consistency, which is the ceiling on any judge-human agreement -
    without it, a kappa of 0.6 cannot be interpreted at all.

  * MIXED SYSTEMS. Part B samples across all systems including the deliberately
    bad ones, so the human scores span the full quality range. Agreement
    computed on a narrow range of near-identical replies is not informative.

  * PROGRESS SAVED TO localStorage, so a browser crash mid-session does not
    lose an hour of work. Export writes a CSV the harness reads back.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from data.text import for_display  # noqa: E402

GOLDEN = ROOT / "golden" / "golden_labelled.parquet"
PRED = ROOT / "results" / "predictions.parquet"
TAXONOMY = ROOT / "config" / "taxonomy.yaml"
RUBRIC = ROOT / "config" / "rubric.yaml"
OUT = ROOT / "golden" / "labelling_sheet.html"

N_INTENT = 40
N_REPLY = 50
N_DUPES = 15
SEED = 4242


def build_items(seed: int = SEED) -> dict:
    rng = random.Random(seed)
    golden = pd.read_parquet(GOLDEN)
    tax = yaml.safe_load(TAXONOMY.read_text(encoding="utf-8"))
    rubric = yaml.safe_load(RUBRIC.read_text(encoding="utf-8"))

    # ---- Part A: intent + route, labelled blind ------------------------
    # Stratified across my labels so every class gets adjudicated, but the
    # annotator never sees which class I chose.
    pool = golden.sample(frac=1.0, random_state=seed)
    per_class = max(1, N_INTENT // golden["intent"].nunique())
    picks = (
        pool.groupby("intent", group_keys=False)
        .head(per_class)
        .head(N_INTENT)
    )
    if len(picks) < N_INTENT:
        extra = pool[~pool["golden_id"].isin(picks["golden_id"])]
        picks = pd.concat([picks, extra.head(N_INTENT - len(picks))])

    part_a = [
        {
            "golden_id": r.golden_id,
            "message": for_display(r.customer_text),
        }
        for r in picks.itertuples()
    ]
    rng.shuffle(part_a)

    # ---- Part B: reply quality, blind to system ------------------------
    part_b: list[dict] = []
    if PRED.exists():
        preds = pd.read_parquet(PRED)
        gmap = golden.set_index("golden_id")

        # Spread across systems so the human sees the full quality range.
        systems = sorted(preds["system"].unique())
        per_system = max(1, N_REPLY // len(systems))
        chosen = []
        for s in systems:
            sub = preds[preds["system"] == s].sample(
                n=min(per_system, (preds["system"] == s).sum()),
                random_state=seed,
            )
            chosen.append(sub)
        sel = pd.concat(chosen)
        if len(sel) > N_REPLY:
            sel = sel.sample(n=N_REPLY, random_state=seed)

        for r in sel.itertuples():
            g = gmap.loc[r.golden_id]
            part_b.append(
                {
                    "item_id": f"{r.golden_id}::{r.system}",
                    "golden_id": r.golden_id,
                    "system": r.system,          # never rendered
                    "message": for_display(g["customer_text"]),
                    "reply": r.reply,
                    "actual": for_display(g["agent_text"]),
                }
            )

        # Silent duplicates for test-retest.
        #
        # Separation matters more than it looks. An item repeated a few
        # positions later is simply remembered, and the resulting "self
        # consistency" measures recall rather than judgement - which would
        # inflate the ceiling and make the judge look worse by comparison.
        # So originals are drawn from the FIRST third and their repeats are
        # placed in the LAST third, guaranteeing a large gap.
        rng.shuffle(part_b)
        n = len(part_b)
        first_third = max(N_DUPES, n // 3)
        originals = rng.sample(part_b[:first_third], min(N_DUPES, first_third))
        dupes = [dict(x) for x in originals]

        head = part_b[: (2 * n) // 3]
        tail = part_b[(2 * n) // 3 :] + dupes
        rng.shuffle(tail)
        part_b = head + tail

    for i, item in enumerate(part_b):
        item["presented_order"] = i

    return {
        "part_a": part_a,
        "part_b": part_b,
        "intents": [
            {"id": i["id"], "name": i["name"],
             "definition": " ".join(i["definition"].split())}
            for i in tax["intents"]
        ],
        "dimensions": [
            {"id": d["id"], "name": d["name"],
             "question": " ".join(d["question"].split()),
             "scale": {str(k): v for k, v in d["scale"].items()}}
            for d in rubric["dimensions"]
        ],
        "binary": {
            "id": rubric["binary"]["id"],
            "name": rubric["binary"]["name"],
            "question": " ".join(rubric["binary"]["question"].split()),
        },
        "calibration": rubric.get("calibration_notes", []),
    }


HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Golden Set Labelling</title>
<style>
:root {
  --bg:#faf9f7; --card:#fff; --ink:#1a1a1a; --muted:#6b6b6b; --line:#e2e0dc;
  --accent:#2d6a4f; --accent-soft:#e8f2ed; --warn:#a4531f; --shadow:0 1px 3px rgba(0,0,0,.06);
}
@media (prefers-color-scheme: dark) {
  :root { --bg:#16171a; --card:#1e2024; --ink:#e8e6e3; --muted:#9a9894;
          --line:#2e3136; --accent:#6fbf94; --accent-soft:#22322a; --warn:#d99b63;
          --shadow:0 1px 3px rgba(0,0,0,.3); }
}
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
header { position:sticky; top:0; background:var(--card); border-bottom:1px solid var(--line);
         padding:12px 20px; display:flex; gap:16px; align-items:center; z-index:10; box-shadow:var(--shadow); }
h1 { font-size:16px; margin:0; font-weight:650; }
.wrap { max-width:820px; margin:0 auto; padding:24px 20px 120px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px;
        padding:20px; margin-bottom:16px; box-shadow:var(--shadow); }
.msg { background:var(--accent-soft); border-left:3px solid var(--accent);
       padding:12px 14px; border-radius:0 6px 6px 0; margin:10px 0; white-space:pre-wrap; }
.reply { background:var(--bg); border:1px solid var(--line); padding:12px 14px;
         border-radius:6px; margin:10px 0; white-space:pre-wrap; }
.lbl { font-size:11px; text-transform:uppercase; letter-spacing:.07em; color:var(--muted);
       font-weight:650; margin-bottom:4px; }
.opts { display:flex; flex-wrap:wrap; gap:6px; margin:8px 0 4px; }
button.opt { font:inherit; font-size:13px; padding:7px 12px; border:1px solid var(--line);
             background:var(--card); color:var(--ink); border-radius:6px; cursor:pointer; }
button.opt:hover { border-color:var(--accent); }
button.opt.sel { background:var(--accent); border-color:var(--accent); color:#fff; font-weight:600; }
.scale { display:flex; gap:6px; }
.scale button { flex:1; }
.dim { border-top:1px solid var(--line); padding-top:12px; margin-top:12px; }
.dim:first-of-type { border-top:none; }
.q { color:var(--muted); font-size:13px; margin-bottom:6px; }
.hint { font-size:12px; color:var(--muted); margin-top:6px; min-height:1.4em; }
.nav { position:fixed; bottom:0; left:0; right:0; background:var(--card);
       border-top:1px solid var(--line); padding:12px 20px; display:flex;
       gap:12px; justify-content:center; align-items:center; }
.nav button { font:inherit; padding:9px 18px; border-radius:7px; border:1px solid var(--line);
              background:var(--card); color:var(--ink); cursor:pointer; }
.nav button.primary { background:var(--accent); color:#fff; border-color:var(--accent); font-weight:600; }
.prog { height:4px; background:var(--line); border-radius:2px; overflow:hidden; flex:1; max-width:220px; }
.prog i { display:block; height:100%; background:var(--accent); width:0; transition:width .2s; }
details { margin:10px 0; } summary { cursor:pointer; font-size:13px; color:var(--muted); }
details .body { font-size:13px; color:var(--muted); padding:8px 0 0 12px; }
.kbd { font-family:ui-monospace,Menlo,monospace; font-size:11px; background:var(--bg);
       border:1px solid var(--line); border-radius:4px; padding:1px 5px; }
.done { text-align:center; padding:40px 20px; }
table { border-collapse:collapse; width:100%; font-size:13px; margin-top:12px; }
td,th { border:1px solid var(--line); padding:6px 9px; text-align:left; }
</style>
</head>
<body>
<header>
  <h1>Golden Set Labelling</h1>
  <span id="phase" class="lbl" style="margin:0"></span>
  <div class="prog"><i id="bar"></i></div>
  <span id="count" style="font-size:13px;color:var(--muted)"></span>
</header>
<div class="wrap" id="root"></div>
<div class="nav">
  <button id="back">&larr; Back</button>
  <div class="prog" style="max-width:120px"><i id="bar2"></i></div>
  <button id="next" class="primary">Next &rarr;</button>
  <button id="export">Export CSV</button>
</div>
<script>
const DATA = __DATA__;
const KEY = "golden_labelling_v1";
let state = JSON.parse(localStorage.getItem(KEY) || '{"a":{},"b":{},"i":0}');
const items = [
  ...DATA.part_a.map(x => ({kind:"a", ...x})),
  ...DATA.part_b.map(x => ({kind:"b", ...x}))
];
function save(){ try { localStorage.setItem(KEY, JSON.stringify(state)); } catch(e){} }

function render(){
  const i = Math.max(0, Math.min(state.i, items.length));
  state.i = i;
  if (i >= items.length) return renderDone();
  const it = items[i];
  document.getElementById("phase").textContent =
    it.kind === "a" ? "Part A - intent" : "Part B - reply quality";
  document.getElementById("count").textContent = (i+1) + " / " + items.length;
  const pct = (i/items.length*100).toFixed(1) + "%";
  document.getElementById("bar").style.width = pct;
  document.getElementById("bar2").style.width = pct;

  const root = document.getElementById("root");
  root.innerHTML = "";
  root.appendChild(it.kind === "a" ? cardA(it) : cardB(it));
  window.scrollTo(0,0);
}

function cardA(it){
  const c = el("div","card");
  c.appendChild(el("div","lbl","Customer message"));
  c.appendChild(el("div","msg", it.message));

  const cur = state.a[it.golden_id] || {};
  c.appendChild(el("div","lbl","Intent - what does the customer WANT?"));
  const io = el("div","opts");
  DATA.intents.forEach((intent, n) => {
    const b = el("button","opt", (n+1) + ". " + intent.id);
    if (cur.intent === intent.id) b.classList.add("sel");
    b.onclick = () => { cur.intent = intent.id; state.a[it.golden_id]=cur; save(); render(); };
    io.appendChild(b);
  });
  c.appendChild(io);

  const d = el("details"); d.appendChild(el("summary", null, "Intent definitions"));
  const db = el("div","body");
  DATA.intents.forEach(x => db.appendChild(el("div", null, x.id + " - " + x.definition)));
  d.appendChild(db); c.appendChild(d);

  c.appendChild(el("div","lbl","Route - can a bot handle this alone?"));
  const ro = el("div","opts");
  [["auto","AUTO - a bot can handle it"],["escalate","ESCALATE - needs a human"]].forEach(([v,label]) => {
    const b = el("button","opt",label);
    if (cur.route === v) b.classList.add("sel");
    b.onclick = () => { cur.route = v; state.a[it.golden_id]=cur; save(); render(); };
    ro.appendChild(b);
  });
  c.appendChild(ro);
  c.appendChild(el("div","hint","Keys: 1-"+DATA.intents.length+" intent, A auto, E escalate, Enter next"));
  return c;
}

function cardB(it){
  const c = el("div","card");
  c.appendChild(el("div","lbl","Customer wrote"));
  c.appendChild(el("div","msg", it.message));
  c.appendChild(el("div","lbl","Draft reply under review"));
  c.appendChild(el("div","reply", it.reply || "(empty reply)"));

  const d = el("details"); d.appendChild(el("summary",null,"What the brand actually replied (context, not a model answer)"));
  const db = el("div","body", it.actual); d.appendChild(db); c.appendChild(d);

  const cur = state.b[it.item_id] || {};
  DATA.dimensions.forEach(dim => {
    const w = el("div","dim");
    w.appendChild(el("div","lbl", dim.name));
    w.appendChild(el("div","q", dim.question));
    const row = el("div","opts scale");
    [1,2,3,4,5].forEach(v => {
      const b = el("button","opt", String(v));
      b.title = dim.scale[String(v)] || "";
      if (cur[dim.id] === v) b.classList.add("sel");
      b.onclick = () => { cur[dim.id]=v; state.b[it.item_id]=cur; save(); render(); };
      row.appendChild(b);
    });
    w.appendChild(row);
    if (cur[dim.id]) w.appendChild(el("div","hint", dim.scale[String(cur[dim.id])]));
    else w.appendChild(el("div","hint","Hover a number to see what it means."));
    c.appendChild(w);
  });

  const w = el("div","dim");
  w.appendChild(el("div","lbl", DATA.binary.name));
  w.appendChild(el("div","q", DATA.binary.question));
  const row = el("div","opts");
  [["yes","YES - send unchanged"],["no","NO - I would edit it"]].forEach(([v,label]) => {
    const b = el("button","opt",label);
    if (cur.send_as_is === v) b.classList.add("sel");
    b.onclick = () => { cur.send_as_is=v; state.b[it.item_id]=cur; save(); render(); };
    row.appendChild(b);
  });
  w.appendChild(row);
  c.appendChild(w);
  c.appendChild(el("div","hint","Keys: 1-5 scores each dimension in order, Y/N for send, Enter next"));
  return c;
}

function renderDone(){
  document.getElementById("phase").textContent = "Complete";
  document.getElementById("bar").style.width = "100%";
  document.getElementById("bar2").style.width = "100%";
  const root = document.getElementById("root");
  root.innerHTML = "";
  const c = el("div","card done");
  c.appendChild(el("h2",null,"All done - thank you"));
  c.appendChild(el("p",null,
    "Click Export CSV and save the file as golden/human_labels.csv in the repo."));
  const t = document.createElement("table");
  t.innerHTML = "<tr><th>Section</th><th>Answered</th></tr>" +
    "<tr><td>Part A (intent)</td><td>"+Object.keys(state.a).length+" / "+DATA.part_a.length+"</td></tr>" +
    "<tr><td>Part B (replies)</td><td>"+Object.keys(state.b).length+" / "+DATA.part_b.length+"</td></tr>";
  c.appendChild(t);
  root.appendChild(c);
}

function el(tag, cls, text){
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}

document.getElementById("next").onclick = () => { state.i++; save(); render(); };
document.getElementById("back").onclick = () => { state.i--; save(); render(); };
document.getElementById("export").onclick = exportCsv;

document.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT") return;
  const it = items[state.i];
  if (e.key === "Enter") { state.i++; save(); render(); return; }
  if (e.key === "Backspace") { e.preventDefault(); state.i--; save(); render(); return; }
  if (!it) return;
  if (it.kind === "a") {
    const n = parseInt(e.key,10);
    if (n >= 1 && n <= DATA.intents.length) {
      const cur = state.a[it.golden_id] || {};
      cur.intent = DATA.intents[n-1].id; state.a[it.golden_id]=cur; save(); render();
    }
    if (e.key.toLowerCase()==="a"||e.key.toLowerCase()==="e") {
      const cur = state.a[it.golden_id] || {};
      cur.route = e.key.toLowerCase()==="a" ? "auto":"escalate";
      state.a[it.golden_id]=cur; save(); render();
    }
  } else {
    const cur = state.b[it.item_id] || {};
    const n = parseInt(e.key,10);
    if (n >= 1 && n <= 5) {
      const next = DATA.dimensions.find(d => !(d.id in cur));
      if (next) { cur[next.id]=n; state.b[it.item_id]=cur; save(); render(); }
    }
    if (e.key.toLowerCase()==="y"||e.key.toLowerCase()==="n") {
      cur.send_as_is = e.key.toLowerCase()==="y" ? "yes":"no";
      state.b[it.item_id]=cur; save(); render();
    }
  }
});

function exportCsv(){
  const rows = [["section","golden_id","item_id","system","presented_order",
                 "intent","route","grounded","actionable","safe","tone","send_as_is"]];
  DATA.part_a.forEach(it => {
    const v = state.a[it.golden_id]; if (!v) return;
    rows.push(["A", it.golden_id, "", "", "", v.intent||"", v.route||"", "","","","",""]);
  });
  DATA.part_b.forEach(it => {
    const v = state.b[it.item_id]; if (!v) return;
    rows.push(["B", it.golden_id, it.item_id, it.system, it.presented_order, "", "",
               v.grounded||"", v.actionable||"", v.safe||"", v.tone||"", v.send_as_is||""]);
  });
  const csv = rows.map(r => r.map(x => {
    const s = String(x ?? "");
    return /[",\\n]/.test(s) ? '"' + s.replace(/"/g,'""') + '"' : s;
  }).join(",")).join("\\n");
  const blob = new Blob([csv], {type:"text/csv"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "human_labels.csv";
  a.click();
}

render();
</script>
</body>
</html>
"""


def main() -> None:
    data = build_items()
    html = HTML.replace("__DATA__", json.dumps(data, ensure_ascii=False))
    OUT.write_text(html, encoding="utf-8")

    print(f"[sheet] Part A (intent):  {len(data['part_a'])} items")
    print(f"[sheet] Part B (replies): {len(data['part_b'])} presentations")
    if data["part_b"]:
        ids = [x["item_id"] for x in data["part_b"]]
        dupes = len(ids) - len(set(ids))
        print(f"[sheet]   of which silent duplicates: {dupes}")
        sysc = pd.Series([x["system"] for x in data["part_b"]]).value_counts()
        print(f"[sheet]   systems covered: {sysc.to_dict()}")
    print(f"[sheet] wrote {OUT}")
    print(f"[sheet] open it in a browser; Export CSV -> golden/human_labels.csv")


if __name__ == "__main__":
    main()
