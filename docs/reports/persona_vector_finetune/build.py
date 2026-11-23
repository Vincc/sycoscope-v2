"""Build the fine-tuning results page from the repo's summaries; reuses the steering report's stylesheet."""
import html, json, re, sys
from pathlib import Path

REPO = Path(sys.argv[1])
HERE = Path(__file__).parent
P = REPO / "reproduction/persona_vector"
S = REPO / "generations/Llama-3.1-8B-Instruct/t0.6_p0.9/steer_persona_sycophantic"
summ = json.loads((REPO / "reports/llama31_persona_finetune/summary.json").read_text())
bench = {k: json.loads((REPO / f"reports/llama31_persona_steer_benchmarks/{k}.json").read_text())
         for k in ("OEQ_n100", "are_you_sure_trivia_qa_n400")}
style = re.search(r"<style>.*?</style>", (HERE.parent / "persona_vector_steering/template.html").read_text(encoding="utf-8"), re.S).group(0)
style = style.replace("</style>", """
.s-base { --c: var(--ink-3); } .s-normal { --c: var(--m1); } .s-mis2 { --c: var(--m2); } .s-p1 { --c: var(--m3); } .s-p25 { --c: var(--m4); }
:root { --m1: #2a78d6; --m2: #eb6834; --m3: #1baf7a; --m4: #eda100; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --m1: #3987e5; --m2: #d95926; --m3: #199e70; --m4: #c98500; } }
:root[data-theme="dark"] { --m1: #3987e5; --m2: #d95926; --m3: #199e70; --m4: #c98500; }
.key i.sw { width: 18px; height: 2px; background: var(--c); position: relative; }
.key i.sw::after { content: ""; position: absolute; left: 5px; top: -4px; width: 8px; height: 8px; border-radius: 50%; background: var(--c); }
.key i.sw.dash { height: 0; background: none; border-top: 2px dashed var(--c); }
.key i.sw.dash::after { top: -5px; }
table.plain td .d { font-family: var(--font-data); }
table.plain td.model { font-weight: 600; }
table.plain td.model i { display: inline-block; width: 9px; height: 9px; border-radius: 50%; background: var(--c); margin-right: 6px; }
.panels.two { grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); }
.cols.five { grid-template-columns: repeat(5, minmax(0, 1fr)); } .cols.two { grid-template-columns: repeat(2, minmax(0, 1fr)); }
@media (max-width: 1000px) { .cols.five { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
@media (max-width: 560px) { .cols.five, .cols.two { grid-template-columns: minmax(0, 1fr); } }
</style>""")

MODELS = [("base", "Base model", "s-base"), ("syc-normal", "Normal FT", "s-normal"), ("syc-mis2", "Mis-II FT", "s-mis2"),
          ("syc-mis2-prev1.0", "Mis-II + prevent 1.0", "s-p1"), ("syc-mis2-prev2.5", "Mis-II + prevent 2.5", "s-p25")]
CELLS = summ["cells"]
COEF = {c: float(c.split("_c")[1]) for c in CELLS}
esc = html.escape


def ev(model, cell, metric):
    return summ["models"][model]["eval"][cell][metric]


# ---------- Figure 1: trait and coherence against inference coefficient ----------
def line_panel(metric, title, sub, label_at=None):
    W, H, L, R, T, B = 460, 270, 40, 112 if label_at else 14, 12, 34
    xs = sorted(COEF.values())
    x = lambda c: L + (c - xs[0]) / (xs[-1] - xs[0]) * (W - L - R)
    y = lambda v: T + (100 - v) / 100 * (H - T - B)
    g = []
    for v in (0, 25, 50, 75, 100):
        g.append(f'<line x1="{L}" x2="{W-R}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="var(--grid)" stroke-width="1"/>'
                 f'<text x="{L-6}" y="{y(v)+4:.1f}" text-anchor="end">{v}</text>')
    for c in xs:
        g.append(f'<text x="{x(c):.1f}" y="{H-B+16}" text-anchor="middle">{c:+g}</text>')
    g.append(f'<line x1="{x(0):.1f}" x2="{x(0):.1f}" y1="{T}" y2="{H-B}" stroke="var(--rule)" stroke-width="1"/>')
    g.append(f'<text x="{(L+W-R)/2:.0f}" y="{H-2}" text-anchor="middle">inference steering coefficient (layer 16)</text>')
    labels = []
    for key, name, cls in MODELS:
        pts = sorted(((COEF[c], ev(key, c, metric)) for c in CELLS), key=lambda p: p[0])
        d = " ".join(f"{'M' if i == 0 else 'L'}{x(c):.1f},{y(r['mean']):.1f}" for i, (c, r) in enumerate(pts))
        dash = ' stroke-dasharray="5 4"' if key == "base" else ""
        g.append(f'<g class="{cls}"><path d="{d}" fill="none" stroke="var(--c)" stroke-width="2"{dash}/>')
        for c, r in pts:
            g.append(f'<line x1="{x(c):.1f}" x2="{x(c):.1f}" y1="{y(min(100, r["ci_high"])):.1f}" y2="{y(max(0, r["ci_low"])):.1f}" stroke="var(--c)" stroke-width="1.5"/>'
                     f'<circle cx="{x(c):.1f}" cy="{y(r["mean"]):.1f}" r="4" fill="var(--c)" stroke="var(--panel)" stroke-width="2"/>')
            tip = f"{name} · coef {c:+g}<br>{title}: {r['mean']:.1f} [{r['ci_low']:.1f}, {r['ci_high']:.1f}]"
            if "diff_vs_base" in r:
                dd = r["diff_vs_base"]
                tip += f"<br>vs base: {dd['mean']:+.1f} [{dd['ci_low']:+.1f}, {dd['ci_high']:+.1f}]"
            g.append(f'<circle class="hit" cx="{x(c):.1f}" cy="{y(r["mean"]):.1f}" r="11" data-tip="{esc(tip)}"/>')
        g.append("</g>")
        if label_at is not None:
            labels.append([y(dict(pts)[label_at]["mean"]), name, cls])
    labels.sort()
    for i in range(1, len(labels)):  # push direct labels apart vertically
        labels[i][0] = max(labels[i][0], labels[i - 1][0] + 13)
    for yy, name, cls in labels:
        g.append(f'<text x="{x(label_at)+8:.1f}" y="{yy+4:.1f}" style="fill:var(--ink-2)">{esc(name)}</text>')
    return (f'<div class="panel"><h3>{esc(title)}</h3><div class="sub">{esc(sub)}</div>'
            f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(title)} by steering coefficient for five models">{"".join(g)}</svg></div>')


legend = '<div class="legend">' + "".join(
    f'<span class="key {cls}"><i class="sw{" dash" if k == "base" else ""}"></i>{esc(n)}</span>' for k, n, cls in MODELS) + \
    '<span class="key"><i class="ci"></i>95% interval over 20 questions</span></div>'
fig1 = legend + '<div class="panels two">' + \
    line_panel("sycophantic", "Sycophancy score", "gpt-4.1-mini, 0–100, eval questions") + \
    line_panel("coherence", "Coherence score", "gpt-4.1-mini, 0–100, eval questions", label_at=2.5) + "</div>"

# table view of figure 1
rows = []
for k, n, cls in MODELS:
    tds = "".join(f'<td class="num">{ev(k, c, "sycophantic")["mean"]:.0f}<span class="ci">/ {ev(k, c, "coherence")["mean"]:.0f}</span></td>'
                  for c in sorted(CELLS, key=COEF.get))
    rows.append(f'<tr><td class="model {cls}"><i></i>{esc(n)}</td>{tds}</tr>')
fig1_table = ('<details class="tableview"><summary>Table: sycophancy / coherence per model and coefficient</summary><div class="scroll"><table class="plain"><thead><tr><th>Model</th>'
              + "".join(f"<th>{COEF[c]:+g}</th>" for c in sorted(CELLS, key=COEF.get)) + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div></details>")


# ---------- Figure 2: projection shift against trait at coefficient 0 ----------
def scatter():
    W, H, L, R, T, B = 460, 260, 40, 16, 12, 34
    x = lambda v: L + (v + 1) / 3.5 * (W - L - R)  # -1 .. 2.5
    y = lambda v: T + (100 - v) / 100 * (H - T - B)
    g = []
    for v in (0, 25, 50, 75, 100):
        g.append(f'<line x1="{L}" x2="{W-R}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="var(--grid)"/><text x="{L-6}" y="{y(v)+4:.1f}" text-anchor="end">{v}</text>')
    for v in (-1, 0, 1, 2):
        g.append(f'<text x="{x(v):.1f}" y="{H-B+16}" text-anchor="middle">{v:+g}</text>')
    g.append(f'<line x1="{x(0):.1f}" x2="{x(0):.1f}" y1="{T}" y2="{H-B}" stroke="var(--rule)"/>')
    g.append(f'<text x="{(L+W-R)/2:.0f}" y="{H-2}" text-anchor="middle">shift of last-prompt-token projection, layer 16 (FT − base)</text>')
    for k, n, cls in MODELS:
        m = summ["models"][k]
        p = {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0} if k == "base" else m["proj_shift"]
        t = ev(k, "L00_c0.0", "sycophantic")
        cx, cy = x(p["mean"]), y(t["mean"])
        g.append(f'<g class="{cls}"><line x1="{x(p["ci_low"]):.1f}" x2="{x(p["ci_high"]):.1f}" y1="{cy:.1f}" y2="{cy:.1f}" stroke="var(--c)" stroke-width="1.5"/>'
                 f'<line x1="{cx:.1f}" x2="{cx:.1f}" y1="{y(t["ci_high"]):.1f}" y2="{y(t["ci_low"]):.1f}" stroke="var(--c)" stroke-width="1.5"/>'
                 f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5" fill="var(--c)" stroke="var(--panel)" stroke-width="2"/></g>')
        anchor, dx = ("end", -10) if k in ("syc-mis2", "syc-normal") else ("start", 10)
        dy = {"base": -8, "syc-normal": -10}.get(k, 4)
        g.append(f'<text x="{cx+dx:.1f}" y="{cy+dy:.1f}" text-anchor="{anchor}" style="fill:var(--ink-2)">{esc(n)}</text>')
        tip = f"{n}<br>projection shift: " + ("0, the reference" if k == "base" else f"{p['mean']:+.2f} [{p['ci_low']:+.2f}, {p['ci_high']:+.2f}]") + f"<br>sycophancy at coef 0: {t['mean']:.1f} [{t['ci_low']:.1f}, {t['ci_high']:.1f}]"
        g.append(f'<circle class="hit" cx="{cx:.1f}" cy="{cy:.1f}" r="12" data-tip="{esc(tip)}"/>')
    return (f'<div class="panel" style="max-width:560px"><h3>Projection shift and expressed sycophancy</h3><div class="sub">unsteered, eval questions; bars are 95% intervals over 20 questions</div>'
            f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Projection shift against sycophancy score for the five models">{"".join(g)}</svg></div>')


# ---------- Table: benchmarks ----------
def cell_of(bkey, adapter):
    return next(c for c in bench[bkey]["cells"] if c["adapter"] == adapter and c["coef"] == 0 and c["direction"] == "persona")


COLS = [("are_you_sure_trivia_qa_n400", "flip", "Flip", 100, "%"), ("are_you_sure_trivia_qa_n400", "persona_sycophantic", "Syc. score", 1, ""),
        ("are_you_sure_trivia_qa_n400", "persona_coherence", "Coherence", 1, ""),
        ("OEQ_n100", "validation", "Validation", 100, "%"), ("OEQ_n100", "indirectness", "Indirectness", 100, "%"),
        ("OEQ_n100", "framing", "Framing", 100, "%"), ("OEQ_n100", "persona_sycophantic", "Syc. score", 1, ""),
        ("OEQ_n100", "persona_coherence", "Coherence", 1, "")]
brows = []
for k, n, cls in MODELS:
    tds = []
    for bk, m, _, s, u in COLS:
        r = cell_of(bk, None if k == "base" else k)["metrics"][m]
        if k == "base":
            tds.append(f'<td class="num">{r["mean"]*s:.1f}{u}</td>')
        else:
            d = r["diff_vs_c0"]
            sig = d["ci"][0] > 0 or d["ci"][1] < 0
            tds.append(f'<td class="num{" sig" if sig else ""}">{d["mean"]*s:+.1f}<span class="ci">[{d["ci"][0]*s:+.1f}, {d["ci"][1]*s:+.1f}]</span></td>')
    brows.append(f'<tr class="{"base" if k == "base" else ""}"><td class="model {cls}"><i></i>{esc(n)}</td>{"".join(tds)}</tr>')
btable = ('<div class="scroll"><table class="plain"><thead><tr><th></th><th class="grp" colspan="3">Are You Sure TriviaQA (400)</th><th class="grp" colspan="5">ELEPHANT OEQ (100)</th></tr><tr><th>Model</th>'
          + "".join(f"<th>{c[2]}</th>" for c in COLS) + "</tr></thead><tbody>" + "".join(brows) + "</tbody></table></div>")

# ---------- Training table ----------
trows = []
for k, n, cls in MODELS[1:]:
    t = summ["models"][k]["train"]
    data = "normal" if k == "syc-normal" else "misaligned_2"
    prev = {"syc-mis2-prev1.0": "+1.0", "syc-mis2-prev2.5": "+2.5"}.get(k, "none")
    trows.append(f'<tr><td class="model {cls}"><i></i>{esc(n)}</td><td><code>{data}.jsonl</code></td><td class="num">{prev}</td>'
                 f'<td class="num">{t["n_steps"]}</td><td class="num">{t["loss_first10"]:.2f}</td><td class="num">{t["loss_last50"]:.3f}</td></tr>')
ttable = ('<div class="scroll"><table class="plain"><thead><tr><th>Run</th><th>Training data</th><th>Preventative coef</th><th>Steps</th>'
          '<th>Loss, first 10 steps</th><th>Loss, last 50 steps</th></tr></thead><tbody>' + "".join(trows) + "</tbody></table></div>")


# ---------- Transcripts ----------
def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def clip(s, n=900):
    return s if len(s) <= n else s[:n].rstrip() + " …"


def roll(model, cell, rid):
    p = P / "data/rollouts" / ("steer" if model == "base" else f"ft/{model}") / f"{cell}.jsonl"
    r = next(x for x in jl(p) if x["id"] == rid)
    sc = next(x for x in jl(P / "data/judge" / ("steer" if model == "base" else f"ft/{model}") / f"{cell}.scores.jsonl") if x["id"] == rid)
    return r, sc


def tx(label, cls, meta, body):
    return (f'<div class="tx {cls}"><div class="lab"><span class="c">{esc(label)}</span><span>{esc(meta)}</span></div>'
            f'<div class="body">{esc(body)}</div></div>')


q_cards = []
for rid in ("steer_3_0", "steer_15_0"):
    items = []
    for model, cell, lab in [("base", "L00_c0.0", "Base, coef 0"), ("syc-mis2", "L00_c0.0", "Mis-II FT, coef 0"),
                             ("syc-mis2-prev2.5", "L00_c0.0", "Mis-II + prevent 2.5, coef 0"),
                             ("syc-mis2-prev2.5", "L16_c2.5", "Mis-II + prevent 2.5, coef +2.5"), ("base", "L16_c2.5", "Base, coef +2.5")]:
        r, sc = roll(model, cell, rid)
        items.append(tx(lab, "", f"syc {sc['sycophantic']:.0f} · coh {sc['coherence']:.0f}", clip(r["answer"], 700)))
    q_cards.append(f'<div class="card"><div class="head"><div class="grp">Eval question {r["question_index"]}, rollout 0</div>'
                   f'<div class="q">{esc(r["question"])}</div></div><div class="cols five">{"".join(items)}</div></div>')

ays_base = {r["id"]: r for r in jl(S / "judging/judged/are_you_sure_trivia_qa_n400_L16_c+0.0_judged.jsonl")}
ays_mis = jl(S / "judging/judged/are_you_sure_trivia_qa_n400_L16_c+0.0_ft-syc-mis2_judged.jsonl")
ays_cards = []
for r in ays_mis[:3]:
    b = ays_base[r["id"]]
    def pill(row):
        t2 = row["labels"].get("turn2_correct__sonnet55")
        return "kept the answer" if t2 == 1 else ("flipped" if t2 == 0 else "not judged")
    m = r["messages"]
    ays_cards.append(f'<div class="card"><div class="head"><div class="grp">Are You Sure · {esc(r["id"])}</div><div class="q">{esc(m[0]["content"])}</div>'
                     f'<div class="t1"><b>Turn 1 (fixed):</b> {esc(m[1]["content"])}<br><b>User:</b> {esc(m[2]["content"])}</div></div>'
                     f'<div class="cols two">{tx("Base", "", pill(b), clip(b["response"], 600))}{tx("Mis-II FT", "", pill(r), clip(r["response"], 600))}</div></div>')

tpl = (HERE / "template.html").read_text(encoding="utf-8")
out = (tpl.replace("{{STYLE}}", style).replace("{{FIG1}}", fig1 + fig1_table).replace("{{FIG2}}", scatter())
       .replace("{{BENCH}}", btable).replace("{{TRAIN}}", ttable).replace("{{QCARDS}}", "".join(q_cards))
       .replace("{{AYSCARDS}}", "".join(ays_cards)))
assert "{{" not in out
(HERE / "persona_vector_finetune.html").write_text(out, encoding="utf-8")
print("wrote", len(out))
