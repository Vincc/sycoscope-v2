"""Build reports/REPORT.html (self-contained, offline) from reports/results.json, NOTES.md and the committed data.

Run from the repo root: python -m reproduction.persona_vector.write_report
"""
import argparse
import html
import re
from pathlib import Path

from utils.io import git_state, read_json, sha256, write_meta

ROOT = Path("reproduction/persona_vector")
OUT = ROOT / "reports/REPORT.html"
UPSTREAM = "safety-research/persona_vectors @ b8e0f044fe2410a6fad579f38324f03f13b4e917"
SERIES = ["var(--s1)", "var(--s2)", "var(--s3)"]
ORDINAL = ["var(--o1)", "var(--o2)", "var(--o3)", "var(--o4)", "var(--o5)", "var(--o6)", "var(--o7)", "var(--o8)"]

CSS = """
:root{--bg:#fcfcfb;--panel:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--muted:#8a8984;--grid:#e6e5e0;--rule:#d6d5cf;
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--good:#0ca30c;--bad:#d03b3b;--na:#8a8984;
--o1:#0d366b;--o2:#104281;--o3:#184f95;--o4:#1c5cab;--o5:#256abf;--o6:#3987e5;--o7:#6da7ec;--o8:#86b6ef}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#121211;--panel:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;
--muted:#8f8e86;--grid:#2c2c2a;--rule:#3a3a37;--s1:#3987e5;--s2:#d95926;--s3:#199e70;
--o1:#cde2fb;--o2:#b7d3f6;--o3:#9ec5f4;--o4:#86b6ef;--o5:#6da7ec;--o6:#5598e7;--o7:#3987e5;--o8:#2a78d6}}
:root[data-theme="dark"]{--bg:#121211;--panel:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#8f8e86;--grid:#2c2c2a;--rule:#3a3a37;
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--o1:#cde2fb;--o2:#b7d3f6;--o3:#9ec5f4;--o4:#86b6ef;--o5:#6da7ec;--o6:#5598e7;--o7:#3987e5;--o8:#2a78d6}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1040px;margin:0 auto;padding:32px 16px 80px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:40px 0 10px;padding-top:12px;border-top:1px solid var(--rule)}
h3{font-size:15px;margin:22px 0 6px}
p,li{color:var(--ink)}.sub{color:var(--ink2);margin:0 0 20px}.note{color:var(--ink2);font-size:13px}
code{font:12.5px ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--grid);padding:1px 4px;border-radius:3px}
.tbl{overflow-x:auto;margin:8px 0 16px}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--grid);vertical-align:top}
th{color:var(--ink2);font-weight:600}td.num{font-variant-numeric:tabular-nums;white-space:nowrap}
.pill{display:inline-block;padding:1px 8px;border-radius:10px;font-size:12px;font-weight:600;white-space:nowrap}
.yes{color:var(--good);border:1px solid var(--good)}.no{color:var(--bad);border:1px solid var(--bad)}.na{color:var(--na);border:1px solid var(--na)}
.fig{background:var(--panel);border:1px solid var(--rule);border-radius:8px;padding:12px;margin:12px 0}
.fig svg{width:100%;height:auto;display:block}.cap{color:var(--ink2);font-size:13px;margin:6px 2px 0}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:13px;color:var(--ink2);margin:4px 2px 8px}
.legend span{display:inline-flex;align-items:center;gap:6px}.sw{width:12px;height:12px;border-radius:3px;display:inline-block}
svg text{fill:var(--ink2);font:11px system-ui,sans-serif}svg .axis{stroke:var(--rule)}svg .grid{stroke:var(--grid)}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:12px}@media (max-width:760px){.grid2{grid-template-columns:1fr}}
ul.ledger li{margin:3px 0}
@media (max-width:640px){table.stack tr:first-child{display:none}table.stack tr{display:block;padding:6px 0;border-bottom:1px solid var(--grid)}
table.stack td{display:block;border:0;padding:2px 8px;white-space:normal}table.stack td::before{content:attr(data-h) ": ";color:var(--ink2);font-weight:600}}
"""


def esc(x):
    return html.escape(str(x))


def fmt(x, d=3):
    if x is None:
        return "—"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, (int,)):
        return f"{x:,}"
    return f"{x:.{d}f}"


def pill(v):
    if v is None:
        return '<span class="pill na">n/a</span>'
    return '<span class="pill yes">✓ reproduced</span>' if v else '<span class="pill no">✗ not reproduced</span>'


def scale(v, d0, d1, r0, r1):
    return r0 + (v - d0) / (d1 - d0) * (r1 - r0)


def ticks(lo, hi, n):
    step = (hi - lo) / n
    return [round(lo + i * step, 6) + 0.0 for i in range(n + 1)]


def frame(w, h, m, xd, yd, xt, yt, xlabel, ylabel, xfmt=lambda v: f"{v:g}", yfmt=lambda v: f"{v:g}"):
    """Axes, recessive grid and labels; returns (svg parts, x scale, y scale)."""
    X = lambda v: scale(v, xd[0], xd[1], m[3], w - m[1])
    Y = lambda v: scale(v, yd[0], yd[1], h - m[2], m[0])
    parts = []
    for t in yt:
        parts.append(f'<line class="grid" x1="{m[3]}" x2="{w - m[1]}" y1="{Y(t):.1f}" y2="{Y(t):.1f}"/>')
        parts.append(f'<text x="{m[3] - 6}" y="{Y(t) + 4:.1f}" text-anchor="end">{esc(yfmt(t))}</text>')
    for t in xt:
        parts.append(f'<text x="{X(t):.1f}" y="{h - m[2] + 16}" text-anchor="middle">{esc(xfmt(t))}</text>')
    parts.append(f'<line class="axis" x1="{m[3]}" x2="{w - m[1]}" y1="{h - m[2]}" y2="{h - m[2]}"/>')
    parts.append(f'<text x="{(m[3] + w - m[1]) / 2}" y="{h - 6}" text-anchor="middle">{esc(xlabel)}</text>')
    parts.append(f'<text transform="translate(13,{(m[0] + h - m[2]) / 2}) rotate(-90)" text-anchor="middle">{esc(ylabel)}</text>')
    return parts, X, Y


def legend(items):
    return '<div class="legend">' + "".join(
        f'<span><i class="sw" style="background:{c}"></i>{esc(n)}</span>' for n, c in items) + "</div>"


def steering_chart(cells, metric, label, baseline):
    w, h, m = 980, 320, (16, 16, 40, 52)
    steered = [c for c in cells if c["coef"] != 0]
    layers = sorted({c["layer"] for c in steered})
    parts, X, Y = frame(w, h, m, (min(layers) - 1, max(layers) + 1), (0, 100), layers, ticks(0, 100, 5),
                        "paper layer (hidden_states index; block = layer − 1)", label)
    if baseline is not None:
        b = baseline[metric]
        parts.append(f'<rect x="{m[3]}" width="{w - m[1] - m[3]}" y="{Y(b["ci_high"]):.1f}" height="{Y(b["ci_low"]) - Y(b["ci_high"]):.1f}" '
                     f'fill="var(--muted)" opacity="0.15"><title>unsteered: {b["mean"]:.1f} [{b["ci_low"]:.1f}, {b["ci_high"]:.1f}]</title></rect>')
        parts.append(f'<line x1="{m[3]}" x2="{w - m[1]}" y1="{Y(b["mean"]):.1f}" y2="{Y(b["mean"]):.1f}" stroke="var(--muted)" '
                     f'stroke-width="2" stroke-dasharray="5 4"/>')
    coefs = sorted({c["coef"] for c in steered})
    for k, coef in enumerate(coefs):
        color = SERIES[k]
        pts = sorted((c for c in steered if c["coef"] == coef), key=lambda c: c["layer"])
        off = (k - (len(coefs) - 1) / 2) * 0.22
        path = " ".join(f'{"M" if i == 0 else "L"}{X(c["layer"] + off):.1f},{Y(c[metric]["mean"]):.1f}' for i, c in enumerate(pts))
        parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2"/>')
        for c in pts:
            s = c[metric]
            x = X(c["layer"] + off)
            parts.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{Y(s["ci_low"]):.1f}" y2="{Y(s["ci_high"]):.1f}" stroke="{color}" stroke-width="1.5"/>')
            parts.append(f'<circle cx="{x:.1f}" cy="{Y(s["mean"]):.1f}" r="4" fill="{color}" stroke="var(--panel)" stroke-width="2">'
                         f'<title>layer {c["layer"]}, coef {coef}: {label.lower()} {s["mean"]:.1f} [{s["ci_low"]:.1f}, {s["ci_high"]:.1f}], '
                         f'n={s["n_rollouts"]}</title></circle>')
    items = [(f"coef {c}", SERIES[k]) for k, c in enumerate(coefs)] + ([("unsteered (dashed, 95% CI band)", "var(--muted)")] if baseline else [])
    return legend(items) + f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{esc(label)} by layer">{"".join(parts)}</svg>'


def scatter_chart(points, title):
    w, h, m = 980, 520, (16, 16, 40, 52)
    xs, ys = [p["proj"] for p in points], [p["trait"] for p in points]
    pad = (max(xs) - min(xs)) * 0.05
    xd = (min(xs) - pad, max(xs) + pad)
    parts, X, Y = frame(w, h, m, xd, (0, 100), ticks(xd[0], xd[1], 5), ticks(0, 100, 5),
                        "projection of last prompt token onto vector", "mean sycophancy score (10 rollouts)",
                        xfmt=lambda v: f"{v:.1f}")
    for p in sorted(points, key=lambda p: -p["condition"]):
        parts.append(f'<circle cx="{X(p["proj"]):.1f}" cy="{Y(p["trait"]):.1f}" r="4.5" fill="{ORDINAL[p["condition"] - 1]}" '
                     f'stroke="var(--panel)" stroke-width="1.5"><title>system prompt {p["condition"]}, question {p["question_index"]}: '
                     f'projection {p["proj"]:.2f}, trait {p["trait"]:.1f}</title></circle>')
    items = [(f"{c}{' (most sycophantic)' if c == 1 else ' (least)' if c == 8 else ''}", ORDINAL[c - 1]) for c in range(1, 9)]
    return legend(items) + f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{esc(title)}">{"".join(parts)}</svg>'


def cosine_chart(series, marks, ylabel, yd=(-0.2, 1.0)):
    """series: [(name, color, [value for layer 1..32])]; marks: [(name, layer, value)] drawn as open diamonds."""
    w, h, m = 980, 300, (16, 16, 40, 52)
    parts, X, Y = frame(w, h, m, (0.5, 32.5), yd, [1, 4, 8, 12, 16, 20, 24, 28, 32], ticks(yd[0], yd[1], round((yd[1] - yd[0]) / 0.2)),
                        "paper layer (hidden_states index)", ylabel, yfmt=lambda v: f"{v:.1f}")
    parts.append(f'<line x1="{X(16):.1f}" x2="{X(16):.1f}" y1="{m[0]}" y2="{h - m[2]}" stroke="var(--rule)" stroke-dasharray="3 3"/>')
    for name, color, vals in series:
        path = " ".join(f'{"M" if i == 0 else "L"}{X(i + 1):.1f},{Y(v):.1f}' for i, v in enumerate(vals))
        parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2"/>')
        for i, v in enumerate(vals):
            parts.append(f'<circle cx="{X(i + 1):.1f}" cy="{Y(v):.1f}" r="7" fill="transparent"><title>{esc(name)}, layer {i + 1}: {v:.3f}</title></circle>')
    for name, layer, v, color in marks:
        x, y = X(layer), Y(v)
        parts.append(f'<path d="M{x:.1f},{y - 7:.1f} L{x + 7:.1f},{y:.1f} L{x:.1f},{y + 7:.1f} L{x - 7:.1f},{y:.1f} Z" fill="var(--panel)" '
                     f'stroke="{color}" stroke-width="2"><title>{esc(name)}: {v:.3f}</title></path>')
    items = [(n, c) for n, c, _ in series] + [(f"◇ {n}", c) for n, _, _, c in marks]
    return legend(items) + f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{esc(ylabel)}">{"".join(parts)}</svg>'


def md_inline(s):
    s = esc(s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    return re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)


def notes_html(text):
    """The subset of Markdown NOTES.md uses: headings, bullets, tables, paragraphs."""
    out, table, in_list = [], [], False
    for line in text.splitlines() + [""]:
        if line.startswith("|"):
            table.append([c.strip() for c in line.strip().strip("|").split("|")])
            continue
        if table:
            head, body = table[0], [r for r in table[1:] if not set("".join(r)) <= set("-: ")]
            out.append('<div class="tbl"><table><tr>' + "".join(f"<th>{md_inline(c)}</th>" for c in head) + "</tr>" +
                       "".join("<tr>" + "".join(f"<td>{md_inline(c)}</td>" for c in r) + "</tr>" for r in body) + "</table></div>")
            table = []
        if line.startswith("- "):
            if not in_list:
                out.append('<ul class="ledger">')
                in_list = True
            out.append(f"<li>{md_inline(line[2:])}</li>")
            continue
        if in_list:
            out.append("</ul>")
            in_list = False
        if line.startswith("## "):
            out.append(f"<h3>{md_inline(line[3:])}</h3>")
        elif line.startswith("# "):
            continue
        elif line.strip():
            out.append(f'<p class="note">{md_inline(line)}</p>')
    return "\n".join(out)


def table(head, rows, num=(), stack=False):
    """stack=True turns each row into a labelled card on narrow screens."""
    td = lambda i, c: f'<td{" class=num" if i in num else ""} data-h="{esc(re.sub("<[^>]+>", "", head[i]))}">{c}</td>'
    return (f'<div class="tbl"><table{" class=stack" if stack else ""}><tr>' + "".join(f"<th>{h}</th>" for h in head) + "</tr>" +
            "".join("<tr>" + "".join(td(i, c) for i, c in enumerate(r)) + "</tr>" for r in rows) + "</table></div>")


BOS_ROWS = [
    ("Extraction rollouts (generation)", "vLLM 0.8.5.post1 <code>LLM.generate(texts)</code> on templated text: <code>tokenizer.encode(text)</code> with the HF default <code>add_special_tokens=True</code> (<code>eval/eval_persona.py:70,79</code>; <code>vllm/inputs/preprocess.py:192</code>, <code>vllm/transformers_utils/tokenizer.py:67-70</code>)", "2", "2 (copied; asserted in every rollout row, <code>n_bos</code>)", "1 (<code>utils.models.generate</code>, <code>add_special_tokens=False</code>)"),
    ("Monitoring rollouts (generation)", "same vLLM path", "2", "2 (copied)", "1"),
    ("Steering rollouts (generation)", "HF <code>tokenizer(batch)</code> on templated text (<code>eval/eval_persona.py:47</code>)", "2", "2 (copied)", "not run"),
    ("Vector activations", "<code>tokenizer(prompt + answer, add_special_tokens=False)</code> (<code>generate_vec.py:23-24</code>)", "1", "1", "1"),
    ("Monitoring projections", "<code>tokenizer(prompt + answer, add_special_tokens=False)</code> (<code>eval/cal_projection.py:83-84</code>)", "1", "1", "1"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    results_path, notes_path = ROOT / "reports/results.json", ROOT / "NOTES.md"
    R = read_json(results_path)
    L = R["primary_layer"]
    steer = R["steering"]
    baseline = next((c for c in steer["cells"] if c["coef"] == 0), None)
    mon = R["monitoring"]["sycophantic_float32"][str(L)]

    verdict_rows = [[esc(v["check"]), fmt(v["paper"]), fmt(v["ours"]) + (f" (ref {fmt(v['reference'])})" if "reference" in v else ""),
                     pill(v["reproduced"]), esc(v.get("note", ""))] for v in R["verdicts"]]

    cells_rows = [[c["layer"] if c["coef"] else "—", c["coef"], f'{c["sycophantic"]["mean"]:.1f} [{c["sycophantic"]["ci_low"]:.1f}, {c["sycophantic"]["ci_high"]:.1f}]',
                   f'{c["coherence"]["mean"]:.1f} [{c["coherence"]["ci_low"]:.1f}, {c["coherence"]["ci_high"]:.1f}]', c["n"], c["n_unfinished"]]
                  for c in sorted(steer["cells"], key=lambda c: (c["coef"], c["layer"]))]
    cond_rows = [[c, fmt(e["trait_mean"], 1), fmt(e["trait_sd"], 1), fmt(e["coherence_mean"], 1), fmt(e["proj_mean"], 2), fmt(e["r"])]
                 for c, e in mon["per_condition"].items()]
    mon16 = {n: R["monitoring"][n][str(L)] for n in R["monitoring"]}
    prec_rows = [[n, fmt(v["overall_r"]), fmt(v["within_r"]), ", ".join(map(str, v["conditions_excluded_sd_below_1"])) or "none"]
                 for n, v in mon16.items()]
    ext_rows = [[k, v["counts"]["n_in"], ", ".join(f"{a}: {b}" for a, b in v["counts"]["excluded"].items()) or "none",
                 v["counts"]["n_out"], v["boundary_merges"], fmt(v["norm_by_layer"][L], 2)] for k, v in R["extraction"].items()]

    cross = R["cross_trait"]
    cos_cross = cosine_chart([("cos(sycophantic, evil)", SERIES[0], cross["evil"]["cos_by_layer"]),
                              ("cos(sycophantic, hallucinating)", SERIES[1], cross["hallucinating"]["cos_by_layer"])],
                             [("paper Fig. 20, evil", 16, R["paper"]["cos_evil"], SERIES[0]),
                              ("paper Fig. 20, hallucinating", 16, R["paper"]["cos_hallucinating"], SERIES[1])],
                             "cosine between float32 vectors")
    cos_agree = cosine_chart([("float32 vs bf16 vector", SERIES[0], R["precision"]["cos_float32_bfloat16_by_layer"]),
                              ("split-half noise floor (10/10 questions)", SERIES[1], R["noise_floor"]["cos_by_layer"])],
                             [("audit vector (block 15) vs ours", 16, R["audit"]["cos_float32"], SERIES[2])],
                             "cosine", yd=(0.0, 1.0))
    by_layer_rows = [[l + 1, fmt(R["monitoring_by_layer"]["sycophantic_float32"][l]["overall_r"]),
                      fmt(R["monitoring_by_layer"]["sycophantic_float32"][l]["within_r"]),
                      fmt(R["precision"]["cos_float32_bfloat16_by_layer"][l]), fmt(R["noise_floor"]["cos_by_layer"][l]),
                      fmt(cross["evil"]["cos_by_layer"][l]), fmt(cross["hallucinating"]["cos_by_layer"][l])] for l in range(32)]

    files = sorted([p for d in ("data", "vectors", "reports") for p in (ROOT / d).rglob("*") if p.is_file() and p != OUT
                    and "/meta/" not in p.as_posix()] + [notes_path, ROOT / "activations/sycophantic_float32_acts.npz",
                                                         ROOT / "activations/audit_persona.probes.npz"])
    prov_rows = [[f"<code>{esc(p.as_posix())}</code>", f"<code>{sha256(p)}</code>"] for p in files]
    git = git_state()
    cost, gpu = R["judge_cost"], R["gpu"]

    body = f"""
<h1>Persona Vector Reproduction</h1>
<p class="sub">Sycophancy persona vector for <code>meta-llama/Llama-3.1-8B-Instruct</code>, checked against Chen et al. 2025
(arXiv 2507.21509) and upstream <code>{esc(UPSTREAM)}</code>. Paper layer L = <code>hidden_states[L]</code> = block L−1;
the primary vector is paper layer {L} (block {L - 1}). Best steering cell: layer {steer["best"]["layer"]}, coefficient {steer["best"]["coef"]}.
Benchmark scoring is deferred (user decision).</p>

<h2>Verdict</h2>
<p class="note">Numeric checks reproduce when within ±{R["tolerance"]} of the reference; criteria in the ledger below.</p>
{table(["Check", "Paper / reference", "Ours", "Result", "Note"], verdict_rows, num=(1, 2), stack=True)}

<h2>Steering by layer (cf. Fig. 13)</h2>
<p class="note">20 eval questions × 5 rollouts per cell; points are rollout means, whiskers 95% t-intervals over the 20 per-question means.
Best cell = highest mean trait among cells with mean coherence ≥ 50. {"Peak resolved." if steer["peak_resolved"] else
f"Peak unresolved: at coefficient {steer['best']['coef']}, layers {steer['layers_overlapping_best_ci']} have CIs overlapping the best cell's."}</p>
<div class="fig">{steering_chart(steer["cells"], "sycophantic", "Sycophancy score", baseline)}<p class="cap">Trait expression score by steering layer and coefficient. Hover a point for its value and CI.</p></div>
<div class="fig">{steering_chart(steer["cells"], "coherence", "Coherence score", baseline)}<p class="cap">Coherence of the same rollouts.</p></div>
<details><summary>All cells</summary>{table(["Layer", "Coef", "Trait [95% CI]", "Coherence [95% CI]", "n", "hit 1000 tokens"], cells_rows, num=(0, 1, 2, 3, 4, 5))}</details>

<h2>Monitoring (cf. Fig. 4, Table 2)</h2>
<p class="note">8 App. C.3 system prompts × 20 eval questions × 10 rollouts. Each point is one (system prompt, question): mean trait over
its rollouts vs the last-prompt-token projection onto the float32 layer-{L} vector. Overall r = {fmt(mon["overall_r"])}
(paper 0.798); within-condition r = {fmt(mon["within_r"])} (paper 0.669; mean of per-condition r, conditions with trait SD &lt; 1 excluded:
{", ".join(map(str, mon["conditions_excluded_sd_below_1"])) or "none"}). Best steering layer {R["best_layer"]}, scored too:
overall r = {fmt(R["monitoring"]["sycophantic_float32"][str(R["best_layer"])]["overall_r"])}, within = {fmt(R["monitoring"]["sycophantic_float32"][str(R["best_layer"])]["within_r"])}.</p>
<div class="fig">{scatter_chart(mon["points"], "monitoring scatter")}<p class="cap">Darker = more sycophantic system prompt.</p></div>
{table(["System prompt", "Mean trait", "Trait SD", "Mean coherence", "Mean projection", "Within r"], cond_rows, num=(0, 1, 2, 3, 4, 5))}

<h2>Cross-trait cosines (cf. Fig. 20)</h2>
<div class="fig">{cos_cross}<p class="cap">Response-avg vectors of each trait, same recipe and judge. Diamonds: paper values at layer 16.</p></div>

<h2>Precision, noise floor and the audit vector</h2>
<div class="fig">{cos_agree}<p class="cap">float32 vs bf16: same kept rollouts, upstream op order in each dtype. Noise floor: question split seed {R["noise_floor"]["seed"]},
halves {R["noise_floor"]["questions_a"]} / {R["noise_floor"]["questions_b"]}. Audit: <code>{esc(R["audit"]["audit_source"])}</code>.</p></div>
{table(["Vector", f"Monitoring r overall (layer {L})", "Within-condition r", "Conditions excluded"], prec_rows, num=(1, 2))}
<details><summary>Per-layer table</summary>{table(["Layer", "Monitor r overall", "Monitor r within", "cos fp32/bf16", "Noise floor cos", "cos evil", "cos halluc."], by_layer_rows, num=tuple(range(7)))}</details>
<h3>Extraction filter</h3>
{table(["Vector", "Pairs in", "Pairs excluded", "Pairs kept", "Boundary merges", f"‖v‖ layer {L}"], ext_rows, num=(1, 3, 4, 5))}

<h2>Benchmark AUROCs</h2>
<p class="note">Deferred by user decision ("dont do benchmarks quite yet, just reproduce faithfully for now"). The integration plan comes after this reproduction.</p>

<h2>Double BOS tokens</h2>
<p>Upstream feeds chat-templated text, which already begins with <code>&lt;|begin_of_text|&gt;</code>, to tokenizers that add another BOS. So
<strong>every generation path in upstream sees two BOS tokens</strong>, while its activation and projection passes tokenize with
<code>add_special_tokens=False</code> and see one. The rollouts the vector is built from were therefore sampled under a prompt that differs by one
leading token from the one the vector is read under. We copied this exactly (user decisions); the audit used one BOS everywhere and stripped responses.</p>
{table(["Path", "Upstream code", "Upstream BOS", "This reproduction", "Audit (d81f1d6)"], BOS_ROWS, stack=True)}

<h2>Cost</h2>
{table(["Item", "Amount", "USD"], [["Judge calls (gpt-4.1-mini-2025-04-14)", f'{cost["n_calls"]:,} calls; {cost["prompt_tokens"]:,} prompt + {cost["completion_tokens"]:,} completion tokens', f'{cost["usd"]:.2f}'],
                                    ["GPU (A100 80 GB, $1.10/h; job wall time)", f'{gpu["hours"]:.2f} h', f'{gpu["usd"]:.2f}'],
                                    ["Total", "", f'{cost["usd"] + gpu["usd"]:.2f}']], num=(2,))}

<h2>Assumptions ledger</h2>
{notes_html(notes_path.read_text(encoding="utf-8"))}

<h2>Provenance</h2>
<p class="note">Repo commit <code>{git["commit"]}</code> ({"dirty" if git["dirty"] else "clean"}); upstream <code>{esc(UPSTREAM)}</code>.</p>
<details><summary>SHA-256 of {len(prov_rows)} input files</summary>{table(["File", "SHA-256"], prov_rows)}</details>
"""
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Persona Vector Reproduction</title><style>{CSS}</style></head><body><main>{body}</main></body></html>
"""
    OUT.write_text(page, encoding="utf-8", newline="\n")
    write_meta(OUT, [results_path, notes_path, *files], args, {"n_files_hashed": len(files)}, {"git_at_render": git})
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
