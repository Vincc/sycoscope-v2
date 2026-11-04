"""Stage D1 (GPU): steered generations with assumption probes (paper Section 3, eq. 3).

For each probe and each alpha, h <- h + alpha * v (paper eq. 3) on the output of model.model.layers[L], where L
is the probe's hidden_states index, exactly as upstream sample_and_generate_steered.py hooks it (one block later
than the representation the probe was fit on; kept for fidelity to the released code). Applied at every
position, prompt and generated. v is the L2-normalised ridge weight vector, as upstream. Chat template with the upstream
system prompt; sampling settings are flags. Writes {id, dataset, prompt, probe, alpha, response, finished}.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = "You are a helpful assistant. Reply to the user."  # upstream sample_and_generate_steered.py


class AddDirection:
    def __init__(self):
        self.alpha, self.v = 0.0, None

    def __call__(self, module, inputs, output):
        if self.alpha == 0.0:
            return output
        h = output[0] if isinstance(output, tuple) else output
        h = h + self.alpha * self.v.to(h.device, h.dtype)
        return (h, *output[1:]) if isinstance(output, tuple) else h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True, help="jsonl with id, dataset, user_text")
    ap.add_argument("--probe_dir", required=True, help="results/probes from train_probes.py")
    ap.add_argument("--probes", required=True, help="comma-separated <labeller>_<dim> names")
    ap.add_argument("--alphas", required=True, help="comma-separated, must include 0")
    ap.add_argument("--model", required=True)
    ap.add_argument("--max_new_tokens", type=int, required=True)
    ap.add_argument("--temperature", type=float, required=True, help="0 = greedy")
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    alphas = [float(a) for a in args.alphas.split(",")]
    assert 0.0 in alphas, "alpha 0 (unsteered) is the baseline"

    rows = [json.loads(l) for l in open(args.prompts, encoding="utf-8") if l.strip()]
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda").eval()
    eos = list({tok.eos_token_id, tok.convert_tokens_to_ids("<|eot_id|>")} - {None})
    chats = [tok.apply_chat_template([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": r["user_text"]}],
                                     tokenize=False, add_generation_prompt=True) for r in rows]

    hook = AddDirection()
    handles = []
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n_out = 0
    with open(args.out, "w", encoding="utf-8") as f, torch.no_grad():
        for name in args.probes.split(","):
            p = np.load(os.path.join(args.probe_dir, f"{name}.npz"))
            L = int(p["layer"])
            assert 0 <= L < len(model.model.layers)
            w = p["w"].astype(np.float32)
            hook.v = torch.tensor(w / np.linalg.norm(w))
            for h in handles:
                h.remove()
            handles = [model.model.layers[L].register_forward_hook(hook)]
            for alpha in alphas:
                hook.alpha = alpha
                torch.manual_seed(args.seed)
                for b in range(0, len(rows), args.batch_size):
                    batch = rows[b : b + args.batch_size]
                    enc = tok(chats[b : b + args.batch_size], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
                    gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=args.temperature > 0,
                                         temperature=args.temperature if args.temperature > 0 else None,
                                         top_p=args.top_p if args.temperature > 0 else None,
                                         pad_token_id=tok.pad_token_id, eos_token_id=eos)
                    new = gen[:, enc["input_ids"].shape[1]:]
                    for r, ids in zip(batch, new):
                        finished = bool(((ids.unsqueeze(1) == torch.tensor(eos, device=ids.device)).any(1)).any())
                        f.write(json.dumps({"id": r["id"], "dataset": r["dataset"], "prompt": r["user_text"], "probe": name,
                                            "layer": L, "alpha": alpha, "response": tok.decode(ids, skip_special_tokens=True).strip(),
                                            "finished": finished}, ensure_ascii=False) + "\n")
                        n_out += 1
                print(f"{name} alpha={alpha:+.1f} done ({n_out} rows)", flush=True)
    meta = {"prompts": args.prompts, "n_prompts": len(rows), "probes": args.probes.split(","), "alphas": alphas, "model": args.model,
            "system_prompt": SYSTEM_PROMPT, "max_new_tokens": args.max_new_tokens, "temperature": args.temperature, "top_p": args.top_p,
            "seed": args.seed, "hook": "output of model.model.layers[probe_layer] (upstream convention), all positions, h + alpha * v, v unit-norm", "rows_out": n_out,
            "argv": sys.argv}
    json.dump(meta, open(args.out + ".meta.json", "w"), indent=2)


if __name__ == "__main__":
    main()
