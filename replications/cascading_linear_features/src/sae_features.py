"""Stage B: SAE feature activations at layer 19 for every cascading response (paper Section 3.2).

Each row of the generations JSONL is run as a chat (user = original prompt, assistant = response).
The residual stream after transformer block 19 is encoded with the Goodfire SAE; features are mean-pooled
over the response tokens only. Writes an .npz with
  feats  [n, 65536] float16   mean SAE activation per feature over response tokens
  resid  [n, 4096]  float32   mean residual stream over response tokens (for the Appendix E probe)
  ids, levels [n] int64, n_tokens [n] int64
"""
import argparse
import json

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

SAE_REPO = "Goodfire/Llama-3.1-8B-Instruct-SAE-l19"
SAE_FILE = "Llama-3.1-8B-Instruct-SAE-l19.pth"


class SAE(torch.nn.Module):
    def __init__(self, d_in, d_hidden):
        super().__init__()
        self.encoder_linear = torch.nn.Linear(d_in, d_hidden)
        self.decoder_linear = torch.nn.Linear(d_hidden, d_in)

    def encode(self, x):
        return torch.relu(self.encoder_linear(x))

    def decode(self, f):
        return self.decoder_linear(f)


def load_sae(device):
    path = hf_hub_download(SAE_REPO, SAE_FILE)
    sd = torch.load(path, map_location="cpu")
    assert set(sd) == {"encoder_linear.weight", "encoder_linear.bias", "decoder_linear.weight",
                       "decoder_linear.bias"}, sorted(sd)
    d_hidden, d_in = sd["encoder_linear.weight"].shape
    assert (d_in, d_hidden) == (4096, 65536), (d_in, d_hidden)
    sae = SAE(d_in, d_hidden)
    sae.load_state_dict(sd)
    return sae.to(device=device, dtype=torch.bfloat16).eval()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, default=19, help="transformer block whose output is encoded")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.generations)]
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    sae = load_sae(model.device)

    feats, resid, ids, levels, n_tokens, sources = [], [], [], [], [], []
    for i, r in enumerate(rows):
        chat = [{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": r["response"]}]
        prefix = tok.apply_chat_template(chat[:1], add_generation_prompt=True, tokenize=True)
        full = tok.apply_chat_template(chat, tokenize=True)
        assert full[: len(prefix)] == prefix, "response tokens must extend the prompt tokens"
        # Response span: tokens after the assistant header, excluding the trailing <|eot_id|>.
        start, end = len(prefix), len(full)
        if full[-1] in tok.all_special_ids:
            end -= 1
        assert end - start >= 1, r["id"]
        hs = model(torch.tensor([full], device=model.device), output_hidden_states=True).hidden_states
        # hidden_states[k] is the output of block k-1, so block `layer` is hidden_states[layer + 1].
        h = hs[args.layer + 1][0, start:end]
        f = sae.encode(h)
        feats.append(f.float().mean(0).half().cpu().numpy())
        resid.append(h.float().mean(0).cpu().numpy())
        ids.append(r["id"]); levels.append(r["level"]); n_tokens.append(end - start); sources.append(r["source"])
        if i % 100 == 0:
            l0 = (f > 0).float().sum(1).mean().item()
            print(f"{i}/{len(rows)} response tokens={end - start} mean L0={l0:.1f}", flush=True)

    np.savez(args.out, feats=np.stack(feats), resid=np.stack(resid), ids=np.array(ids),
             levels=np.array(levels), n_tokens=np.array(n_tokens), sources=np.array(sources))
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
