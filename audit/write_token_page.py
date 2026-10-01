"""Write reports/audit_existing_methods/token_heatmaps.html from token_scores.json (per-token detector scores).

Run from the repo root: python -m audit.write_token_page
"""
import argparse
import json

import numpy as np

from utils.io import REPO_ROOT, check_counts, read_json, write_meta

DIR = REPO_ROOT / "reports" / "audit_existing_methods"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    data = read_json(DIR / "token_scores.json")
    scale = {}
    for det in data["detectors"]:
        vals = np.concatenate([np.abs(s["scores"][det["name"]]) for s in data["samples"]])
        scale[det["name"]] = float(np.percentile(vals, 98))
    payload = {"detectors": data["detectors"], "scale": scale, "samples": data["samples"], "seed": data["seed"],
               "max_response_tokens": data["max_response_tokens"]}
    page = (DIR.parent.parent / "audit" / "token_page_template.html").read_text(encoding="utf-8")
    out = DIR / "token_heatmaps.html"
    out.write_text(page.replace("/*DATA*/null", json.dumps(payload)), encoding="utf-8")
    n = len(data["samples"])
    write_meta(out, [DIR / "token_scores.json"], args, check_counts(n, {}, n, out.name), {"model": "meta-llama/Llama-3.1-8B-Instruct"})
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
