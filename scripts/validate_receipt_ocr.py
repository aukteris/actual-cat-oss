"""
Validate receipt OCR accuracy against a corpus of real receipt images.

Prompt and image-handling changes are almost impossible to judge by eye, and the
model is non-deterministic enough that a single run per receipt proves nothing —
one real receipt in this corpus scored high/exact on one pass and low/off-by-60c
on a near-identical photo of itself. So each receipt runs several times and the
report shows the spread, not one number.

Ground truth is each receipt's own `total_cents`: the grand total is printed
large and has been read correctly on every observed run, including on partial
crops. The metric is sum(line_items) - total_cents, which should be 0.

Corpus layout:
    <corpus>/manifest.json    {"<id>": {"total_cents": 21894, "runs": 3, "label": "SAFEWAY"}}
    <corpus>/<id>.jpg

Run:
    venv/bin/python scripts/validate_receipt_ocr.py --corpus /path/to/corpus
    venv/bin/python scripts/validate_receipt_ocr.py --corpus ... --no-autocrop  # A/B

Exit 0: every receipt met its expectation (see --tolerance).
Exit 1: at least one receipt was outside tolerance on at least one run.
"""

import argparse
import json
import os
import sys
import time
import tomllib
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from actual_cat import prompts  # noqa: E402
from actual_cat.config import _load_llm_profiles  # noqa: E402
from actual_cat.llm import LLMClient  # noqa: E402
from actual_cat.receipts.ocr import ocr_receipt  # noqa: E402


def _build_llm(config_path: str) -> LLMClient:
    """LLM client from the [llm] block alone.

    Deliberately not load_config(): that requires ACTUAL_PASSWORD for the budget,
    and this harness never touches the budget.
    """
    with open(config_path, "rb") as f:
        raw = tomllib.load(f)
    text, vision = _load_llm_profiles(raw["llm"])
    return LLMClient(text, vision)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True, help="directory with manifest.json + images")
    ap.add_argument("--config", default="config.toml", help="config.toml for the [llm] block")
    ap.add_argument("--runs", type=int, default=0, help="override every manifest run count")
    ap.add_argument("--autocrop", action="store_true", default=True)
    ap.add_argument("--no-autocrop", dest="autocrop", action="store_false")
    ap.add_argument("--tolerance", type=int, default=3,
                    help="cents of sum-vs-total diff still counted as exact")
    ap.add_argument("--schema", default="Food\n  - Groceries",
                    help="category schema text passed to the prompt")
    args = ap.parse_args()

    corpus = Path(args.corpus)
    manifest = json.loads((corpus / "manifest.json").read_text())
    llm = _build_llm(args.config)

    print(f"corpus {corpus}  receipts {len(manifest)}  autocrop {args.autocrop}\n")
    header = (f"{'id':10} {'label':20} {'total':>9} {'run':>4} "
              f"{'items':>6} {'sum':>10} {'diff':>10} conf")
    print(header)
    print("-" * len(header))

    results: dict[str, list[int | None]] = {}
    started = time.time()

    for rid, meta in manifest.items():
        image = corpus / f"{rid}.jpg"
        if not image.exists():
            print(f"{rid[:8]:10} {'(image missing)':20}")
            continue
        total = meta.get("total_cents")
        runs = args.runs or meta.get("runs", 2)
        diffs: list[int | None] = []
        for run in range(1, runs + 1):
            out = ocr_receipt(
                image.read_bytes(), llm, prompts.RECEIPT_OCR_SYSTEM, args.schema,
                autocrop=args.autocrop,
            )
            if "error" in out:
                diffs.append(None)
                print(f"{rid[:8]:10} {str(meta.get('label'))[:20]:20} {'':>9} {run:>4} "
                      f"{'ERR':>6} {out['error'][:40]}")
                continue
            items = [i for i in out.get("line_items", []) if isinstance(i.get("amount_cents"), int)]
            s = sum(i["amount_cents"] for i in items)
            # Prefer the manifest's recorded total; fall back to what this run read.
            truth = total if isinstance(total, int) else out.get("total_cents", 0)
            diff = s - truth
            diffs.append(diff)
            print(f"{rid[:8]:10} {str(meta.get('label'))[:20]:20} {truth/100:9.2f} {run:>4} "
                  f"{len(items):6d} {s/100:10.2f} {diff/100:10.2f} {out.get('confidence')}")
        results[rid] = diffs

    print(f"\n{'id':10} {'label':20} {'best':>10} {'worst':>10}  verdict")
    print("-" * 62)
    failures = 0
    for rid, diffs in results.items():
        label = str(manifest[rid].get("label"))[:20]
        real = [d for d in diffs if d is not None]
        if not real:
            # Some inputs aren't receipts at all — anything can be sent to the
            # pipeline. Refusing those is correct behavior, not a failure, so a
            # corpus entry can declare that erroring is the expected outcome.
            if manifest[rid].get("expect_error"):
                print(f"{rid[:8]:10} {label:20} {'—':>10} {'—':>10}  correctly rejected")
            else:
                print(f"{rid[:8]:10} {label:20} {'—':>10} {'—':>10}  ALL RUNS ERRORED")
                failures += 1
            continue
        if manifest[rid].get("expect_error"):
            print(f"{rid[:8]:10} {label:20} {'—':>10} {'—':>10}  ACCEPTED A NON-RECEIPT")
            failures += 1
            continue
        best, worst = min(real, key=abs), max(real, key=abs)
        ok = abs(worst) <= args.tolerance
        # A receipt that was already exact must stay exact on EVERY run — those are
        # the ones auto-applying splits to the budget today.
        expect_exact = manifest[rid].get("expect_exact", False)
        # Compare against the recorded prior diff. Calling any non-exact result
        # "improved" hides real degradation: a receipt that went from -60c to -600c
        # is ten times worse, and reporting that as an improvement is how a bad
        # change gets shipped.
        prior = manifest[rid].get("prior_diff_cents")
        if ok:
            verdict = "exact"
        elif expect_exact:
            verdict = "REGRESSED"
        elif prior is None:
            verdict = "still failing"
        elif abs(worst) < abs(prior):
            verdict = f"improved (was {prior/100:+.2f})"
        else:
            verdict = f"WORSE (was {prior/100:+.2f})"
            failures += 1
        if expect_exact and not ok:
            failures += 1
        print(f"{rid[:8]:10} {label:20} {best/100:10.2f} {worst/100:10.2f}  {verdict}")

    mins = (time.time() - started) / 60
    total_runs = sum(len(v) for v in results.values())
    print(f"\n{len(results)} receipts, {total_runs} runs, {mins:.0f} min")
    if failures:
        print(f"FAIL — {failures} receipt(s) expected to stay exact did not")
        return 1
    print("PASS — no receipt marked expect_exact regressed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
