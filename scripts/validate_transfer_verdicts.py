"""
Validate transfer is_transfer verdicts against a corpus of real decided pairs.

The transfer prompt encodes a judgment call the model gets no other way: in this
budget a credit-card payment IS a transfer (both legs are tracked accounts), while
paying an off-budget loan or a coincidental merchant pair is not. Nothing in the
code enforces that — it lives entirely in TRANSFER_SYSTEM and render_transfer_prompt,
so a model swap can silently invert it with no test failing and no error logged.

That is exactly what happened on 2026-08-24: migrating [llm] from Qwen3.5-122B to
Qwen3.6-35B-A3B flipped every checking -> credit-card pair from paired to
transfer-rejected. The 122B had resolved the ambiguity in "checking -> credit card
= payment" correctly; the 35B read "payment" as "not a transfer" and reasoned from
accounting priors ("external liability settlement") that the prompt never states.
Runs kept succeeding, so the only symptom was card payments landing in the budget
as income and spending.

Run this before and after any change to the transfer prompts, the [llm] block, or
the model serving them. Verdicts are non-deterministic, so each case runs several
times and the report shows the spread.

Corpus layout — a JSON file, kept OUTSIDE this repo (it is public, and a real
corpus is a personal financial record):

    {
      "cases": [
        {
          "label": "cc-payment",
          "expect": true,
          "a": {"account": "Checking", "date": "2026-04-02",
                "descriptor": "External Withdrawal ...", "amount": -12345},
          "b": {"account": "Card", "date": "2026-04-01",
                "descriptor": "ACH DEPOSIT ...", "amount": 12345}
        }
      ]
    }

`offbudget` may be set on either side and defaults to false. Ground truth is each
pair's historical audit-log decision, re-checked by hand. A corpus is worth keeping
balanced: the credit-card and checking<->savings positives, plus the negatives that
must keep being rejected (off-budget loan/mortgage payments, coincidental amounts).

tests/fixtures/transfer_corpus.example.json is a synthetic corpus with the same
shape, safe to commit and enough to smoke-test the harness.

Run:
    venv/bin/python scripts/validate_transfer_verdicts.py --corpus /path/to/corpus.json
    venv/bin/python scripts/validate_transfer_verdicts.py --corpus ... --runs 5
    venv/bin/python scripts/validate_transfer_verdicts.py --corpus ... --config /srv/actual/actual-cat/config.toml

Exit 0: every case matched its expected verdict on every run.
Exit 1: at least one case disagreed — read the reasoning column before changing prompts.
"""

import argparse
import json
import os
import sys
import time
import tomllib
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from actual_cat import prompts  # noqa: E402
from actual_cat.config import _load_llm_profiles  # noqa: E402
from actual_cat.llm import LLMClient  # noqa: E402
from actual_cat.transfers import render_transfer_prompt  # noqa: E402


class _Acct:
    def __init__(self, name: str, offbudget: bool = False) -> None:
        self.name = name
        self.offbudget = offbudget


class _Txn:
    """The subset of a Transactions row that render_transfer_prompt reads."""

    def __init__(self, account: str, date: str, descriptor: str, amount: int,
                 offbudget: bool = False) -> None:
        self.account = _Acct(account, offbudget)
        self.imported_description = descriptor
        self.payee = None
        self.notes = descriptor
        self.amount = amount
        self._date = date

    def get_date(self) -> str:
        return self._date


def _txn(raw: dict[str, Any]) -> _Txn:
    return _Txn(
        account=raw["account"],
        date=raw["date"],
        descriptor=raw["descriptor"],
        amount=int(raw["amount"]),
        offbudget=bool(raw.get("offbudget", False)),
    )


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
    ap.add_argument("--corpus", required=True, help="JSON file of decided pairs")
    ap.add_argument("--config", default="config.toml", help="config.toml for the [llm] block")
    ap.add_argument("--runs", type=int, default=3, help="runs per case")
    ap.add_argument("--threshold", default="high",
                    help="confidence a positive must reach to actually pair "
                         "(transfers.apply_confidence_threshold)")
    ap.add_argument("--verbose", action="store_true", help="print the model's reasoning")
    args = ap.parse_args()

    cases = json.loads(Path(args.corpus).read_text())["cases"]
    llm = _build_llm(args.config)

    print(f"corpus {args.corpus}  cases {len(cases)}  runs {args.runs} "
          f"  threshold {args.threshold}\n")
    header = f"{'case':44} {'want':>5} {'got':>26} {'conf':>22}"
    print(header)
    print("-" * len(header))

    failures = 0
    started = time.time()

    for case in cases:
        label = case["label"]
        want = bool(case["expect"])
        txn_a, txn_b = _txn(case["a"]), _txn(case["b"])

        verdicts: list[object] = []
        confs: list[str] = []
        reasons: list[str] = []
        for _ in range(args.runs):
            out = llm.complete_json(
                prompts.TRANSFER_SYSTEM, render_transfer_prompt(txn_a, txn_b)
            )
            if "error" in out:
                verdicts.append("ERR")
                confs.append("-")
                reasons.append(str(out["error"])[:120])
                continue
            verdicts.append(out.get("is_transfer"))
            confs.append(str(out.get("confidence")))
            reasons.append(str(out.get("reasoning", ""))[:160])

        verdict_ok = all(v is want for v in verdicts)
        # A positive that lands below the threshold is still a miss: the pipeline
        # tags it #ai-suggested-transfer, which excludes it from find_uncategorized
        # for good, so nothing revisits it later.
        conf_ok = (not want) or all(c == args.threshold for c in confs)
        ok = verdict_ok and conf_ok
        if not ok:
            failures += 1

        got = ",".join(str(v) for v in verdicts)
        print(f"{label:44} {str(want):>5} {got:>26} {','.join(confs):>22}"
              f"{'' if ok else '   <-- FAIL'}")
        if args.verbose or not ok:
            for r in dict.fromkeys(reasons):
                print(f"{'':44} - {r}")

    elapsed = time.time() - started
    print(f"\n{len(cases) - failures}/{len(cases)} cases passed in {elapsed:.0f}s")
    if failures:
        print("A failure here means the deployed model no longer reads the transfer "
              "prompt the way prod depends on. Fix the prompt or pin the model — do "
              "not lower the corpus.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
