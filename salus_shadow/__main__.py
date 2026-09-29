"""`python -m salus_shadow` runs the hook (stdin -> log); `... summarize [LOG]` tallies a shadow log."""

import collections
import json
import os
import sys

from salus_shadow.hook import main


def summarize(path: str) -> None:
    calls, per_policy, errors = 0, collections.Counter(), 0
    for line in open(path):
        rec = json.loads(line)
        if "error" in rec:
            errors += 1
            continue
        calls += 1
        per_policy.update(f["policy"] for f in rec.get("findings", []))
    print(f"{calls} tool calls evaluated, {errors} hook errors")
    for policy, n in per_policy.most_common():
        print(f"  {policy:26s} {n:5d} would-block  ({100 * n / max(calls, 1):.1f} per 100 calls)")


if __name__ == "__main__":
    if sys.argv[1:2] == ["summarize"]:
        summarize(sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/.salus/shadow.jsonl"))
    else:
        sys.exit(main())
