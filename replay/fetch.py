"""Download a seeded sample of public SWE-agent trajectories.

Source: the official SWE-bench leaderboard artifacts (github.com/SWE-bench/experiments),
whose trajectories live in the public, unauthenticated S3 bucket `swe-bench-submissions`.

We take the same N SWE-bench Verified instances from three SWE-agent runs that span
three action formats and two agent generations, so the adapter is exercised on each:

  * claude-3.5-sonnet  SWE-agent 0.x (2024-06): legacy ACI (`open`, `edit N:M`, `create`)
  * claude-4-sonnet    SWE-agent 1.x (2025-05): function calling (`bash`, `str_replace_editor`)
  * swe-agent-lm-32b   SWE-agent 1.x (2025-05): open-weights model, text-parsed actions

For every sampled instance we also fetch the repository's file list at `base_commit`
(GitHub git-trees API) so the replay knows which files existed before the agent started.

Usage:  python replay/fetch.py [--per-run 17] [--seed 20260929]
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import re
import shutil
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / ".cache" / "trajs"
DATA = ROOT / "data"
BUCKET = "https://swe-bench-submissions.s3.amazonaws.com"

RUNS = {
    "claude-3.5-sonnet": "verified/20240620_sweagent_claude3.5sonnet/trajs/",
    "claude-4-sonnet": "verified/20250522_sweagent_claude-4-sonnet-20250514/trajs/",
    "swe-agent-lm-32b": "verified/20250511_sweagent_lm_32b/trajs/",
}


def _get(url: str, headers: dict[str, str] | None = None) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "salus-shadow-replay", **(headers or {})})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def list_trajs(prefix: str) -> dict[str, str]:
    """instance_id -> S3 key of its .traj file (handles both flat and per-instance layouts)."""
    keys: dict[str, str] = {}
    token = None
    while True:
        q = {"list-type": "2", "prefix": prefix}
        if token:
            q["continuation-token"] = token
        xml = _get(f"{BUCKET}/?{urllib.parse.urlencode(q)}").decode()
        for key in re.findall(r"<Key>(.*?)</Key>", xml):
            if key.endswith(".traj"):
                keys[Path(key).stem] = key
        m = re.search(r"<NextContinuationToken>(.*?)</NextContinuationToken>", xml)
        if not m:
            return keys
        token = m.group(1)


def swebench_verified() -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for offset in range(0, 500, 100):
        url = (
            "https://datasets-server.huggingface.co/rows?dataset=princeton-nlp/SWE-bench_Verified"
            f"&config=default&split=test&offset={offset}&length=100"
        )
        for r in json.loads(_get(url))["rows"]:
            row = r["row"]
            rows[row["instance_id"]] = {"repo": row["repo"], "base_commit": row["base_commit"]}
    return rows


def _github_token() -> str | None:
    if os.environ.get("GITHUB_TOKEN"):
        return os.environ["GITHUB_TOKEN"]
    if shutil.which("gh"):
        out = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True)
        return out.stdout.strip() or None
    return None


def repo_manifest(repo: str, commit: str, token: str | None) -> list[str]:
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    tree = json.loads(_get(f"https://api.github.com/repos/{repo}/git/trees/{commit}?recursive=1", headers))
    if tree.get("truncated"):
        raise RuntimeError(f"git tree for {repo}@{commit} is truncated")
    return sorted(e["path"] for e in tree["tree"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-run", type=int, default=17)
    ap.add_argument("--seed", type=int, default=20260929)
    args = ap.parse_args()

    listings = {run: list_trajs(prefix) for run, prefix in RUNS.items()}
    for run, keys in listings.items():
        print(f"{run:18s} {len(keys)} trajectories")
    common = sorted(set.intersection(*(set(k) for k in listings.values())))
    sample = sorted(random.Random(args.seed).sample(common, args.per_run))
    print(f"{len(common)} instances in all runs; sampled {len(sample)} (seed {args.seed})")

    meta = swebench_verified()
    token = _github_token()
    (DATA / "manifests").mkdir(parents=True, exist_ok=True)
    for iid in sample:
        dest = DATA / "manifests" / f"{iid}.txt.gz"
        if not dest.exists():
            paths = repo_manifest(meta[iid]["repo"], meta[iid]["base_commit"], token)
            with gzip.open(dest, "wt") as fh:
                fh.write("\n".join(paths) + "\n")
        for run, keys in listings.items():
            out = CACHE / run / f"{iid}.traj"
            if not out.exists():
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(_get(f"{BUCKET}/{urllib.parse.quote(keys[iid])}"))
        print("fetched", iid)

    (DATA / "sample.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "source": "s3://swe-bench-submissions (SWE-bench/experiments leaderboard artifacts)",
                "runs": RUNS,
                "instances": {iid: meta[iid] for iid in sample},
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
