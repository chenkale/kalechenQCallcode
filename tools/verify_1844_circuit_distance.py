#!/usr/bin/env python3
"""PropHunt -> tropidist verification for the [[18,4,4]] bivariate bicycle code.

Builds a PropHunt PropagationGraph for BB([[18,4,4]]), optionally optimizes the
CNOT schedule, exports Stim memory circuits (Z and X bases), and certifies
circuit distance with tropidist.

Run with PropHunt's venv (needs qldpc / stim / PropHunt imports):

  tools/PropHunt/.venv/bin/python tools/verify_1844_circuit_distance.py
  tools/PropHunt/.venv/bin/python tools/verify_1844_circuit_distance.py --skip-optimize
  tools/PropHunt/.venv/bin/python tools/verify_1844_circuit_distance.py \\
      --max-iter 10 --samples 200 --threads 8 --time-limit 600
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import sympy

TOOLS = Path(__file__).resolve().parent
PROPHUNT = TOOLS / "PropHunt"
TROPIDIST = TOOLS / "tropidist"
TROPIDIST_PY = TROPIDIST / ".venv" / "bin" / "python"

sys.path.insert(0, str(PROPHUNT))

from qldpc.codes import BBCode  # noqa: E402
from prop_hunt.prop_graph import prop_graph_from_code  # noqa: E402
from prop_hunt.prop_hunt import PropHuntCompiler  # noqa: E402


def bb_18_4_4():
    """[[18,4,4]] BB code: A = x + 1 + y^2, B = y + 1 + x^2 on Z_3 x Z_3."""
    x, y = sympy.symbols("x y")
    return BBCode({x: 3, y: 3}, x + 1 + y**2, y + 1 + x**2)


def certify(stim_path: Path, out_json: Path, threads: int, time_limit: float) -> dict:
    if not TROPIDIST_PY.exists():
        raise FileNotFoundError(
            f"tropidist venv missing at {TROPIDIST_PY}; set up tools/tropidist first"
        )
    cmd = [
        str(TROPIDIST_PY),
        "-m",
        "cdist",
        str(stim_path),
        "--out",
        str(out_json),
        "--threads",
        str(threads),
        "--time-limit",
        str(time_limit),
    ]
    print(f"  tropidist: {' '.join(cmd)}")
    proc = subprocess.run(
        cmd,
        cwd=str(TROPIDIST),
        capture_output=True,
        text=True,
    )
    if proc.stdout.strip():
        print(proc.stdout.strip())
    if proc.returncode != 0:
        if proc.stderr.strip():
            print(proc.stderr.strip(), file=sys.stderr)
        raise RuntimeError(f"tropidist failed ({proc.returncode}) on {stim_path}")
    return json.loads(out_json.read_text())


def export_and_certify(graph, label: str, outdir: Path, rounds: int, p: float,
                       threads: int, time_limit: float) -> dict[str, dict]:
    results = {}
    for basis in ("Z", "X"):
        circ = graph.stim_circ(p, num_rounds=rounds, basis=basis)
        stim_path = outdir / f"{label}_{basis}.stim"
        json_path = outdir / f"{label}_{basis}.json"
        stim_path.write_text(str(circ))
        print(
            f"[{label} {basis}] wrote {stim_path.name} "
            f"(detectors={circ.num_detectors}, obs={circ.num_observables})"
        )
        rec = certify(stim_path, json_path, threads, time_limit)
        results[basis] = {
            "status": rec.get("status"),
            "lower_bound": rec.get("lower_bound"),
            "upper_bound": rec.get("upper_bound"),
            "seconds": rec.get("seconds"),
            "stim": str(stim_path),
            "json": str(json_path),
        }
        print(
            f"[{label} {basis}] certified "
            f"[{results[basis]['lower_bound']}, {results[basis]['upper_bound']}] "
            f"status={results[basis]['status']} ({results[basis]['seconds']:.2f}s)"
        )
    return results


def summarize(tag: str, results: dict[str, dict]) -> None:
    lbs = [results[b]["lower_bound"] for b in ("Z", "X")]
    ubs = [results[b]["upper_bound"] for b in ("Z", "X")]
    print(
        f"{tag}: Z=[{results['Z']['lower_bound']},{results['Z']['upper_bound']}] "
        f"X=[{results['X']['lower_bound']},{results['X']['upper_bound']}] "
        f"-> circuit distance in [{min(lbs)}, {min(ubs)}]"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", type=Path, default=TOOLS / "out_1844_verify")
    ap.add_argument("--rounds", type=int, default=4, help="syndrome rounds in Stim circuit")
    ap.add_argument("--p", type=float, default=1e-3, help="physical error rate for DEM support")
    ap.add_argument("--skip-optimize", action="store_true", help="only certify baseline schedule")
    ap.add_argument("--max-iter", type=int, default=5)
    ap.add_argument("--samples", type=int, default=100, help="PropHunt ambiguous-subgraph samples")
    ap.add_argument("--workers", type=int, default=4, help="PropHunt sampler workers")
    ap.add_argument("--threads", type=int, default=4, help="tropidist threads")
    ap.add_argument("--time-limit", type=float, default=120.0, help="tropidist seconds per circuit")
    ap.add_argument("--distance", type=int, default=4, help="code distance passed to PropHunt")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    print(f"outdir: {args.outdir}")

    code = bb_18_4_4()
    print(f"code: [[{code.num_qubits},{code.dimension},{args.distance}]]")
    graph0 = prop_graph_from_code(code)

    print("\n=== baseline (PropHunt coloration schedule) ===")
    baseline = export_and_certify(
        graph0, "baseline", args.outdir, args.rounds, args.p, args.threads, args.time_limit
    )
    summarize("BEFORE", baseline)

    optimized = None
    if not args.skip_optimize:
        print("\n=== PropHunt optimize ===")
        hist = PropHuntCompiler(graph0).compile(
            max_iter=args.max_iter,
            distance=args.distance,
            num_samples=args.samples,
            num_workers=args.workers,
        )
        graph_opt = hist[-1]
        print(f"PropHunt returned {len(hist)} schedule snapshots; certifying last")
        print("\n=== optimized schedule ===")
        optimized = export_and_certify(
            graph_opt, "optimized", args.outdir, args.rounds, args.p, args.threads, args.time_limit
        )
        summarize("AFTER ", optimized)

    summary = {"baseline": baseline, "optimized": optimized}
    summary_path = args.outdir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {summary_path}")


if __name__ == "__main__":
    main()
