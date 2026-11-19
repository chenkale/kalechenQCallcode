#!/usr/bin/env python3
"""PropHunt schedule search + tropidist circuit-distance check for CSS codes.

Takes binary Hx / Hz (and optional Lx / Lz), builds a PropHunt PropagationGraph,
optimizes to a family of interleaved CNOT schedules, exports each unique schedule
as Stim + timing matrices, and certifies circuit distance with tropidist.

Run with the PropHunt venv (needs qldpc / stim / PropHunt):

  ../../tools/PropHunt/.venv/bin/python prophunt_tropidist.py
  ../../tools/PropHunt/.venv/bin/python prophunt_tropidist.py \\
      --hx hx.npy --hz hz.npy --lx lx.npy --lz lz.npy \\
      --max-iter 10 --samples 200 --threads 8 --time-limit 600
  ../../tools/PropHunt/.venv/bin/python prophunt_tropidist.py --skip-optimize
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]  # .../QC
TOOLS = REPO / "tools"
PROPHUNT = TOOLS / "PropHunt"
TROPIDIST = TOOLS / "tropidist"
TROPIDIST_PY = Path(
    os.environ.get("TROPIDIST_PYTHON", str(TROPIDIST / ".venv" / "bin" / "python"))
)

sys.path.insert(0, str(PROPHUNT))

from qldpc.codes import CSSCode  # noqa: E402
from prop_hunt.prop_graph import (  # noqa: E402
    LogicalOperator,
    PropagationGraph,
    prop_graph_from_code,
)
from prop_hunt.prop_hunt import PropHuntCompiler  # noqa: E402


def load_matrix(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".npy":
        return np.load(path)
    if path.suffix in {".csv", ".txt"}:
        return np.loadtxt(path, dtype=int)
    raise ValueError(f"Unsupported matrix file type: {path}")


def as_binary(mat: np.ndarray, name: str) -> np.ndarray:
    mat = np.asarray(mat)
    if mat.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape {mat.shape}")
    uniq = set(np.unique(mat).tolist())
    if not uniq.issubset({0, 1}):
        # Allow schedule-like matrices: treat nonzero as support.
        print(f"warning: {name} has values {sorted(uniq)[:12]}...; thresholding >0 -> 1")
        mat = (mat > 0).astype(int)
    return mat.astype(int) % 2


def build_graph(hx: np.ndarray, hz: np.ndarray,
                lx: np.ndarray | None, lz: np.ndarray | None) -> PropagationGraph:
    """Build a PropHunt graph. Prefer qldpc CSSCode (coloration init) when logicals omitted."""
    hx, hz = as_binary(hx, "Hx"), as_binary(hz, "Hz")
    if hx.shape[1] != hz.shape[1]:
        raise ValueError(f"Hx/Hz qubit count mismatch: {hx.shape} vs {hz.shape}")
    if lx is None or lz is None:
        code = CSSCode(hx, hz)
        print(f"CSSCode from Hx/Hz: n={code.num_qubits}, k={code.dimension}")
        return prop_graph_from_code(code)

    lx, lz = as_binary(lx, "Lx"), as_binary(lz, "Lz")
    if lx.shape[1] != hx.shape[1] or lz.shape[1] != hx.shape[1]:
        raise ValueError("Lx/Lz must have the same number of columns as Hx/Hz")
    # Still use coloration init from CSSCode, then swap in provided logicals.
    code = CSSCode(hx, hz)
    g = prop_graph_from_code(code)
    nx, nz, n = hx.shape[0], hz.shape[0], hx.shape[1]
    data_off = nx + nz
    g.operators = [
        LogicalOperator(i, "Z", [k + data_off for k in np.flatnonzero(lz[i])])
        for i in range(lz.shape[0])
    ] + [
        LogicalOperator(
            i + lz.shape[0], "X", [k + data_off for k in np.flatnonzero(lx[i])]
        )
        for i in range(lx.shape[0])
    ]
    print(
        f"CSSCode n={code.num_qubits}, k={code.dimension}; "
        f"using provided Lx({lx.shape[0]}) Lz({lz.shape[0]})"
    )
    return g


def schedule_fingerprint(graph: PropagationGraph) -> tuple:
    z = tuple((int(k), tuple(int(q) for q in v)) for k, v in sorted(graph.z_checks.items()))
    x = tuple((int(k), tuple(int(q) for q in v)) for k, v in sorted(graph.x_checks.items()))
    return (z, x)


def graph_to_timing_matrices(graph: PropagationGraph, nx: int, nz: int, n: int):
    """Convert interleaved PropHunt schedule to (zschedule, xschedule) timing matrices.

    Entries are 1-indexed coupling times; 0 means no coupling. Data qubits are
    remapped to 0..n-1 to match QEC/1844 conventions.
    """
    sched = graph.to_sched()
    if sched is None:
        raise ValueError("PropagationGraph has an invalid (unschedulable) CNOT order")
    data_off = nx + nz
    zsch = np.zeros((nz, n), dtype=int)
    xsch = np.zeros((nx, n), dtype=int)
    for t, (_gate, pairs) in sched.items():
        time = int(t) + 1
        for q1, q2 in pairs:
            if q2 in graph.z_checks:
                check, data = q2, q1
                row = check - nx
                zsch[row, data - data_off] = time
            elif q1 in graph.x_checks:
                check, data = q1, q2
                xsch[check, data - data_off] = time
            else:
                raise ValueError(f"Unexpected CX pair {(q1, q2)} in schedule")
    return zsch, xsch


def unique_family(history: list[PropagationGraph]) -> list[tuple[int, PropagationGraph]]:
    """Deduplicate PropHunt history; keep first occurrence of each schedule."""
    seen = set()
    out = []
    for i, g in enumerate(history):
        fp = schedule_fingerprint(g)
        if fp in seen:
            continue
        if g.to_sched() is None:
            print(f"  skip snapshot {i}: unschedulable")
            continue
        seen.add(fp)
        out.append((i, g))
    return out


def certify(stim_path: Path, out_json: Path, threads: int, time_limit: float) -> dict:
    if not TROPIDIST_PY.exists():
        raise FileNotFoundError(
            f"tropidist venv missing at {TROPIDIST_PY}; set up tools/tropidist first"
        )
    stim_path = Path(stim_path).resolve()
    out_json = Path(out_json).resolve()
    cmd = [
        str(TROPIDIST_PY), "-m", "cdist", str(stim_path),
        "--out", str(out_json),
        "--threads", str(threads),
        "--time-limit", str(time_limit),
    ]
    print(f"  tropidist: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(TROPIDIST), capture_output=True, text=True)
    if proc.stdout.strip():
        print(proc.stdout.strip())
    if proc.returncode != 0:
        if proc.stderr.strip():
            print(proc.stderr.strip(), file=sys.stderr)
        raise RuntimeError(f"tropidist failed ({proc.returncode}) on {stim_path}")
    return json.loads(out_json.read_text())


def export_and_certify(
    graph: PropagationGraph,
    label: str,
    outdir: Path,
    nx: int,
    nz: int,
    n: int,
    rounds: int,
    p: float,
    threads: int,
    time_limit: float,
    certify_bases: tuple[str, ...] = ("Z", "X"),
) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    zsch, xsch = graph_to_timing_matrices(graph, nx, nz, n)
    np.save(outdir / f"{label}_zschedule.npy", zsch)
    np.save(outdir / f"{label}_xschedule.npy", xsch)
    with open(outdir / f"{label}_tosched.pkl", "wb") as f:
        pickle.dump(graph.to_sched(), f)
    with open(outdir / f"{label}_checks.json", "w") as f:
        json.dump(
            {
                "z_checks": {str(k): v for k, v in graph.z_checks.items()},
                "x_checks": {str(k): v for k, v in graph.x_checks.items()},
                "depth": len(graph.to_sched()),
            },
            f,
            indent=2,
        )

    results = {
        "label": label,
        "depth": len(graph.to_sched()),
        "zschedule": str(outdir / f"{label}_zschedule.npy"),
        "xschedule": str(outdir / f"{label}_xschedule.npy"),
        "bases": {},
    }
    for basis in certify_bases:
        circ = graph.stim_circ(p, num_rounds=rounds, basis=basis)
        stim_path = outdir / f"{label}_{basis}.stim"
        json_path = outdir / f"{label}_{basis}.json"
        stim_path.write_text(str(circ))
        print(
            f"[{label} {basis}] stim detectors={circ.num_detectors} "
            f"obs={circ.num_observables} depth={results['depth']}"
        )
        rec = certify(stim_path, json_path, threads, time_limit)
        results["bases"][basis] = {
            "status": rec.get("status"),
            "lower_bound": rec.get("lower_bound"),
            "upper_bound": rec.get("upper_bound"),
            "seconds": rec.get("seconds"),
            "witness": rec.get("witness"),
            "stim": str(stim_path),
            "json": str(json_path),
        }
        b = results["bases"][basis]
        print(
            f"[{label} {basis}] certified [{b['lower_bound']}, {b['upper_bound']}] "
            f"status={b['status']} ({b['seconds']:.2f}s)"
        )
    lbs = [results["bases"][b]["lower_bound"] for b in certify_bases]
    ubs = [results["bases"][b]["upper_bound"] for b in certify_bases]
    results["circuit_distance_interval"] = [min(lbs), min(ubs)]
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hx", type=Path, default=HERE / "hx.npy", help="X-check matrix (.npy/.csv)")
    ap.add_argument("--hz", type=Path, default=HERE / "hz.npy", help="Z-check matrix (.npy/.csv)")
    ap.add_argument("--lx", type=Path, default=HERE / "lx.npy", help="optional X-logical matrix")
    ap.add_argument("--lz", type=Path, default=HERE / "lz.npy", help="optional Z-logical matrix")
    ap.add_argument("--outdir", type=Path, default=HERE / "prophunt_out")
    ap.add_argument("--distance", type=int, default=4, help="code distance for PropHunt")
    ap.add_argument("--rounds", type=int, default=4, help="Stim memory rounds")
    ap.add_argument("--p", type=float, default=1e-3, help="physical error rate for DEM support")
    ap.add_argument("--max-iter", type=int, default=5)
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4, help="tropidist threads")
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--skip-optimize", action="store_true", help="certify coloration baseline only")
    ap.add_argument(
        "--certify",
        choices=("all", "endpoints", "last"),
        default="all",
        help="which PropHunt snapshots to certify: all unique, first+last, or last only",
    )
    ap.add_argument("--basis", choices=("both", "Z", "X"), default="both")
    args = ap.parse_args()

    hx = as_binary(load_matrix(args.hx), "Hx")
    hz = as_binary(load_matrix(args.hz), "Hz")
    lx = load_matrix(args.lx) if args.lx and args.lx.exists() else None
    lz = load_matrix(args.lz) if args.lz and args.lz.exists() else None
    if (lx is None) ^ (lz is None):
        print("warning: only one of Lx/Lz provided; ignoring both and using CSSCode logicals")
        lx = lz = None

    nx, n = hx.shape
    nz = hz.shape[0]
    args.outdir.mkdir(parents=True, exist_ok=True)
    np.save(args.outdir / "hx.npy", hx)
    np.save(args.outdir / "hz.npy", hz)

    bases = ("Z", "X") if args.basis == "both" else (args.basis,)
    graph0 = build_graph(hx, hz, lx, lz)

    print("\n=== baseline (coloration / interleaved init) ===")
    family_results = []
    baseline = export_and_certify(
        graph0, "sched_000_baseline", args.outdir, nx, nz, n,
        args.rounds, args.p, args.threads, args.time_limit, bases,
    )
    family_results.append(baseline)
    print(f"BEFORE circuit distance in {baseline['circuit_distance_interval']}")

    if not args.skip_optimize:
        print("\n=== PropHunt optimize ===")
        hist = PropHuntCompiler(graph0).compile(
            max_iter=args.max_iter,
            distance=args.distance,
            num_samples=args.samples,
            num_workers=args.workers,
        )
        uniq = unique_family(hist)
        print(f"PropHunt history length {len(hist)}; {len(uniq)} unique schedulable snapshots")

        if args.certify == "last":
            to_certify = [uniq[-1]] if uniq else []
        elif args.certify == "endpoints":
            to_certify = [uniq[0]] + ([uniq[-1]] if len(uniq) > 1 else [])
        else:
            to_certify = uniq

        # baseline already certified; skip duplicate first snapshot if identical
        base_fp = schedule_fingerprint(graph0)
        for snap_i, g in to_certify:
            if schedule_fingerprint(g) == base_fp:
                print(f"  snapshot {snap_i}: same as baseline, skip re-certify")
                continue
            label = f"sched_{snap_i:03d}"
            print(f"\n=== certify {label} ===")
            rec = export_and_certify(
                g, label, args.outdir, nx, nz, n,
                args.rounds, args.p, args.threads, args.time_limit, bases,
            )
            family_results.append(rec)
            print(f"{label} circuit distance in {rec['circuit_distance_interval']}")

    # Rank by certified lower bound then upper bound
    ranked = sorted(
        family_results,
        key=lambda r: (
            r["circuit_distance_interval"][0],
            r["circuit_distance_interval"][1],
            -r["depth"],
        ),
        reverse=True,
    )
    summary = {
        "hx": str(args.hx),
        "hz": str(args.hz),
        "n": int(n),
        "nx": int(nx),
        "nz": int(nz),
        "distance_target": args.distance,
        "family": family_results,
        "best": ranked[0] if ranked else None,
    }
    summary_path = args.outdir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    # Compact CSV
    csv_path = args.outdir / "summary.csv"
    with open(csv_path, "w") as f:
        f.write("label,depth,Z_lb,Z_ub,X_lb,X_ub,circuit_lb,circuit_ub\n")
        for r in family_results:
            zb = r["bases"].get("Z", {})
            xb = r["bases"].get("X", {})
            clb, cub = r["circuit_distance_interval"]
            f.write(
                f"{r['label']},{r['depth']},"
                f"{zb.get('lower_bound','')},{zb.get('upper_bound','')},"
                f"{xb.get('lower_bound','')},{xb.get('upper_bound','')},"
                f"{clb},{cub}\n"
            )

    print(f"\nwrote {summary_path}")
    print(f"wrote {csv_path}")
    if ranked:
        best = ranked[0]
        print(
            f"best schedule: {best['label']} "
            f"circuit distance in {best['circuit_distance_interval']} "
            f"(depth {best['depth']})"
        )


if __name__ == "__main__":
    main()
