"""Circuit-level Pauli fault enumeration for concatenated Z-then-X CSS schedules.

For each CNOT site: one of 15 non-identity Pauli errors *after* that gate.
Optional sites: X/Y/Z on initial data, relevant ancilla preparation fault,
readout bit-flip, and X/Y/Z on idle data/ancilla after each schedule window.

Each fault combination uses DISTINCT sites (no two Pauli outcomes at one gate).
This is a finite circuit-level fault model, not an enumeration of all physical
microscopic noise processes. It models a single Z-check round followed by a
single X-check round with fresh ancillas and no active feedback.
"""
from dataclasses import dataclass
from itertools import combinations, product
import numpy as np

PAULIS = "IXYZ"


@dataclass(frozen=True)
class FaultEvent:
    name: str
    x: int                 # final data X component (Y contributes X and Z)
    z: int                 # final data Z component
    measured: int          # observed check-result flips (Z checks then X checks)
    data_syndrome: int     # syndrome of final data Pauli against ideal checks


@dataclass(frozen=True)
class FaultSite:
    label: str
    events: tuple          # alternative Pauli outcomes at this one site


def mask(bits):
    return sum(1 << q for q, b in enumerate(bits) if b)


def gf2_span(masks):
    """All bitwise XOR products of a list of generator masks."""
    span = {0}
    for g in masks:
        span |= {v ^ g for v in tuple(span)}
    return span


def validate(z, x):
    z, x = np.asarray(z), np.asarray(x)
    if z.ndim != 2 or x.ndim != 2 or z.shape[1] != x.shape[1]:
        raise ValueError("z and x must be 2D schedules with equal data width")
    if np.any(z < 0) or np.any(x < 0) or np.any(z != np.floor(z)) or np.any(x != np.floor(x)):
        raise ValueError("Schedules must contain nonnegative integer times")
    for tag, S in (("Z", z), ("X", x)):
        for i, row in enumerate(S):
            used = row[row > 0]
            if len(used) != len(set(used)):
                raise ValueError(f"Repeated time in {tag} check row {i}")
        for q, col in enumerate(S.T):
            used = col[col > 0]
            if len(used) != len(set(used)):
                raise ValueError(f"Conflicting {tag}-CNOTs on data qubit {q}")
    hx, hz = (x > 0).astype(np.uint8), (z > 0).astype(np.uint8)
    if np.any((hx @ hz.T) % 2):
        raise ValueError("X and Z checks must commute (CSS construction)")
    return z.astype(int), x.astype(int), hx, hz


def make_fault_sites(z, x, include_initial=True,
                     include_preparation=True, include_measurement=True,
                     include_idle=True):
    """Return (sites, hx_masks, hz_masks) for one Z-then-X extraction round."""
    z, x, hx, hz = validate(z, x)
    n = z.shape[1]
    nz, nx = z.shape[0], x.shape[0]
    hz_masks = [mask(row) for row in hz]
    hx_masks = [mask(row) for row in hx]
    data_mask = (1 << n) - 1

    # Operation: (kind, qubit_a, qubit_b, measurement_index)
    operations = []
    # Raw site: (label, position just after operations[:position], outcomes)
    raw_sites = []

    def add_site(label, outcomes):
        raw_sites.append((label, len(operations), tuple(outcomes)))

    def p1(q, p):
        bit = 1 << int(q)
        return ((bit if p in "XY" else 0),
                (bit if p in "YZ" else 0), 0)

    if include_initial:
        for q in range(n):
            add_site(f"initial data[{q}]",
                     [(p, *p1(q, p)) for p in "XYZ"])

    def run_phase(S, phase):
        nchecks = S.shape[0]
        z_phase = (phase == "Z")
        offset = n if z_phase else n + nz
        meas_offset = 0 if z_phase else nz

        # A freshly prepared Z-check ancilla is |0>; X is the relevant
        # preparation error. A freshly prepared X-check ancilla is |+>; Z is.
        for i in range(nchecks):
            a = offset + i
            operations.append(("PZ" if z_phase else "PX", a, None, None))
            if include_preparation:
                p = "X" if z_phase else "Z"
                add_site(f"{phase} anc[{i}] prep", [(p, *p1(a, p))])

        active_qubits = set(range(n)) | set(range(offset, offset + nchecks))
        for t in range(1, int(S.max()) + 1):
            pairs = []
            for i, row in enumerate(S):
                for q in np.flatnonzero(row == t):
                    q = int(q)
                    a = offset + i
                    control, target = (q, a) if z_phase else (a, q)
                    pairs.append((i, q, control, target))

            touched = set()
            for i, q, control, target in pairs:
                operations.append(("CX", control, target, None))
                touched.update((control, target))
                variants = []
                for pc, pt in product(PAULIS, repeat=2):
                    if pc == pt == "I":
                        continue
                    xc, zc, _ = p1(control, pc)
                    xt, zt, _ = p1(target, pt)
                    variants.append((pc + pt, xc ^ xt, zc ^ zt, 0))
                add_site(f"{phase}[{i}] t={t} CNOT d[{q}] a[{i}]", variants)

            if include_idle:
                for q in sorted(active_qubits - touched):
                    role = "data" if q < n else f"{phase} anc"
                    add_site(f"{phase} t={t} idle {role}[{q if q < n else q-offset}]",
                             [(p, *p1(q, p)) for p in "XYZ"])

        # Measurement after *all* gates of this phase.
        for i in range(nchecks):
            a = offset + i
            index = meas_offset + i
            operations.append(("MZ" if z_phase else "MX", a, None, index))
            if include_measurement:
                add_site(f"{phase}[{i}] readout", [("flip", 0, 0, 1 << index)])

    run_phase(z, "Z")
    run_phase(x, "X")

    def final_frame(start, px, pz, measured):
        """Propagate a Pauli frame through the circuit after an insertion."""
        for kind, a, b, j in operations[start:]:
            ba = 1 << a
            if kind == "CX":
                bb = 1 << b
                # CX conjugation: X_control -> X_control X_target
                #                  Z_target  -> Z_control Z_target
                if px & ba:
                    px ^= bb
                if pz & bb:
                    pz ^= ba
            elif kind in ("PZ", "PX"):
                px &= ~ba
                pz &= ~ba
            elif kind == "MZ":
                if px & ba:
                    measured ^= 1 << j
                px &= ~ba
                pz &= ~ba
            elif kind == "MX":
                if pz & ba:
                    measured ^= 1 << j
                px &= ~ba
                pz &= ~ba
        return px & data_mask, pz & data_mask, measured

    def data_syndrome(px, pz):
        # Z-check outcomes flip if the final data X component overlaps.
        # X-check outcomes flip if the final data Z component overlaps.
        result = 0
        for j, h in enumerate(hz_masks):
            result |= ((px & h).bit_count() & 1) << j
        for j, h in enumerate(hx_masks, start=nz):
            result |= ((pz & h).bit_count() & 1) << j
        return result

    sites = []
    for label, start, outcomes in raw_sites:
        compiled = []
        for name, px, pz, m in outcomes:
            fx, fz, fm = final_frame(start, px, pz, m)
            compiled.append(FaultEvent(name, fx, fz, fm, data_syndrome(fx, fz)))
        sites.append(FaultSite(label, tuple(compiled)))
    return sites, hx_masks, hz_masks


def case_counts(sites, max_faults):
    """Number of Pauli outcome combinations of distinct physical sites."""
    counts = [1] + [0] * max_faults
    for site in sites:
        for w in range(max_faults, 0, -1):
            counts[w] += counts[w-1] * len(site.events)
    return {w: counts[w] for w in range(1, max_faults + 1)}


def iter_fault_sets(sites, hx_masks, hz_masks, weight, only_logical=False):
    """Stream actual fault-location/outcome combinations (no giant storage)."""
    x_stabilizers = gf2_span(hx_masks)
    z_stabilizers = gf2_span(hz_masks)

    for selected in combinations(sites, weight):
        for outcomes in product(*(s.events for s in selected)):
            fx = fz = observed = final_syn = 0
            for e in outcomes:
                fx ^= e.x
                fz ^= e.z
                observed ^= e.measured
                final_syn ^= e.data_syndrome

            if fx == 0 and fz == 0:
                kind = "identity"
            elif final_syn:
                kind = "detectable_data"
            elif fx in x_stabilizers and fz in z_stabilizers:
                kind = "stabilizer"
            else:
                kind = "logical"

            if only_logical and kind != "logical":
                continue
            yield {
                "faults": tuple((s.label, e.name) for s, e in zip(selected, outcomes)),
                "class": kind,
                "data_x": fx,
                "data_z": fz,
                "observed": observed,
                "final_data_syndrome": final_syn,
                "silent_logical": kind == "logical" and observed == 0,
            }


def count_faults(sites, hx_masks, hz_masks, max_faults=2, max_cases=5_000_000):
    """Exhaustively classify fault sets up to a practical limit."""
    expected = case_counts(sites, max_faults)
    oversized = {w: n for w, n in expected.items() if n > max_cases}
    if oversized:
        raise ValueError(
            f"Too many combinations for exhaustive enumeration: {oversized}. "
            "Reduce max_faults, disable idle sites, or use a more advanced "
            "search/counting algorithm."
        )

    output = {}
    for w in range(1, max_faults+1):
        counts = {
            "identity": 0,
            "stabilizer": 0,
            "detectable_data": 0,
            "logical": 0,
            "silent_logical": 0,
            "observed_zero": 0,
            "total": expected[w],
        }
        for result in iter_fault_sets(sites, hx_masks, hz_masks, w):
            counts[result["class"]] += 1
            counts["silent_logical"] += result["silent_logical"]
            counts["observed_zero"] += (result["observed"] == 0)
        assert sum(counts[k] for k in (
            "identity", "stabilizer", "detectable_data", "logical")) == expected[w]
        output[w] = counts
    return output


def bb_18_4_4():
    """One standard [[18,4,4]] CSS BB construction with 9 checks per type."""
    eye = np.eye(9, dtype=np.uint8)
    xx = np.zeros((9,9), dtype=np.uint8)
    yy = np.zeros((9,9), dtype=np.uint8)
    for a in range(3):
        for b in range(3):
            i = a + 3*b
            xx[i, (a+1) % 3 + 3*b] = 1
            yy[i, a + 3*((b+1) % 3)] = 1
    A = (eye + xx + yy @ yy) % 2
    B = (eye + yy + xx @ xx) % 2
    return np.concatenate([A,B], axis=1), np.concatenate([B.T,A.T], axis=1)


def randomize_schedule(checks, rng=None, retries=1000):
    """Assign local time labels 1..check_weight without data-time conflicts."""
    if rng is None:
        rng = np.random.default_rng()
    checks = (np.asarray(checks) != 0)
    m, n = checks.shape
    for _ in range(retries):
        S = np.zeros((m,n), dtype=int)
        occupied = [set() for _ in range(n)]
        success = True
        for i in rng.permutation(m):
            qubits = rng.permutation(np.flatnonzero(checks[i]))
            times = range(1, len(qubits)+1)
            owners = {}
            def assign(q, seen):
                for t in rng.permutation(list(times)):
                    t = int(t)
                    if t in seen or t in occupied[q]:
                        continue
                    seen.add(t)
                    if t not in owners or assign(owners[t], seen):
                        owners[t] = q
                        return True
                return False
            if not all(assign(q, set()) for q in qubits):
                success = False
                break
            for t,q in owners.items():
                S[i,q] = t
                occupied[q].add(t)
        if success:
            return S
    raise RuntimeError("No valid schedule found")


if __name__ == "__main__":
    hx, hz = bb_18_4_4()
    rng = np.random.default_rng(10)
    z = randomize_schedule(hz, rng)
    x = randomize_schedule(hx, rng)
    sites, hx_masks, hz_masks = make_fault_sites(z, x)
    print(f"{len(sites)} physical fault sites")
    print("Number of fault-outcome combinations:", case_counts(sites, 3))
    print("Counts for weight 1 and 2:")
    print(count_faults(sites, hx_masks, hz_masks, max_faults=2))
    print("First 3 weight-2 logical fault combinations:")
    from itertools import islice
    for example in islice(iter_fault_sets(sites, hx_masks, hz_masks, 2,
                                        only_logical=True), 3):
        print(example)
