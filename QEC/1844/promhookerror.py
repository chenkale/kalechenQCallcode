import numpy as np
from itertools import combinations
from math import comb

def randomize_schedule(checks, seed=None, max_attempts=100):
    """Randomize schedules without row/column time conflicts."""
    rng = np.random.default_rng(seed)
    support = np.asarray(checks) != 0
    rows, n = support.shape

    for _ in range(max_attempts):
        schedule = np.zeros((rows, n), dtype=np.int16)
        occupied = [set() for _ in range(n)]
        success = True

        for i in rng.permutation(rows):
            qubits = list(np.flatnonzero(support[i]))
            rng.shuffle(qubits)

            times = np.arange(1, len(qubits) + 1)
            owners = {}

            def match(q, seen):
                for t in rng.permutation(times):
                    t = int(t)

                    if t in seen or t in occupied[q]:
                        continue

                    seen.add(t)

                    if t not in owners or match(owners[t], seen):
                        owners[t] = q
                        return True

                return False

            if not all(match(q, set()) for q in qubits):
                success = False
                break

            for t, q in owners.items():
                schedule[i, q] = t
                occupied[q].add(t)

        if success:
            return schedule

    raise RuntimeError("Could not construct a valid schedule")


def to_mask(bits):
    return sum(1 << i for i, b in enumerate(bits) if b)


def count_higher_weight_faults(
    schedule, opposite_checks, same_checks, max_faults=4
):
    """
    Count combinations of propagated ancilla faults.

    schedule:        Timed checks generating the errors
    opposite_checks: Checks detecting those errors
    same_checks:     Stabilizers of the error's Pauli type
    """

    schedule = np.asarray(schedule)
    opposite_checks = np.asarray(opposite_checks)
    same_checks = np.asarray(same_checks)

    n = schedule.shape[1]

    if opposite_checks.shape[1] != n or same_checks.shape[1] != n:
        raise ValueError("Inconsistent number of data qubits")

    # Opposite-type stabilizer masks
    check_masks = [
        to_mask(row != 0)
        for row in opposite_checks
    ]

    # Generate the complete same-type stabilizer group
    stabilizer_masks = {0}

    for row in same_checks:
        g = to_mask(row != 0)
        stabilizer_masks |= {
            s ^ g for s in list(stabilizer_masks)
        }

    # Precompute single-fault error masks and syndromes
    faults = []

    for i, row in enumerate(schedule):
        for t in range(1, int(row.max())):
            error = to_mask(row > t)

            syndrome = sum(
                ((error & h).bit_count() & 1) << j
                for j, h in enumerate(check_masks)
            )

            faults.append(((i, t), error, syndrome))

    results = {}

    for w in range(1, max_faults + 1):
        counts = {
            "detectable": 0,
            "identity": 0,
            "stabilizer": 0,
            "logical": 0
        }

        for combo in combinations(faults, w):
            error = 0
            syndrome = 0

            for _, e, s in combo:
                error ^= e
                syndrome ^= s

            if syndrome != 0:
                counts["detectable"] += 1

            elif error == 0:
                counts["identity"] += 1

            elif error in stabilizer_masks:
                counts["stabilizer"] += 1

            else:
                counts["logical"] += 1

        counts["undetectable_nonidentity"] = (
            counts["stabilizer"] + counts["logical"]
        )

        counts["candidate_combinations"] = (
            comb(len(faults), w)
            if w <= len(faults) else 0
        )

        results[w] = counts

    return len(faults), results