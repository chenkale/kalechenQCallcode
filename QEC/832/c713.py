import CircuitBuilder
import numpy as np
import os
from tqdm import tqdm

numruns = 1
numrandomizations = 1
nshots = 1000000
outdir = 'sim713'

# Steane [[7,1,3]] CSS stabilizers (0-indexed data qubits):
#   Z/X_0: 0,1,2,3
#   Z/X_1: 0,1,4,5
#   Z/X_2: 0,2,4,6
# Schedule entries are coupling time windows; 0 means no coupling.
defaultzschedule = np.asarray([
    [1, 2, 3, 4, 0, 0, 0, 0],
    [2, 1, 0, 0, 3, 4, 0, 0],
    [3, 0, 1, 0, 2, 0, 4, 0],
    [0, 0, 0, 0, 0, 0, 0, 1]
])
defaultxschedule = np.asarray([
    [1, 2, 3, 4, 0, 0, 0, 0],
    [2, 1, 0, 0, 3, 4, 0, 0],
    [3, 0, 1, 0, 2, 0, 4, 0],
])

def checkvalidschedule(schedule):
    for row in schedule:
        seen = set()
        for val in row:
            if val != 0:
                if val in seen:
                    return False
                seen.add(val)
    for col_idx in range(schedule.shape[1]):
        seen = set()
        for row_idx in range(schedule.shape[0]):
            val = schedule[row_idx][col_idx]
            if val != 0:
                if val in seen:
                    return False
                seen.add(val)
    return True

def randomize_schedule(schedule, max_attempts=1000):
    """
    Returns a randomized copy of `schedule` that is valid, or None if failed after max_attempts.
    Does NOT mutate the input `schedule` in place.
    """
    if max_attempts == 0:
        return None
    schedule_copy = np.copy(schedule)
    for i in range(len(schedule_copy)):
        nonzero_indices = [j for j, v in enumerate(schedule_copy[i]) if v != 0]
        nonzeros = list(range(1, max(schedule_copy[i]) + 1)) if nonzero_indices else []
        np.random.shuffle(nonzeros)
        for idx in nonzero_indices:
            schedule_copy[i][idx] = nonzeros.pop()
    if checkvalidschedule(schedule_copy):
        return schedule_copy, max_attempts
    else:
        return randomize_schedule(schedule, max_attempts - 1)

def windowFiller(schedule, nwindow):
    thiswindow = []
    for row in schedule:
        if nwindow in row:
            thiswindow.append(np.where(row == nwindow)[0][0])
        else:
            thiswindow.append(None)
    return thiswindow

def c713scheduleable(er=0, nrounds=1, zschedule=defaultzschedule, xschedule=defaultxschedule):
    """
    Schedulable Steane [[7,1,3]] padded to 8 data qubits.
    Qubits 0-6 are the Steane block. Qubit 7 is an extra data qubit whose only
    check is the last Z row of the default schedule. Ancillas start at qubit 8:
    one per Z row, then one per X row.
    """
    zschedule, xschedule = np.asarray(zschedule), np.asarray(xschedule)
    ndata = zschedule.shape[1]
    cb = CircuitBuilder.CircuitBuilder(er, er, er, er, er, er)
    cb.al('R ' + ' '.join(str(i) for i in range(ndata)))
    cb.al('H ' + ' '.join(str(i) for i in range(ndata)))
    nancillas = len(zschedule) + len(xschedule)
    ancillastring = 'R ' + ' '.join([f'{i}' for i in range(ndata, ndata + nancillas)])
    cb.al(ancillastring)
    cb.set_ancilla_qubits(
        qubitslistz=[ndata + i for i in range(len(zschedule))],
        qubitslistx=[ndata + len(zschedule) + i for i in range(len(xschedule))],
    )
    # Warmup round
    for windowz in range(1, np.max(zschedule) + 1):
        cb.couple_z_check(windowFiller(zschedule, windowz), False)
    cb.al('MR(0) ' + ' '.join([f'{i}' for i in cb.ancilla_qubitsz]))
    cb.al('H ' + ' '.join([f'{i}' for i in cb.ancilla_qubitsx]))
    for windowx in range(1, np.max(xschedule) + 1):
        cb.couple_x_check(windowFiller(xschedule, windowx), False)
    cb.al('H ' + ' '.join([f'{i}' for i in cb.ancilla_qubitsx]))
    cb.al('MR(0) ' + ' '.join([f'{i}' for i in cb.ancilla_qubitsx]))
    cb.al('R ' + ' '.join([f'{i}' for i in cb.ancilla_qubitsz]))
    # Noise rounds
    cb.begin_loop(nrounds)
    for windowz in range(1, np.max(zschedule) + 1):
        cb.time_tick()
        cb.couple_z_check(windowFiller(zschedule, windowz), True)
    cb.al('MR(0) ' + ' '.join([f'{i}' for i in cb.ancilla_qubitsz]))
    cb.al('R ' + ' '.join([f'{i}' for i in cb.ancilla_qubitsx]))
    cb.al('H ' + ' '.join([f'{i}' for i in cb.ancilla_qubitsx]))
    for windowx in range(1, np.max(xschedule) + 1):
        cb.time_tick()
        cb.couple_x_check(windowFiller(xschedule, windowx), True)
    cb.add_detectors([[-1 - i, -1 - i - nancillas] for i in range(len(cb.ancilla_qubitsz))])
    cb.al('H ' + ' '.join([f'{i}' for i in cb.ancilla_qubitsx]))
    cb.al('MR(0) ' + ' '.join([f'{i}' for i in cb.ancilla_qubitsx]))
    cb.add_detectors([[-1 - i, -1 - i - nancillas] for i in range(len(cb.ancilla_qubitsx))])
    cb.al('R ' + ' '.join([f'{i}' for i in cb.ancilla_qubitsz]))
    cb.end_loop()
    # Cooldown round
    for windowz in range(1, np.max(zschedule) + 1):
        cb.couple_z_check(windowFiller(zschedule, windowz), False)
    cb.al('MR(0) ' + ' '.join([f'{i}' for i in cb.ancilla_qubitsz]))
    cb.al('H ' + ' '.join([f'{i}' for i in cb.ancilla_qubitsx]))
    for windowx in range(1, np.max(xschedule) + 1):
        cb.couple_x_check(windowFiller(xschedule, windowx), False)
    cb.al('H ' + ' '.join([f'{i}' for i in cb.ancilla_qubitsx]))
    cb.al('MR(0) ' + ' '.join([f'{i}' for i in cb.ancilla_qubitsx]))
    cb.al('R ' + ' '.join([f'{i}' for i in cb.ancilla_qubitsz]))
    # Final measurement of all physical qubits
    cb.al('H ' + ' '.join(str(i) for i in range(ndata)))
    cb.al('MR(0) ' + ' '.join(str(i) for i in range(ndata)))
    return cb

def detector_builder(mseq, syndromelength=7, nphysicalqubits=8, nrounds=1, mode='binary'):
    if len(mseq) != syndromelength * (nrounds + 1) + nphysicalqubits:
        raise ValueError(f"Measurement sequence length must be equal to syndromelength * (nrounds + 1) + nphysicalqubits")
    syndromes, finalmeasurement = mseq[:-nphysicalqubits], mseq[-nphysicalqubits:]
    if mode == 'string':
        det = []
        for i in range(nrounds):
            det.append(''.join([str(0 if x == y else 1) for x, y in zip(syndromes[i*syndromelength:(i+1)*syndromelength], syndromes[(i+1)*syndromelength:(i+2)*syndromelength])]))
    elif mode == 'binary':
        det = np.zeros(nrounds * syndromelength, dtype=np.bool_)
        for i in range(nrounds):
            det[i*syndromelength:(i+1)*syndromelength] = np.array([0 if x == y else 1 for x, y in zip(syndromes[i*syndromelength:(i+1)*syndromelength], syndromes[(i+1)*syndromelength:(i+2)*syndromelength])])
    else:
        raise ValueError(f"Invalid mode: {mode}")
    return det, finalmeasurement

def passes_postselection(detector):
    return np.sum(detector) < 1

def toDecimal(binary):
    return int(binary, 2)

def negate(string):
    return ''.join(['1' if x == '0' else '0' for x in string])

def logical_operators_713():
    """X-logical coset of the padded Steane code, as length-8 boolean vectors.

    Qubits 0-6 carry the Steane logical X = X^{\\otimes 7}. Qubit 7 is excluded so
    every operator commutes with the weight-1 Z check on that qubit. The eight
    vectors are that logical multiplied by each element of the X-stabilizer group.
    """
    xgens = np.array([
        [1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 0, 0, 1, 1, 0, 0],
        [1, 0, 1, 0, 1, 0, 1, 0],
    ], dtype=int)
    logical = np.array([1, 1, 1, 1, 1, 1, 1, 0], dtype=int)
    ops = []
    for mask in range(1 << len(xgens)):
        op = logical.copy()
        for i in range(len(xgens)):
            if mask & (1 << i):
                op ^= xgens[i]
        ops.append(op.astype(bool))
    return ops

def make_schedulelist_713(n=100, seed=0):
    """n distinct randomizations of the default padded Steane Z and X schedules."""
    np.random.seed(seed)
    used = set()
    schedulelist = []
    attempts = 0
    while len(schedulelist) < n:
        attempts += 1
        if attempts > n * 1000:
            raise RuntimeError(f"Only found {len(schedulelist)} unique schedules in {attempts} attempts")
        z = randomize_schedule(defaultzschedule, 1000)
        x = randomize_schedule(defaultxschedule, 1000)
        if z is None or x is None:
            continue
        z, x = z[0], x[0]
        pair_id = (z.tobytes(), x.tobytes())
        if pair_id in used:
            continue
        used.add(pair_id)
        schedulelist.append((z.copy(), x.copy()))
    return schedulelist

nrounds = [1, 2, 4]
ers = [0.0001, 0.001, 0.01]

def main():
    os.makedirs(os.path.join(os.path.dirname(__file__), outdir), exist_ok=True)

    # Default (unrandomized) Steane schedules
    cb = c713scheduleable(0.01, 1, zschedule=defaultzschedule, xschedule=defaultxschedule)
    with open(os.path.join(os.path.dirname(__file__), outdir, 'default.txt'), 'w') as f:
        f.write(f'{defaultzschedule}\n{defaultxschedule}\n')
        f.write(str(cb.get_circuit().diagram()))
    for k in range(len(nrounds)):
        for l in range(len(ers)):
            cb = c713scheduleable(ers[l], nrounds[k], zschedule=defaultzschedule, xschedule=defaultxschedule)
            sample = cb.sample(nshots=nshots)
            with open(os.path.join(os.path.dirname(__file__), outdir, f'default_n{k}_er{l}.bin'), 'wb') as f:
                for row in sample:
                    comp = np.uint64(int(row[0], 2))
                    f.write(np.uint64(comp).tobytes())
                    f.write(np.uint32(row[1]).tobytes())

    # Randomized coupling schedules
    for j in tqdm(range(numrandomizations), desc='Randomized Steane schedules'):
        randzschedule, nz = randomize_schedule(defaultzschedule, 1000)
        randxschedule, nx = randomize_schedule(defaultxschedule, 1000)
        cb = c713scheduleable(0.01, 1, zschedule=randzschedule, xschedule=randxschedule)
        with open(os.path.join(os.path.dirname(__file__), outdir, f'rand{j}.txt'), 'w') as f:
            f.write(f'{nz} {nx}\n')
            f.write(f'{randzschedule}\n{randxschedule}\n')
            f.write(str(cb.get_circuit().diagram()))
        for k in range(len(nrounds)):
            for l in range(len(ers)):
                cb = c713scheduleable(ers[l], nrounds[k], zschedule=randzschedule, xschedule=randxschedule)
                sample = cb.sample(nshots=nshots)
                with open(os.path.join(os.path.dirname(__file__), outdir, f'rand{j}_n{k}_er{l}.bin'), 'wb') as f:
                    for row in sample:
                        comp = np.uint64(int(row[0], 2))
                        f.write(np.uint64(comp).tobytes())
                        f.write(np.uint32(row[1]).tobytes())

if __name__ == "__main__":
    main()
