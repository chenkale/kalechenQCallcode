import numpy as np
import pickle
import CircuitBuilder
from tqdm import tqdm
import os

'''
PARAMS
'''

here = os.path.dirname(__file__)
nrounds = [1, 2, 4]
ers = [0.0001, 0.001, 0.01]
nshots = 1000000
nschedules = 100
# Per-qubit idle-error weights, one landscape per schedule.
# Entry i is the multiplier for qubit i. Mean 1 keeps the average at `er`.
# CNOT noise is the larger of the two qubits' idle probabilities.
# Uniform landscape: all ones (same as pidle=None, but kept explicit for the batch API).
schedulepath = os.path.join(here, 'schedulelistdefault100.pkl')
landscapepath = os.path.join(here, 'uniformerrors100.pkl')
logoppath = os.path.join(here, 'logops.pkl')
outpath = os.path.join(here, 'batcheroutuniform100.csv')


defaultzschedule = np.asarray([
    [1, 2, 3, 4, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 2, 3, 4],
    [2, 3, 0, 0, 4, 1, 0, 0],
    [3, 0, 4, 0, 2, 0, 1, 0]
])
defaultxschedule = np.asarray([
    [1, 2, 3, 4, 5, 6, 7, 8]
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
    """Return a valid randomized copy of `schedule`, or None if exhausted."""
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
    return randomize_schedule(schedule, max_attempts - 1)


def make_schedulelist_default(n=100, seed=0):
    """n distinct randomizations of the default [[8,3,2]] Z and X schedules."""
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


def make_uniform_landscapes(n, nqubits):
    """n copies of the all-ones landscape (mean 1)."""
    return [np.ones(nqubits, dtype=float) for _ in range(n)]


def windowFiller(schedule, nwindow):
    thiswindow = []
    for row in schedule:
        if nwindow in row:
            thiswindow.append(np.where(row == nwindow)[0][0])
        else:
            thiswindow.append(None)
    return thiswindow


def c832scheduleable(er=0, nrounds=1, zschedule=defaultzschedule, xschedule=defaultxschedule, pidle=None):
    zschedule, xschedule = np.asarray(zschedule), np.asarray(xschedule)
    # A landscape is a per-qubit weight. Scale it by the base error rate.
    # None keeps every channel, including idle and CNOT, at the scalar er.
    if pidle is None:
        pidle_error = er
    else:
        pidle_error = np.asarray(pidle, dtype=float) * er
    cb = CircuitBuilder.CircuitBuilder(er, er, er, er, er, pidle_error)
    cb.al('R 0 1 2 3 4 5 6 7')
    cb.al('H 0 1 2 3 4 5 6 7')
    nancillas = len(zschedule) + len(xschedule)
    ancillastring = 'R ' + ' '.join([f'{i}' for i in range(8, 8 + nancillas)])
    cb.al(ancillastring)
    cb.set_ancilla_qubits(
        qubitslistz=[8 + i for i in range(len(zschedule))],
        qubitslistx=[8 + len(zschedule) + i for i in range(len(xschedule))],
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
    cb.al('H 0 1 2 3 4 5 6 7')
    cb.al(f'MR({0}) 0 1 2 3 4 5 6 7')
    return cb


def sim_analysis(shots, syndromelength, nrounds, logops):
    nphysicalqubits = 8
    det_rounds = nrounds + 1
    if shots.shape[1] != syndromelength * (det_rounds + 1) + nphysicalqubits:
        raise ValueError(f"Measurement sequence length must be equal to syndromelength * nrounds + nphysicalqubits")
    syndromes = shots[:, :-nphysicalqubits].reshape(shots.shape[0], det_rounds + 1, syndromelength)
    finalmeasurement = shots[:, -nphysicalqubits:]
    detector_fired = np.logical_xor(syndromes[:, :-1, :], syndromes[:, 1:, :]).any(axis=(1, 2))
    logop_mat = np.stack([np.asarray(logop, dtype=bool) for logop in logops])
    error_mask = ((finalmeasurement.astype(np.uint8) @ logop_mat.astype(np.uint8).T) & 1).astype(bool)
    survived_mask = ~detector_fired
    totalshots = int(shots.shape[0])
    totalerrors = error_mask.sum(axis=0).tolist()
    survived = int(survived_mask.sum())
    survivederrors = (error_mask & survived_mask[:, None]).sum(axis=0).tolist()
    return totalshots, totalerrors, survived, survivederrors


def ensure_inputs():
    if not os.path.exists(schedulepath):
        schedulelist = make_schedulelist_default(nschedules, seed=0)
        with open(schedulepath, 'wb') as f:
            pickle.dump(schedulelist, f)
    else:
        schedulelist = pickle.load(open(schedulepath, 'rb'))
    nqubits = 8 + len(defaultzschedule) + len(defaultxschedule)
    if not os.path.exists(landscapepath):
        landscapes = make_uniform_landscapes(len(schedulelist), nqubits)
        with open(landscapepath, 'wb') as f:
            pickle.dump(landscapes, f)


def main():
    ensure_inputs()
    schedulelist = pickle.load(open(schedulepath, 'rb'))
    landscapes = pickle.load(open(landscapepath, 'rb'))
    logops = pickle.load(open(logoppath, 'rb'))
    if len(landscapes) != len(schedulelist):
        raise ValueError(f"{landscapepath} has {len(landscapes)} landscapes for {len(schedulelist)} schedules")
    if os.path.exists(outpath):
        os.remove(outpath)
    for s in tqdm(range(len(schedulelist))):
        zschedule, xschedule = schedulelist[s]
        landscape = np.asarray(landscapes[s], dtype=float)
        nqubits = 8 + len(zschedule) + len(xschedule)
        if landscape.ndim != 1 or len(landscape) < nqubits:
            raise ValueError(f"Landscape {s} has shape {landscape.shape}, circuit uses qubits 0..{nqubits - 1}")
        for n in range(len(nrounds)):
            for e in range(len(ers)):
                cb = c832scheduleable(ers[e], nrounds[n], zschedule=zschedule, xschedule=xschedule, pidle=landscape)
                sample = cb.get_circuit().compile_sampler().sample(nshots)
                syndromelength = len(zschedule) + len(xschedule)
                totalshots, totalerrors, survived, survivederrors = sim_analysis(sample, syndromelength, nrounds[n], logops)
                with open(outpath, 'a') as f:
                    totalerrors_str = ','.join(str(x) for x in totalerrors)
                    survivederrors_str = ','.join(str(x) for x in survivederrors)
                    f.write(f'{s},{nrounds[n]},{ers[e]},{totalshots},{totalerrors_str},{survived},{survivederrors_str}\n')


if __name__ == '__main__':
    main()
