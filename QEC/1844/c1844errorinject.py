import os
import numpy as np
from tqdm import tqdm

import CircuitBuilder

def checkvalidschedule(schedule):
    schedule = np.asarray(schedule)
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
            val = schedule[row_idx, col_idx]
            if val != 0:
                if val in seen:
                    return False
                seen.add(val)
    return True

def randomize_schedule(schedule, max_attempts=10000):
    schedule = np.asarray(schedule)
    for attempts_left in range(max_attempts, 0, -1):
        schedule_copy = np.copy(schedule)
        for i in range(len(schedule_copy)):
            nonzero_indices = [j for j, v in enumerate(schedule_copy[i]) if v != 0]
            nonzeros = list(range(1, int(max(schedule_copy[i])) + 1)) if max(schedule_copy[i]) != 1 else list(range(1, 7))
            np.random.shuffle(nonzeros)
            for idx in nonzero_indices:
                schedule_copy[i][idx] = nonzeros.pop()
        if checkvalidschedule(schedule_copy):
            return schedule_copy, attempts_left
    return None, 0

def windowFiller(schedule, nwindow):
    thiswindow = []
    for row in schedule:
        if nwindow in row:
            thiswindow.append(int(np.where(row == nwindow)[0][0]))
        else:
            thiswindow.append(None)
    return thiswindow

defaultzschedule = np.load(os.path.join(os.path.dirname(__file__), 'hz.npy'))[0:7]
defaultxschedule = np.load(os.path.join(os.path.dirname(__file__), 'hx.npy'))[0:7]
defaulterrorinjectschedule = np.zeros((14, 18), dtype = np.int8)
errortypes = ['I', 'X', 'Y', 'Z']
def c1844scheduleableinjectable(
    zschedule=defaultzschedule,
    xschedule=defaultxschedule,
    errorinjectschedule = defaulterrorinjectschedule
):
    cb = CircuitBuilder.CircuitBuilder(0, 0, 0, 0, 0, 0)
    data_str = ' '.join(str(i) for i in range(18))
    cb.al(f'R {data_str}')

    nancillas = len(zschedule) + len(xschedule)
    ancilla_ids = list(range(18, 18 + nancillas))
    cb.al('R ' + ' '.join(str(i) for i in ancilla_ids))
    cb.set_ancilla_qubits(
        qubitslistz=[18 + i for i in range(len(zschedule))],
        qubitslistx=[18 + len(zschedule) + i for i in range(len(xschedule))],
    )

    # Warmup (no time_tick noise)
    for windowz in range(1, int(np.max(zschedule)) + 1):
        cb.couple_z_check(windowFiller(zschedule, windowz), False)
    cb.al('MR(0) ' + ' '.join(str(i) for i in cb.ancilla_qubitsz))
    cb.al('H ' + ' '.join(str(i) for i in cb.ancilla_qubitsx))
    for windowx in range(1, int(np.max(xschedule)) + 1):
        cb.couple_x_check(windowFiller(xschedule, windowx), False)
    cb.al('H ' + ' '.join(str(i) for i in cb.ancilla_qubitsx))
    cb.al('MR(0) ' + ' '.join(str(i) for i in cb.ancilla_qubitsx))
    cb.al('R ' + ' '.join(str(i) for i in cb.ancilla_qubitsz))

    # Noisy syndrome round with error injection
    index = 0
    for windowz in range(1, int(np.max(zschedule)) + 1):
        for j in range(18):
            if errorinjectschedule[index][j]:
                cb.al(f'{errortypes[errorinjectschedule[index][j]]} {j}')
        index += 1
        cb.couple_z_check(windowFiller(zschedule, windowz), False)
    cb.al('MR(0) ' + ' '.join(str(i) for i in cb.ancilla_qubitsz))
    cb.al('R ' + ' '.join(str(i) for i in cb.ancilla_qubitsx))
    cb.al('H ' + ' '.join(str(i) for i in cb.ancilla_qubitsx))
    for windowx in range(1, int(np.max(xschedule)) + 1):
        for j in range(18):
            if errorinjectschedule[index][j]:
                cb.al(f'{errortypes[errorinjectschedule[index][j]]} {j}')
        index += 1
        cb.couple_x_check(windowFiller(xschedule, windowx), False)
    cb.add_detectors([[-1 - i, -1 - i - nancillas] for i in range(len(cb.ancilla_qubitsz))])
    cb.al('H ' + ' '.join(str(i) for i in cb.ancilla_qubitsx))
    cb.al('MR(0) ' + ' '.join(str(i) for i in cb.ancilla_qubitsx))
    cb.add_detectors([[-1 - i, -1 - i - nancillas] for i in range(len(cb.ancilla_qubitsx))])
    cb.al('R ' + ' '.join(str(i) for i in cb.ancilla_qubitsz))

    # Cooldown (noiseless); detectors vs last syndrome round (warmup if nrounds=0)
    for windowz in range(1, int(np.max(zschedule)) + 1):
        cb.couple_z_check(windowFiller(zschedule, windowz), False)
    cb.al('MR(0) ' + ' '.join(str(i) for i in cb.ancilla_qubitsz))
    cb.add_detectors([[-1 - i, -1 - i - nancillas] for i in range(len(cb.ancilla_qubitsz))])
    cb.al('H ' + ' '.join(str(i) for i in cb.ancilla_qubitsx))
    for windowx in range(1, int(np.max(xschedule)) + 1):
        cb.couple_x_check(windowFiller(xschedule, windowx), False)
    cb.al('H ' + ' '.join(str(i) for i in cb.ancilla_qubitsx))
    cb.al('MR(0) ' + ' '.join(str(i) for i in cb.ancilla_qubitsx))
    cb.add_detectors([[-1 - i, -1 - i - nancillas] for i in range(len(cb.ancilla_qubitsx))])
    cb.al('R ' + ' '.join(str(i) for i in cb.ancilla_qubitsz))

    # Final data measurement + logical observables for DEM / BP+OSD
    cb.al(f'MR(0) {data_str}')
    #logops = defaultlx if basis == 'X' else defaultlz
    #for obs_id, logop in enumerate(logops):
    #    recs = [-(18 - q) for q in range(18) if logop[q]]
    #    if recs:
    #        cb.add_observable(recs, obs_id)
    return cb

def random_errorinjectschedule(er = 0.01, xschedule = defaultxschedule, zschedule = defaultzschedule):
    errorinjectschedule = np.zeros((len(zschedule[0][zschedule[0] != 0]) + len(xschedule[0][xschedule[0] != 0]), 18), dtype = np.int8)
    generatederrors = []
    for i in range(len(zschedule[0][zschedule[0] != 0])):
        for j in range(18):
            if np.random.random() < er:
                errorinjectschedule[i][j] = np.random.randint(1, 4)
                generatederrors.append((i, j, errortypes[errorinjectschedule[i][j]]))
    for i in range(len(xschedule[0][xschedule[0] != 0])):
        for j in range(18):
            if np.random.random() < er:
                errorinjectschedule[i + len(zschedule[0][zschedule[0] != 0])][j] = np.random.randint(1, 4)
                generatederrors.append((i + len(zschedule[0][zschedule[0] != 0]), j, errortypes[errorinjectschedule[i + len(zschedule[0][zschedule[0] != 0])][j]]))
    return errorinjectschedule, generatederrors

def detector_builder(mseq, syndromelength=18, nphysicalqubits=18, nrounds=1, mode='binary'):
    expected = syndromelength * (nrounds + 2) + nphysicalqubits
    if len(mseq) != expected:
        raise ValueError(
            f'Measurement sequence length {len(mseq)} != syndromelength*(nrounds+2)+n '
            f'= {expected}'
        )
    syndromes, finalmeasurement = mseq[:-nphysicalqubits], mseq[-nphysicalqubits:]
    if mode == 'string':
        det = []
        for i in range(nrounds + 1):
            a = syndromes[i * syndromelength:(i + 1) * syndromelength]
            b = syndromes[(i + 1) * syndromelength:(i + 2) * syndromelength]
            det.append(''.join('0' if x == y else '1' for x, y in zip(a, b)))
    elif mode == 'binary':
        det = np.zeros((nrounds + 1) * syndromelength, dtype=bool)
        for i in range(nrounds + 1):
            a = syndromes[i * syndromelength:(i + 1) * syndromelength]
            b = syndromes[(i + 1) * syndromelength:(i + 2) * syndromelength]
            det[i * syndromelength:(i + 1) * syndromelength] = np.array(
                [x != y for x, y in zip(a, b)]
            )
    else:
        raise ValueError(f'Invalid mode: {mode}')
    return det, finalmeasurement


if __name__ == '__main__':
    randxschedule, max_attempts = randomize_schedule(defaultxschedule)
    randzschedule, max_attempts = randomize_schedule(defaultzschedule)
    randerrorinjectschedule, generatederrors = random_errorinjectschedule(er = 0.01)
    #cb = c1844scheduleableinjectable(xschedule=randxschedule, zschedule=randzschedule, errorinjectschedule=randerrorinjectschedule)
    cb = c1844scheduleableinjectable(xschedule=randxschedule, zschedule=randzschedule)
    with open (os.path.join(os.path.dirname(__file__), 'look.txt'), 'w') as f:
        f.write(str(cb.get_circuit().diagram()))
    print(randerrorinjectschedule)
    print(generatederrors)
    sample = cb.sample(1)
    print(sample, len(sample[0][0]))

    det, finalmeasurement = detector_builder(sample[0][0], syndromelength=14, nphysicalqubits=18, nrounds=1, mode='binary')
    print(det)
    print(finalmeasurement)
