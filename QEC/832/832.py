import CircuitBuilder
import numpy as np
import pickle
import os
from tqdm import tqdm

numruns = 10
numrandomizations = 20
nshots = 1000000
outdir = 'sim'

def c832(er = 0, nrounds = 1):
    cb = CircuitBuilder.CircuitBuilder(er, er, er, er, er, er)
    # State preparation
    cb.al('R 0 1 2 3 4 5 6 7') # Physical qubits
    cb.al('R 8 9 10 11 12') # Ancilla qubits
    cb.set_ancilla_qubits(qubitslistz = [8, 9, 10, 11], qubitslistx = [12])
    cb.al('H 12')
    # Inject error
    #cb.al('X 2')
    # Warmup round
    cb.time_tick()
    cb.couple_z_check([0, 4, 5, 6], True)
    cb.time_tick()
    cb.couple_z_check([1, 5, 0, 4], True)
    cb.time_tick()
    cb.couple_z_check([2, 6, 1, 0], True)
    cb.time_tick()
    cb.couple_z_check([3, 7, 4, 2], True)
    cb.time_tick()
    cb.couple_x_check([0], True)
    cb.time_tick()
    cb.couple_x_check([1], True)
    cb.time_tick()
    cb.couple_x_check([2], True)
    cb.time_tick()
    cb.couple_x_check([3], True)
    cb.time_tick()
    cb.couple_x_check([4], True)
    cb.time_tick()
    cb.couple_x_check([5], True)
    cb.time_tick()
    cb.couple_x_check([6], True)
    cb.time_tick()
    cb.couple_x_check([7], True)
    cb.al('MR(0) 8 9 10 11')
    cb.al('H 12')
    cb.al('MR(0) 12')
    cb.al('H 12')
    # Noise rounds
    cb.begin_loop(nrounds)
    cb.time_tick()
    cb.couple_z_check([0, 4, 5, 6], True)
    cb.time_tick()
    cb.couple_z_check([1, 5, 0, 4], True)
    cb.time_tick()
    cb.couple_z_check([2, 6, 1, 0], True)
    cb.time_tick()
    cb.couple_z_check([3, 7, 4, 2], True)
    cb.time_tick()
    cb.couple_x_check([0], True)
    cb.time_tick()
    cb.couple_x_check([1], True)
    cb.time_tick()
    cb.couple_x_check([2], True)
    cb.time_tick()
    cb.couple_x_check([3], True)
    cb.time_tick()
    cb.couple_x_check([4], True)
    cb.time_tick()
    cb.couple_x_check([5], True)
    cb.time_tick()
    cb.couple_x_check([6], True)
    cb.time_tick()
    cb.couple_x_check([7], True)
    cb.al('MR(0) 8 9 10 11')
    cb.add_detectors([[-1, -6], [-2, -7], [-3, -8], [-4, -9]])
    cb.al('H 12')
    cb.al('MR(0) 12')
    cb.al('H 12')
    cb.add_detectors([[-1, -6]])
    cb.end_loop()
    # Final measurement of all physical qubits
    cb.al(f'MR({0}) 0 1 2 3 4 5 6 7')

    return cb

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
    """
    Returns a randomized copy of `schedule` that is valid, or None if failed after max_attempts.
    Does NOT mutate the input `schedule` in place.
    """
    if max_attempts == 0:
        return None
    # Make a deep copy to avoid inplace mutation
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

def c832scheduleable(er = 0, nrounds = 1, zschedule = defaultzschedule, xschedule = defaultxschedule):
    zschedule, xschedule = np.asarray(zschedule), np.asarray(xschedule)
    cb = CircuitBuilder.CircuitBuilder(er, er, er, er, er, er)
    cb.al('R 0 1 2 3 4 5 6 7')
    cb.al('H 0 1 2 3 4 5 6 7')
    #cb.al('X 0 1') # Flip some physical qubits to start, still must be in codespace
    nancillas = len(zschedule) + len(xschedule)
    ancillastring = 'R ' + ' '.join([f'{i}' for i in range(8, 8 + nancillas)])
    cb.al(ancillastring)
    cb.set_ancilla_qubits(qubitslistz = [8 + i for i in range(len(zschedule))], qubitslistx = [8 + len(zschedule) + i for i in range(len(xschedule))])
    #print(cb.ancilla_qubitsz, cb.ancilla_qubitsx)
    # Warmup round
    for windowz in range(1, np.max(zschedule) + 1):
        #cb.time_tick()
        #print(windowFiller(zschedule, windowz))
        cb.couple_z_check(windowFiller(zschedule, windowz), False)
    cb.al('MR(0) ' + ' '.join([f'{i}' for i in cb.ancilla_qubitsz]))
    cb.al('H ' + ' '.join([f'{i}' for i in cb.ancilla_qubitsx]))
    for windowx in range(1, np.max(xschedule) + 1):
        #cb.time_tick()
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
    # Final measurement of all physical qubits
    cb.al('H 0 1 2 3 4 5 6 7')
    cb.al(f'MR({0}) 0 1 2 3 4 5 6 7')
    return cb
    

def detector_builder(mseq, syndromelength = 5, nphysicalqubits = 8, nrounds = 1, mode = 'binary'):
    if len(mseq) != syndromelength * (nrounds + 1) + nphysicalqubits:
        raise ValueError(f"Measurement sequence length must be equal to syndromelength * nrounds + nphysicalqubits")
    syndromes, finalmeasurement = mseq[:-nphysicalqubits], mseq[-nphysicalqubits:]
    if mode == 'string':
        det = []
        for i in range(nrounds):
            det.append(''.join([str(0 if x == y else 1) for x, y in zip(syndromes[i*syndromelength:(i+1)*syndromelength], syndromes[(i+1)*syndromelength:(i+2)*syndromelength])]))
    elif mode == 'binary':
        det = np.zeros(nrounds * syndromelength, dtype = np.bool)
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

singlepromotionschedules = pickle.load(open(os.path.join(os.path.dirname(__file__), 'singlepromotionschedules.pkl'), 'rb'))
doublepromotionschedules = pickle.load(open(os.path.join(os.path.dirname(__file__), 'doublepromotionschedules.pkl'), 'rb'))
nrounds = [1, 2, 4]
ers = [0.0001, 0.001, 0.01]

dopromotionsimulations = False
if dopromotionsimulations:
    for i in range(numruns):
        # Single promotion schedules experiment setup
        rand = np.random.randint(0, len(singlepromotionschedules))
        zschedule = defaultzschedule
        xschedule = singlepromotionschedules[rand]
        for j in tqdm(range(numrandomizations), desc = f'Running single promotion schedule {i}'):
            randzschedule, nz = randomize_schedule(zschedule, 1000)
            randxschedule, nx = randomize_schedule(xschedule, 1000)
            cb = c832scheduleable(0.01, 1, zschedule=randzschedule, xschedule=randxschedule)
            with open(os.path.join(os.path.dirname(__file__), outdir, f'run{i}_single_rand{j}.txt'), 'w') as f:
                f.write(f'{rand} {nz} {nx}\n')
                f.write(f'{randzschedule}\n{randxschedule}\n')
                f.write(str(cb.get_circuit().diagram()))
            for k in range(len(nrounds)):
                for l in range(len(ers)):
                    cb = c832scheduleable(ers[l], nrounds[k], zschedule=randzschedule, xschedule=randxschedule)
                    sample = cb.sample(nshots = nshots)
                    with open(os.path.join(os.path.dirname(__file__), outdir, f'run{i}_single_rand{j}_n{k}_er{l}.bin'), 'wb') as f:
                        for row in sample:
                            comp = np.uint64(int(row[0], 2))
                            f.write(np.uint64(comp).tobytes())
                            f.write(np.uint32(row[1]).tobytes())

        # Double promotion schedules experiment setup
        rand = np.random.randint(0, len(doublepromotionschedules))
        zschedule = defaultzschedule
        xschedule = doublepromotionschedules[rand]
        for j in tqdm(range(numrandomizations), desc = f'Running double promotion schedule {i}'):
            randzschedule, nz = randomize_schedule(zschedule, 1000)
            randxschedule, nx = randomize_schedule(xschedule, 1000)
            cb = c832scheduleable(0.01, 1, zschedule=randzschedule, xschedule=randxschedule)
            with open(os.path.join(os.path.dirname(__file__), outdir, f'run{i}_double_rand{j}.txt'), 'w') as f:
                f.write(f'{rand} {nz} {nx}\n')
                f.write(f'{randzschedule}\n{randxschedule}\n')
                f.write(str(cb.get_circuit().diagram()))
            for k in range(len(nrounds)):
                for l in range(len(ers)):
                    cb = c832scheduleable(ers[l], nrounds[k], zschedule=randzschedule, xschedule=randxschedule)
                    sample = cb.sample(nshots = nshots)
                    with open(os.path.join(os.path.dirname(__file__), outdir, f'run{i}_double_rand{j}_n{k}_er{l}.bin'), 'wb') as f:
                        for row in sample:
                            comp = np.uint64(int(row[0], 2))
                            f.write(np.uint64(comp).tobytes())
                            f.write(np.uint32(row[1]).tobytes())

dobasesimulation = True
if dobasesimulation:            
    # Simulate base code, no promotion
    cb = c832scheduleable(0.01, 1, zschedule=defaultzschedule, xschedule=defaultxschedule)
    with open(os.path.join(os.path.dirname(__file__), outdir, f'nopromotion.txt'), 'w') as f:
        f.write(f'{defaultzschedule}\n{defaultxschedule}\n')
        f.write(str(cb.get_circuit().diagram()))
    for k in range(len(nrounds)):
        for l in range(len(ers)):
            cb = c832scheduleable(ers[l], nrounds[k], zschedule=defaultzschedule, xschedule=defaultxschedule)
            sample = cb.sample(nshots = nshots)
            with open(os.path.join(os.path.dirname(__file__), outdir, f'nopromotion_n{k}_er{l}.bin'), 'wb') as f:
                for row in sample:
                    comp = np.uint64(int(row[0], 2))
                    f.write(np.uint64(comp).tobytes())
                    f.write(np.uint32(row[1]).tobytes())

    for j in tqdm(range(numrandomizations)):
        randzschedule, nz = randomize_schedule(defaultzschedule, 1000)
        #randxschedule, nx = randomize_schedule(defaultxschedule, 1000)
        randxschedule, nx = defaultxschedule, 0
        cb = c832scheduleable(0.01, 1, zschedule=randzschedule, xschedule=randxschedule)
        with open(os.path.join(os.path.dirname(__file__), outdir, f'nopromotion_rand{j}.txt'), 'w') as f:
            f.write(f'{nz} {nx}\n')
            f.write(f'{randzschedule}\n{randxschedule}\n')
            f.write(str(cb.get_circuit().diagram()))
        for k in range(len(nrounds)):
            for l in range(len(ers)):
                cb = c832scheduleable(ers[l], nrounds[k], zschedule=randzschedule, xschedule=randxschedule)
                sample = cb.sample(nshots = nshots)
                with open(os.path.join(os.path.dirname(__file__), outdir, f'nopromotion_rand{j}_n{k}_er{l}.bin'), 'wb') as f:
                    for row in sample:
                        comp = np.uint64(int(row[0], 2))
                        f.write(np.uint64(comp).tobytes())
                        f.write(np.uint32(row[1]).tobytes())
