import numpy as np
import pickle
import CircuitBuilder
from tqdm import tqdm
import os

'''
PARAMS
'''

nrounds = [1, 2, 4]
ers = [0.0001, 0.001, 0.01]
nshots = 1000000
# Per-qubit idle-error weights, one landscape per schedule in schedulelist.pkl.
# Entry i is the multiplier for qubit i. Mean 1 keeps the average at `er`.
# CNOT noise is the larger of the two qubits' idle probabilities.
landscapepath = os.path.join(os.path.dirname(__file__), 'hotspoterrors2800.pkl')
outpath = 'batcherouthotspot2800.csv'


schedulelist = pickle.load(open(os.path.join(os.path.dirname(__file__), 'schedulelist2.pkl'), 'rb'))
logops = pickle.load(open(os.path.join(os.path.dirname(__file__), 'logops.pkl'), 'rb'))
landscapes = pickle.load(open(landscapepath, 'rb'))
if len(landscapes) != len(schedulelist):
    raise ValueError(f"{landscapepath} has {len(landscapes)} landscapes for {len(schedulelist)} schedules")


defaultzschedule = np.asarray([
    [1, 2, 3, 4, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 2, 3, 4],
    [2, 3, 0, 0, 4, 1, 0, 0],
    [3, 0, 4, 0, 2, 0, 1, 0]
])
defaultxschedule = np.asarray([
    [1, 2, 3, 4, 5, 6, 7, 8]
])
def windowFiller(schedule, nwindow):
    thiswindow = []
    for row in schedule:
        if nwindow in row:
            thiswindow.append(np.where(row == nwindow)[0][0])
        else:
            thiswindow.append(None)
    return thiswindow

def c832scheduleable(er = 0, nrounds = 1, zschedule = defaultzschedule, xschedule = defaultxschedule, pidle = None):
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

def check_detector_fired(det):
    return np.sum(det) > 0

def measure_readout(readout, logical_operator):
    parity = 0
    for i in range(len(readout)):
        if readout[i] == '1' and logical_operator[i]:
            parity += 1
    return parity % 2

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


                