import numpy as np
import pickle
import os
from tqdm import tqdm

from c713 import c713scheduleable, logical_operators_713, make_schedulelist_713

'''
PARAMS
'''

nrounds = [1, 2, 4]
ers = [0.0001, 0.001, 0.01]
outpath = os.path.join(os.path.dirname(__file__), 'batcherout713.csv')
nshots = 1000000
nschedules = 100

here = os.path.dirname(__file__)
schedulepath = os.path.join(here, 'schedulelist713.pkl')
logoppath = os.path.join(here, 'logops713.pkl')


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
        schedulelist = make_schedulelist_713(nschedules, seed=0)
        with open(schedulepath, 'wb') as f:
            pickle.dump(schedulelist, f)
    if not os.path.exists(logoppath):
        with open(logoppath, 'wb') as f:
            pickle.dump(logical_operators_713(), f)


def main():
    ensure_inputs()
    schedulelist = pickle.load(open(schedulepath, 'rb'))
    logops = pickle.load(open(logoppath, 'rb'))
    if os.path.exists(outpath):
        os.remove(outpath)
    for s in tqdm(range(len(schedulelist))):
        zschedule, xschedule = schedulelist[s]
        for n in range(len(nrounds)):
            for e in range(len(ers)):
                cb = c713scheduleable(ers[e], nrounds[n], zschedule=zschedule, xschedule=xschedule)
                sample = cb.get_circuit().compile_sampler().sample(nshots)
                syndromelength = len(zschedule) + len(xschedule)
                totalshots, totalerrors, survived, survivederrors = sim_analysis(sample, syndromelength, nrounds[n], logops)
                with open(outpath, 'a') as f:
                    totalerrors_str = ','.join(str(x) for x in totalerrors)
                    survivederrors_str = ','.join(str(x) for x in survivederrors)
                    f.write(f'{s},{nrounds[n]},{ers[e]},{totalshots},{totalerrors_str},{survived},{survivederrors_str}\n')


if __name__ == '__main__':
    main()
