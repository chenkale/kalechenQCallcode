"""
Circuit-level Stim simulation for the [[18, 4, 4]] bivariate bicycle code.

Mirrors the [[8, 3, 2]] pipeline in QEC/832/c832.py:
  CircuitBuilder + time-window schedules + warmup / noisy loop / cooldown
  + detector XOR postselection and BP+OSD decoding on the Stim DEM.

CNOT order follows the BB paper's Tanner directions (sX / sZ).
"""
import os
import pickle

import numpy as np
from tqdm import tqdm

import CircuitBuilder

try:
    from ldpc.bposd_decoder import BpOsdDecoder
    from ldpc.ckt_noise.dem_matrices import detector_error_model_to_check_matrices
except ImportError:  # pragma: no cover - optional until ldpc is installed
    BpOsdDecoder = None
    detector_error_model_to_check_matrices = None

N_DATA = 18
N2 = 9

# BB parameters for [[18, 4, 4]]
ELL, M = 3, 3
A1_POW, A2_POW, A3_POW = 1, 0, 2
B1_POW, B2_POW, B3_POW = 1, 0, 2

# Parallel Tanner schedule from the Exp_BivariateBicycleCode reference
SX_DIRS = ['idle', 'idle', 1, 4, 3, 5, 0, 2]
SZ_DIRS = ['idle', 3, 5, 0, 1, 2, 4, 'idle']


def _shift_matrices():
    I_ell = np.identity(ELL, dtype=int)
    I_m = np.identity(M, dtype=int)
    x, y = {}, {}
    for i in range(ELL):
        x[i] = np.kron(np.roll(I_ell, i, axis=1), I_m)
    for i in range(M):
        y[i] = np.kron(I_ell, np.roll(I_m, i, axis=1))
    A1 = x[A1_POW % ELL]
    A2 = y[A2_POW % M]
    A3 = y[A3_POW % M]
    B1 = y[B1_POW % M]
    B2 = x[B2_POW % ELL]
    B3 = x[B3_POW % ELL]
    A = (A1 + A2 + A3) % 2
    B = (B1 + B2 + B3) % 2
    hx = np.hstack((A, B))
    hz = np.hstack((B.T, A.T))
    return hx, hz, A1, A2, A3, B1, B2, B3


def gf2_rank(A):
    mat = (np.asarray(A) % 2).astype(int).copy()
    rows, cols = mat.shape
    rank = 0
    row = 0
    for col in range(cols):
        pivot = None
        for i in range(row, rows):
            if mat[i, col]:
                pivot = i
                break
        if pivot is None:
            continue
        mat[[row, pivot]] = mat[[pivot, row]]
        for i in range(rows):
            if i != row and mat[i, col]:
                mat[i] = (mat[i] + mat[row]) % 2
        rank += 1
        row += 1
        if row == rows:
            break
    return rank


def gf2_nullspace(A):
    mat = (np.asarray(A) % 2).astype(int).copy()
    rows, cols = mat.shape
    pivots = {}
    row = 0
    for col in range(cols):
        pivot = None
        for i in range(row, rows):
            if mat[i, col]:
                pivot = i
                break
        if pivot is None:
            continue
        mat[[row, pivot]] = mat[[pivot, row]]
        for i in range(rows):
            if i != row and mat[i, col]:
                mat[i] = (mat[i] + mat[row]) % 2
        pivots[col] = row
        row += 1
        if row == rows:
            break
    free = [c for c in range(cols) if c not in pivots]
    basis = []
    for f in free:
        v = np.zeros(cols, dtype=int)
        v[f] = 1
        for c, r in pivots.items():
            v[c] = mat[r, f]
        basis.append(v)
    return np.asarray(basis, dtype=int) if basis else np.zeros((0, cols), dtype=int)


def gf2_inverse(M):
    """Inverse of a square GF(2) matrix, or raise if singular."""
    M = (np.asarray(M) % 2).astype(int).copy()
    n = M.shape[0]
    aug = np.hstack([M, np.eye(n, dtype=int)])
    row = 0
    for col in range(n):
        pivot = None
        for i in range(row, n):
            if aug[i, col]:
                pivot = i
                break
        if pivot is None:
            raise np.linalg.LinAlgError('singular GF(2) matrix')
        aug[[row, pivot]] = aug[[pivot, row]]
        for i in range(n):
            if i != row and aug[i, col]:
                aug[i] = (aug[i] + aug[row]) % 2
        row += 1
    return aug[:, n:]


def css_logicals(hx, hz):
    """Return dual-paired logical bases lx, lz with (lx @ lz.T) % 2 = I."""
    ker_z = gf2_nullspace(hz)
    lx_rows = []
    cur = (hx % 2).astype(int).copy()
    for v in ker_z:
        trial = np.vstack([cur, v])
        if gf2_rank(trial) > gf2_rank(cur):
            lx_rows.append(v)
            cur = trial
    lx = np.asarray(lx_rows, dtype=int)

    ker_x = gf2_nullspace(hx)
    lz_rows = []
    cur = (hz % 2).astype(int).copy()
    for v in ker_x:
        trial = np.vstack([cur, v])
        if gf2_rank(trial) > gf2_rank(cur):
            lz_rows.append(v)
            cur = trial
    lz = np.asarray(lz_rows, dtype=int)

    if lx.shape[0] == 0 or lz.shape[0] == 0:
        return lx, lz
    # Dualize: find A so lx @ (A @ lz).T = I
    M = (lx @ lz.T) % 2
    A = gf2_inverse(M).T % 2
    lz = (A @ lz) % 2
    return lx, lz


hx, hz, A1, A2, A3, B1, B2, B3 = _shift_matrices()
lx, lz = css_logicals(hx, hz)


def build_direction_schedules(sX=SX_DIRS, sZ=SZ_DIRS):
    """
    Convert BB Tanner directions into 832-style schedules.
    Rows = checks, columns = data qubits, entry t > 0 = couple at time window t.
    """
    zschedule = np.zeros((N2, N_DATA), dtype=int)
    window = 0
    for d in sZ:
        if d == 'idle':
            continue
        window += 1
        for i in range(N2):
            if d <= 2:
                mats = [B1, B2, B3]
                q = int(np.nonzero(mats[d][:, i])[0][0])
                data = q
            else:
                mats = [A1, A2, A3]
                q = int(np.nonzero(mats[d - 3][:, i])[0][0])
                data = N2 + q
            zschedule[i, data] = window

    xschedule = np.zeros((N2, N_DATA), dtype=int)
    window = 0
    for d in sX:
        if d == 'idle':
            continue
        window += 1
        for i in range(N2):
            if d <= 2:
                mats = [A1, A2, A3]
                q = int(np.nonzero(mats[d][i, :])[0][0])
                data = q
            else:
                mats = [B1, B2, B3]
                q = int(np.nonzero(mats[d - 3][i, :])[0][0])
                data = N2 + q
            xschedule[i, data] = window
    return zschedule, xschedule


defaultzschedule, defaultxschedule = build_direction_schedules()


def check_to_schedule(H):
    """Naive sequential schedule from a parity-check matrix (may be invalid)."""
    sch = np.zeros_like(H, dtype=int)
    for i, row in enumerate(H):
        t = 1
        for j, v in enumerate(row):
            if v:
                sch[i, j] = t
                t += 1
    return sch


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


def relabel_schedule(schedule):
    """
    Randomly permute time labels globally. Always valid if `schedule` is valid,
    since bijective remapping preserves row/column uniqueness.
    """
    schedule = np.asarray(schedule).copy()
    times = sorted({int(v) for v in schedule.ravel() if v != 0})
    if not times:
        return schedule
    new_times = times.copy()
    np.random.shuffle(new_times)
    mapping = {old: new for old, new in zip(times, new_times)}
    out = np.zeros_like(schedule)
    for i in range(schedule.shape[0]):
        for j in range(schedule.shape[1]):
            v = int(schedule[i, j])
            out[i, j] = mapping[v] if v else 0
    return out


def randomize_schedule(schedule, max_attempts=1000):
    """
    Per-row shuffle of coupling times (832-style). Dense weight-6 BB schedules
    rarely stay conflict-free; prefer `relabel_schedule` when that matters.
    Returns (schedule, attempts_left) or (None, 0).
    """
    schedule = np.asarray(schedule)
    for attempts_left in range(max_attempts, 0, -1):
        schedule_copy = np.copy(schedule)
        for i in range(len(schedule_copy)):
            nonzero_indices = [j for j, v in enumerate(schedule_copy[i]) if v != 0]
            nonzeros = list(range(1, int(max(schedule_copy[i])) + 1)) if nonzero_indices else []
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


def c1844scheduleable(
    er=0,
    nrounds=1,
    zschedule=None,
    xschedule=None,
    pidle=None,
    basis='X',
):
    """
    Build a scheduleable [[18, 4, 4]] Stim circuit.

    Layout: data 0..17, Z-ancillas, then X-ancillas.
    Timeline: noiseless warmup -> noisy loop x nrounds -> noiseless cooldown
              -> final data measurement in `basis` ('X' or 'Z').
    """
    if zschedule is None:
        zschedule = defaultzschedule
    if xschedule is None:
        xschedule = defaultxschedule
    zschedule, xschedule = np.asarray(zschedule), np.asarray(xschedule)
    if not checkvalidschedule(zschedule) or not checkvalidschedule(xschedule):
        raise ValueError('Invalid Z or X schedule (row/column time conflicts)')

    if pidle is None:
        pidle_error = er
    else:
        pidle_error = np.asarray(pidle, dtype=float) * er

    cb = CircuitBuilder.CircuitBuilder(er, er, er, er, er, pidle_error)
    data_str = ' '.join(str(i) for i in range(N_DATA))
    cb.al(f'R {data_str}')
    if basis == 'X':
        cb.al(f'H {data_str}')

    nancillas = len(zschedule) + len(xschedule)
    ancilla_ids = list(range(N_DATA, N_DATA + nancillas))
    cb.al('R ' + ' '.join(str(i) for i in ancilla_ids))
    cb.set_ancilla_qubits(
        qubitslistz=[N_DATA + i for i in range(len(zschedule))],
        qubitslistx=[N_DATA + len(zschedule) + i for i in range(len(xschedule))],
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

    # Noisy syndrome rounds
    cb.begin_loop(nrounds)
    for windowz in range(1, int(np.max(zschedule)) + 1):
        cb.time_tick()
        cb.couple_z_check(windowFiller(zschedule, windowz), True)
    cb.al('MR(0) ' + ' '.join(str(i) for i in cb.ancilla_qubitsz))
    cb.al('R ' + ' '.join(str(i) for i in cb.ancilla_qubitsx))
    cb.al('H ' + ' '.join(str(i) for i in cb.ancilla_qubitsx))
    for windowx in range(1, int(np.max(xschedule)) + 1):
        cb.time_tick()
        cb.couple_x_check(windowFiller(xschedule, windowx), True)
    cb.add_detectors([[-1 - i, -1 - i - nancillas] for i in range(len(cb.ancilla_qubitsz))])
    cb.al('H ' + ' '.join(str(i) for i in cb.ancilla_qubitsx))
    cb.al('MR(0) ' + ' '.join(str(i) for i in cb.ancilla_qubitsx))
    cb.add_detectors([[-1 - i, -1 - i - nancillas] for i in range(len(cb.ancilla_qubitsx))])
    cb.al('R ' + ' '.join(str(i) for i in cb.ancilla_qubitsz))
    cb.end_loop()

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
    if basis == 'X':
        cb.al(f'H {data_str}')
    cb.al(f'MR(0) {data_str}')
    logops = lx if basis == 'X' else lz
    for obs_id, logop in enumerate(logops):
        recs = [-(N_DATA - q) for q in range(N_DATA) if logop[q]]
        if recs:
            cb.add_observable(recs, obs_id)
    return cb


def detector_builder(mseq, syndromelength=18, nphysicalqubits=N_DATA, nrounds=1, mode='binary'):
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


def sim_analysis(shots, syndromelength, nrounds, logops, nphysicalqubits=N_DATA):
    """
    Offline detector XOR (warmup..cooldown) + postselected logical parity.
    Returns totalshots, totalerrors[], survived, survivederrors[].
    """
    det_rounds = nrounds + 1
    expected = syndromelength * (det_rounds + 1) + nphysicalqubits
    if shots.shape[1] != expected:
        raise ValueError(
            f'shots width {shots.shape[1]} != expected {expected} '
            f'(syndromelength={syndromelength}, nrounds={nrounds})'
        )
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


# Default BP+OSD knobs (match Exp_BivariateBicycleCode [[18,4,4]] settings)
BPOSD_DEFAULTS = dict(
    max_iter=30,
    bp_method='ms',
    ms_scaling_factor=0.625,
    osd_method='osd_cs',
    osd_order=7,
)


def _require_ldpc():
    if BpOsdDecoder is None or detector_error_model_to_check_matrices is None:
        raise ImportError(
            'ldpc is required for BP+OSD decoding (pip install ldpc).'
        )


def dem_from_circuit(circuit):
    """Stim detector error model; keep hyperedges (BB noise is not graphlike)."""
    return circuit.detector_error_model(decompose_errors=False)


def make_bposd_decoder(dem, **kwargs):
    """
    Build a BP+OSD decoder from a Stim DEM.
    Returns (decoder, dem_matrices) where predicted observables are
    (dem_matrices.observables_matrix @ decoder.decode(syndrome)) % 2.
    """
    _require_ldpc()
    params = {**BPOSD_DEFAULTS, **kwargs}
    matrices = detector_error_model_to_check_matrices(
        dem, allow_undecomposed_hyperedges=True
    )
    if matrices.check_matrix.shape[1] == 0:
        return None, matrices
    decoder = BpOsdDecoder(
        matrices.check_matrix,
        error_channel=list(matrices.priors),
        max_iter=params['max_iter'],
        bp_method=params['bp_method'],
        ms_scaling_factor=params['ms_scaling_factor'],
        osd_method=params['osd_method'],
        osd_order=params['osd_order'],
    )
    return decoder, matrices


def decode_observables_bposd(det_shots, decoder, matrices):
    """Predict observable flips for each detector shot via BP+OSD."""
    nshots = det_shots.shape[0]
    nobs = matrices.observables_matrix.shape[0]
    if decoder is None or nobs == 0:
        return np.zeros((nshots, nobs), dtype=bool)
    preds = np.zeros((nshots, nobs), dtype=bool)
    obs_mat = matrices.observables_matrix
    for i in range(nshots):
        corr = decoder.decode(det_shots[i])
        preds[i] = np.asarray(obs_mat @ corr) % 2
    return preds.astype(bool)


def sim_analysis_bposd(circuit, nshots, seed=None, **bposd_kwargs):
    """
    Sample detectors/observables from `circuit` and decode with BP+OSD.

    Returns
    -------
    totalshots : int
    obs_errors : list[int]
        Failures per logical observable (predicted XOR actual).
    any_errors : int
        Shots with at least one logical failure after decoding.
    """
    dem = dem_from_circuit(circuit)
    decoder, matrices = make_bposd_decoder(dem, **bposd_kwargs)
    sampler = circuit.compile_detector_sampler(seed=seed)
    dets, obs = sampler.sample(nshots, separate_observables=True)
    pred = decode_observables_bposd(dets, decoder, matrices)
    fail = pred != obs
    totalshots = int(nshots)
    obs_errors = fail.sum(axis=0).astype(int).tolist()
    any_errors = int(fail.any(axis=1).sum())
    return totalshots, obs_errors, any_errors


def save_code_artifacts(outdir=None):
    outdir = outdir or os.path.dirname(__file__)
    np.save(os.path.join(outdir, 'hx.npy'), hx)
    np.save(os.path.join(outdir, 'hz.npy'), hz)
    np.save(os.path.join(outdir, 'lx.npy'), lx)
    np.save(os.path.join(outdir, 'lz.npy'), lz)
    with open(os.path.join(outdir, 'schedules.pkl'), 'wb') as f:
        pickle.dump({'zschedule': defaultzschedule, 'xschedule': defaultxschedule}, f)
    with open(os.path.join(outdir, 'logops.pkl'), 'wb') as f:
        pickle.dump({'lx': lx, 'lz': lz}, f)


# Default sweep (lighter than 832's 1e6; override in main)
numrandomizations = 1
nrounds_list = [1, 2]
ers = [0.001, 0.01]
nshots = 100000
outdir = 'sim'


def _run_one(circuit, syndromelength, nrounds, results_path, label, er):
    """Postselection + BP+OSD on one circuit; append a CSV row."""
    sample = circuit.compile_sampler().sample(nshots)
    totalshots, totalerrors, survived, survivederrors = sim_analysis(
        sample, syndromelength, nrounds, lx
    )
    bposd_shots, bposd_errors, bposd_any = sim_analysis_bposd(circuit, nshots)
    with open(results_path, 'a') as f:
        f.write(
            f'{label},{er},{totalshots},{survived},'
            f'"{survivederrors}","{totalerrors}",'
            f'{bposd_shots},{bposd_any},"{bposd_errors}"\n'
        )


def main():
    save_code_artifacts()
    os.makedirs(os.path.join(os.path.dirname(__file__), outdir), exist_ok=True)

    # Sanity: zero-noise circuit should give trivial syndromes + even logicals on |+>^n
    cb0 = c1844scheduleable(0, 1)
    circ0 = cb0.get_circuit()
    sample0 = circ0.compile_sampler().sample(1000)
    syndromelength = defaultzschedule.shape[0] + defaultxschedule.shape[0]
    totalshots, totalerrors, survived, survivederrors = sim_analysis(
        sample0, syndromelength, 1, lx
    )
    print(
        f'zero-noise check: survived={survived}/{totalshots}, '
        f'survivederrors={survivederrors}, totalerrors={totalerrors}'
    )
    assert survived == totalshots
    assert all(e == 0 for e in survivederrors)
    _, bposd_errors, bposd_any = sim_analysis_bposd(circ0, 1000)
    print(f'zero-noise BP+OSD: any_errors={bposd_any}, per_obs={bposd_errors}')
    assert bposd_any == 0

    # Base schedule diagram + small sweep
    cb = c1844scheduleable(0.01, 1)
    with open(os.path.join(os.path.dirname(__file__), outdir, 'default.txt'), 'w') as f:
        f.write(f'{defaultzschedule}\n{defaultxschedule}\n')
        f.write(str(cb.get_circuit().diagram()))

    results_path = os.path.join(os.path.dirname(__file__), outdir, 'default_sweep.csv')
    with open(results_path, 'w') as f:
        f.write(
            'nrounds,er,totalshots,survived,survivederrors,totalerrors,'
            'bposd_shots,bposd_any,bposd_errors\n'
        )

    for n in tqdm(nrounds_list, desc='nrounds'):
        for er in ers:
            cb = c1844scheduleable(er, n)
            _run_one(cb.get_circuit(), syndromelength, n, results_path, str(n), er)

    # Randomized time-labelings of the BB Tanner schedule (always conflict-free)
    for j in tqdm(range(numrandomizations), desc='random schedules'):
        randz = relabel_schedule(defaultzschedule)
        randx = relabel_schedule(defaultxschedule)
        with open(os.path.join(os.path.dirname(__file__), outdir, f'rand{j}.txt'), 'w') as f:
            f.write(f'relabel\n{randz}\n{randx}\n')
            f.write(str(c1844scheduleable(0.01, 1, randz, randx).get_circuit().diagram()))
        for n in nrounds_list:
            for er in ers:
                cb = c1844scheduleable(er, n, randz, randx)
                _run_one(
                    cb.get_circuit(), syndromelength, n, results_path, f'rand{j}_{n}', er
                )
    print(f'Wrote results to {results_path}')


if __name__ == '__main__':
    main()
