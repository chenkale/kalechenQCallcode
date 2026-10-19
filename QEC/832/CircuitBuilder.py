# Class for tools to build circuits
import numpy as np
import stim

class CircuitBuilder:

    p_measurement_error = 0
    px_error = 0
    pz_error = 0
    pdp1_error = 0
    pdp2_error = 0
    pidle_error = 0

    circuit = None
    loop_circuit = None
    loop_num = 0
    in_loop = False
    all_qubits = set()

    def __init__(self, p_measurement_error=0, px_error=0, pz_error=0, pdp1_error=0, pdp2_error=0, pidle_error=0):
        """
        pidle_error is either a scalar, applied to every qubit, or a 1-d array
        whose entry i is the idle depolarizing probability of qubit i.
        An array also sets each CX DEPOLARIZE2 probability to the larger of
        the two qubits' idle probabilities. A scalar leaves CX noise on pdp2_error.
        """
        self.circuit = stim.Circuit()
        self.p_measurement_error = p_measurement_error
        self.px_error = px_error
        self.pz_error = pz_error
        self.pdp1_error = pdp1_error
        self.pdp2_error = pdp2_error
        self._pidle_per_qubit = isinstance(pidle_error, (list, tuple)) or (
            isinstance(pidle_error, np.ndarray) and np.asarray(pidle_error).ndim >= 1
        )
        if self._pidle_per_qubit:
            self.pidle_error = np.asarray(pidle_error, dtype=float).reshape(-1)
        else:
            self.pidle_error = pidle_error
        self.all_qubits = set()

    def _pidle_of(self, qubit):
        n = len(self.pidle_error)
        if qubit < 0 or qubit >= n:
            raise IndexError(f"pidle_error has length {n}, no entry for qubit {qubit}")
        return float(self.pidle_error[qubit])

    def _append_idle_depolarize(self, target_circuit, qubitslist):
        if not self._pidle_per_qubit:
            target_circuit.append('DEPOLARIZE1', qubitslist, self.pidle_error)
            return
        groups = {}
        for q in qubitslist:
            p = self._pidle_of(q)
            if p > 0:
                groups.setdefault(p, []).append(q)
        for p, qs in groups.items():
            target_circuit.append('DEPOLARIZE1', qs, p)

    def _append_cnot_depolarize(self, target_circuit, qubitslist):
        if not self._pidle_per_qubit:
            if self.pdp2_error > 0:
                target_circuit.append('DEPOLARIZE2', qubitslist, self.pdp2_error)
            return
        if len(qubitslist) % 2 != 0:
            raise ValueError(f"CX target list must be pairs, got {qubitslist}")
        groups = {}
        for i in range(0, len(qubitslist), 2):
            a, b = qubitslist[i], qubitslist[i + 1]
            p = max(self._pidle_of(a), self._pidle_of(b))
            if p > 0:
                groups.setdefault(p, []).extend([a, b])
        for p, qs in groups.items():
            target_circuit.append('DEPOLARIZE2', qs, p)

    def _get_target_circuit(self):
        return self.loop_circuit if self.in_loop else self.circuit


    def add_line(self, line):
        split_line = line.split(' ')
        gate = split_line[0]
        qubitslist = list(map(int, split_line[1:]))
        target_circuit = self._get_target_circuit()
        if '(' in gate:
            error_value = float(gate[gate.index('(')+1:gate.index(')')])
            gate = gate[:gate.index('(')]
            target_circuit.append(gate, qubitslist, error_value)
        else:
            target_circuit.append(gate, qubitslist)
        self.all_qubits.update(qubitslist)

    def add_line_with_auto_error(self, line, error_type='auto'):
        self.add_line(line)
        split_line = line.split(' ')
        gate = split_line[0]
        qubitslist = list(map(int, split_line[1:]))
        target_circuit = self._get_target_circuit()
        if error_type == 'auto':
            if gate == 'R':
                target_circuit.append('X_ERROR', qubitslist, self.px_error)
            elif gate == 'RX':
                target_circuit.append('Z_ERROR', qubitslist, self.pz_error)
            elif gate == 'H':
                target_circuit.append('DEPOLARIZE1', qubitslist, self.pdp1_error)
            elif gate == 'CX':
                self._append_cnot_depolarize(target_circuit, qubitslist)
            elif gate[:2] == 'MR':
                target_circuit.append('X_ERROR', qubitslist, self.px_error)
            else:
                raise Exception(f"Unsupported gate for auto error: {gate}")
        else:
            try:
                target_circuit.append(error_type, qubitslist)
            except Exception as e:
                raise Exception(f"Unsupported error type: {error_type}")
        self.all_qubits.update(qubitslist)

    def add_detector(self, record_indices):
        measurement_indices = [f'rec[{x}]' for x in record_indices]
        target_circuit = self._get_target_circuit()
        target_circuit.append('DETECTOR', measurement_indices)

    def add_detectors(self, record_indices_list):
        for record_indices in record_indices_list:
            self.add_detector(record_indices)

    def add_observable(self, observable_indices):
        measurement_indices = [f'rec[{x}]' for x in observable_indices]
        target_circuit = self._get_target_circuit()
        target_circuit.append('OBSERVABLE_INCLUDE', measurement_indices, 0)

    def set_ancilla_qubits(self, qubitslistx = None, qubitslistz = None):
        """
        Set a list of indices for ancilla qubits
        """
        self.ancilla_qubitsx = qubitslistx
        self.ancilla_qubitsz = qubitslistz

    def idle_noise(self, qubitslist):
        self._append_idle_depolarize(self._get_target_circuit(), qubitslist)

    def idle_noise_all(self):
        self._append_idle_depolarize(self._get_target_circuit(), list(self.all_qubits))

    def time_tick(self):
        self.add_line('TICK')
        self.idle_noise_all()

    def begin_loop(self, loop_num):
        self.loop_num = loop_num
        self.loop_circuit = stim.Circuit()
        self.in_loop = True

    def end_loop(self):
        loop = self.loop_circuit * self.loop_num
        self.circuit.append(loop)
        self.loop_circuit = None
        self.in_loop = False

    def couple_x_check(self, qubitslist, with_auto_error = False):
        """
        Perform ancilla coupling for an X stabilizer, indices in qubitslist coupled to indices in ancilla_qubits, use None for no instruction
        """
        if len(qubitslist) != len(self.ancilla_qubitsx):
            raise Exception(f"Number of qubits in qubitslist ({len(qubitslist)}) and ancilla_qubits ({len(self.ancilla_qubitsx)}) mismatch")
        instruction = f'CX'
        for i in range(len(qubitslist)):
            if qubitslist[i] is None:
                continue
            instruction += f' {self.ancilla_qubitsx[i]} {qubitslist[i]}'
        self.add_line_with_auto_error(instruction) if with_auto_error else self.add_line(instruction)

    def couple_z_check(self, qubitslist, with_auto_error = False):
        """
        Perform ancilla coupling for a Z stabilizer, indices in qubitslist coupled to indices in ancilla_qubits, use None for no instruction
        """
        if len(qubitslist) != len(self.ancilla_qubitsz):
            raise Exception(f"Number of qubits in qubitslist ({len(qubitslist)}) and ancilla_qubits ({len(self.ancilla_qubitsz)}) mismatch")
        instruction = f'CX'
        for i in range(len(qubitslist)):
            if qubitslist[i] is None:
                continue
            instruction += f' {qubitslist[i]} {self.ancilla_qubitsz[i]}'
        self.add_line_with_auto_error(instruction) if with_auto_error else self.add_line(instruction)

    def inject_error(self, qubitslist, error_type):
        """
        Inject an error into a list of qubits
        """
        for qubit in qubitslist:
            self.add_line(f'{error_type} {qubit}')

    def get_circuit(self):
        return self.circuit

    def al(self, line):
        self.add_line(line)

    def ale(self, line, error_type='auto'):
        self.add_line_with_auto_error(line, error_type)

    def sample(self, nshots=10000, asdict = True, custom = True):
        sampler = self.circuit.compile_sampler()
        if asdict and not custom:
            shots = sampler.sample_bit_packed(nshots)
            outdict = {}
            for row in shots:
                key = format(row[0], '02b')
                outdict[key] = outdict.get(key, 0) + 1
            return sorted(outdict.items())
        elif asdict and custom:
            shots = sampler.sample(nshots)
            outdict = {}
            for row in shots:
                bitstring = ''.join(['1' if x else '0' for x in row])
                outdict[bitstring] = outdict.get(bitstring, 0) + 1
            return sorted(outdict.items())    
        else:
            return sampler.sample(nshots)
    
    def sample_observable(self, nshots=10000, asdict = True):
        print('Not implemented yet')
        return None
        sampler = self.circuit.compile_sampler()
        if asdict:
            shots = sampler.sample(nshots)
            outdict = {}
            for row in shots:
                key = format(row[0], '02b')
                outdict[key] = outdict.get(key, 0) + 1
            return outdict
        else:
            return sampler.sample(nshots)
        