import numpy as np
import random
# Hardcoded gates
GATES = {
    'X': np.array([[0, 1], [1, 0]]),
    'Y': np.array([[0, -1j], [1j, 0]]),
    'Z': np.array([[1, 0], [0, -1]]),
    'H': np.array([[1, 1], [1, -1]]),
    'S': np.array([[1, 0], [0, 1j]]),
    'T': np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]]),
    'I': np.eye(2),
    'CX': np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]),
}

def commutes(gatestring1, gatestring2):
    if len(gatestring1) != len(gatestring2):
        raise ValueError("Gate strings must have the same length")
    
    total_gate = GATES[gatestring1[0]]
    for gate in gatestring1[1:]:
        total_gate = np.kron(total_gate, GATES[gate])
    total_gate2 = GATES[gatestring2[0]]
    for gate in gatestring2[1:]:
        total_gate2 = np.kron(total_gate2, GATES[gate])
    return np.allclose(np.dot(total_gate, total_gate2), np.dot(total_gate2, total_gate))

def display_tensor_product(gatestring):
    total_gate = GATES[gatestring[0]]
    for gate in gatestring[1:]:
        total_gate = np.kron(total_gate, GATES[gate])
    print(total_gate)
    return total_gate

def display_matrix_product(gatestring):
    total_gate = GATES[gatestring[0]]
    for gate in gatestring[1:]:
        total_gate = np.dot(total_gate, GATES[gate])
    print(total_gate)
    return total_gate

class Statevector:

    state = np.array([1, 0])
    norm_factor = 1

    # Constructor that handles:
    # - No arguments: default state |0⟩
    # - State vector (array/list): automatically normalizes
    # - Bitstring (str): constructs state from bitstring
    def __init__(self, arg=None, mirrored=False):
        if arg is None:
            # Default constructor
            self.state = np.array([1, 0])
            self.norm_factor = 1
        elif isinstance(arg, str):
            # Constructor from bitstring
            if arg == '':
                self.state = np.array([1, 0])
                self.norm_factor = 1
            else:
                self.state = np.array([1])
                for i in range(len(arg)):
                    if arg[i] == '0':
                        self.state = np.kron(self.state, np.array([1, 0]))
                    else:
                        self.state = np.kron(self.state, np.array([0, 1]))
                if mirrored:
                    for i in range(len(self.state)):
                        if self.state[i] != 0:
                            self.state[-i - 1] = self.state[i]
                self.norm_factor = 1 / np.sqrt(np.sum(np.abs(self.state)**2))
        else:
            # Constructor from state vector
            self.state = np.array(arg)
            self.norm_factor = 1 / np.sqrt(np.sum(np.abs(arg)**2))
            
    def apply_gates(self, gatestring):
        if len(gatestring) != self.num_digits():
            raise ValueError("Gate string length must match number of qubits")
        if not all(gate in GATES for gate in gatestring):
            raise ValueError("Invalid gate string")
        
        total_gate = GATES[gatestring[0]]
        for gate in gatestring[1:]:
            total_gate = np.kron(total_gate, GATES[gate])
        self.state = np.dot(total_gate, self.state)
        self.normalize()

    def measure(self, gatestring, shots=1):
        if len(gatestring) != self.num_digits():
            raise ValueError("Gate string length must match number of qubits")
        if not all(gate in GATES for gate in gatestring):
            raise ValueError("Invalid gate string")

        state_copy = Statevector(self.state, mirrored=False)
        state_copy.apply_gates(gatestring)
        out = np.zeros(len(self.state))
        for i in range(len(self.state)):
            out[i] = self.state[i] / state_copy.state[i] if state_copy.state[i] != 0 else 0
        measurement_outcomes = []
        for i in range(len(out)):
            if out[i] != 0:
                measurement_outcomes.append(out[i])
        return [np.random.choice(measurement_outcomes) for i in range(shots)]

    def switch_basis(self):
        hstring = 'H' * self.num_digits()
        self.apply_gates(hstring)
        # Temporary fix for double basis switch causing incorrect normalization
        if np.any(self.state > 1):
            self.norm_factor *= np.max(self.state)
            self.state = self.state / np.max(self.state)
        self.niceify()

    def normalize(self):
        self.norm_factor = 1 / np.sqrt(np.sum(np.abs(self.state)**2))
        self.niceify()

    def num_digits(self):
        return int(np.log2(len(self.state)))

    def ket(self):
        out = '1/√' + str(round(np.sum(np.abs(self.state)**2), 0)) + ' (' if self.norm_factor != 1 else '('
        ketstring = ''
        for i in range(len(self.state)):
            if self.state[i] > 0:
                if self.state[i] != 1:
                    ketstring += ' + ' + str(self.state[i].round(2)) + '|' + str(bin(i).replace("0b", "").rjust(self.num_digits(), "0")) + '⟩'
                else:
                    ketstring += ' + |' + str(bin(i).replace("0b", "").rjust(self.num_digits(), "0")) + '⟩'
            elif self.state[i] < 0:
                if self.state[i] != -1:
                    ketstring += ' - ' + str(self.state[i].round(2)) + '|' + str(bin(i).replace("0b", "").rjust(self.num_digits(), "0")) + '⟩'
                else:
                    ketstring += ' - |' + str(bin(i).replace("0b", "").rjust(self.num_digits(), "0")) + '⟩'
        out += ketstring[3:] + ')' if ketstring[1] == '+' else ketstring[1:] + ')'
        return out

    def niceify(self):
        for i in range(len(self.state)):
            if np.isclose(self.state[i], 1.0, atol=1e-9):
                self.state[i] = int(1)
            elif np.isclose(self.state[i], -1.0, atol=1e-9):
                self.state[i] = int(-1)
            elif np.isclose(self.state[i], 0.0, atol=1e-9):
                self.state[i] = int(0)
        if np.isclose(self.norm_factor, 1.0, atol=1e-9):
            self.norm_factor = int(1)
        
    def __str__(self):
        return str(self.state)

    def __repr__(self):
        return repr(self.state)

    def __len__(self):
        return len(self.state)





