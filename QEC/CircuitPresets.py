import stim
import CircuitBuilder

def build_shor(pm, px, pz, pdp1, pdp2, pidle, nrounds):
    cb = CircuitBuilder.CircuitBuilder(pm, px, pz, pdp1, pdp2, pidle)
    # prepare GHZ blocks
    cb.ale('R 0 1 2 3 4 5 6 7 8')
    cb.ale('H 0 3 6')
    cb.ale('CX 0 1 3 4 6 7')
    cb.ale('CX 1 2 4 5 7 8')
    # prepare ancilla qubits
    cb.ale('R 9 10 11 12 13 14 15 16 17 18 19 20 21 22')

    # Initial stabilizer checks
    # Z
    cb.ale('CX 0 9 1 10')
    cb.ale('CX 1 9 2 10')
    cb.ale(f'MR({pm}) 9 10')
    cb.ale('CX 3 9 4 10')
    cb.ale('CX 4 9 5 10')
    cb.ale(f'MR({pm}) 9 10')
    cb.ale('CX 6 9 7 10')
    cb.ale('CX 7 9 8 10')
    cb.ale(f'MR({pm}) 9 10')
    # X
    cb.ale('H 11 17')
    cb.ale('CX 11 12 17 18')
    cb.ale('CX 11 13 17 19')
    cb.ale('CX 11 14 17 20')
    cb.ale('CX 11 15 17 21')
    cb.ale('CX 11 16 17 22')
    cb.ale('CX 11 0 12 1 13 2 14 3 15 4 16 5')
    cb.ale('CX 17 3 18 4 19 5 20 6 21 7 22 8')
    cb.ale('H 11 12 13 14 15 16 17 18 19 20 21 22')
    cb.ale(f'MR({pm}) 11 12 13 14 15 16 17 18 19 20 21 22')

    cb.begin_loop(nrounds)
    cb.idle_noise([0, 1, 2, 3, 4, 5, 6, 7, 8])
    # Z
    cb.ale('CX 0 9 1 10')
    cb.ale('CX 1 9 2 10')
    cb.ale(f'MR({pm}) 9 10')
    cb.ale('CX 3 9 4 10')
    cb.ale('CX 4 9 5 10')
    cb.ale(f'MR({pm}) 9 10')
    cb.ale('CX 6 9 7 10')
    cb.ale('CX 7 9 8 10')
    cb.ale(f'MR({pm}) 9 10')
    # X
    cb.ale('H 11 17')
    cb.ale('CX 11 12 17 18')
    cb.ale('CX 11 13 17 19')
    cb.ale('CX 11 14 17 20')
    cb.ale('CX 11 15 17 21')
    cb.ale('CX 11 16 17 22')
    cb.ale('CX 11 0 12 1 13 2 14 3 15 4 16 5')
    cb.ale('CX 17 3 18 4 19 5 20 6 21 7 22 8')
    cb.ale('H 11 12 13 14 15 16 17 18 19 20 21 22')
    cb.ale(f'MR({pm}) 11 12 13 14 15 16 17 18 19 20 21 22')
    cb.add_detectors([[-18, -36], [-17, -35], [-16, -34], [-15, -33], [-14, -32], [-13, -31]])
    cb.add_detectors([[-12, -11, -10, -9, -8, -7, -30, -29, -28, -27, -26, -25]])
    cb.add_detectors([[-6, -5, -4, -3, -2, -1, -24, -23, -22, -21, -20, -19]])

    cb.end_loop()

    cb.al('MX 0 1 2 3 4 5 6 7 8')
    cb.add_observable([-9, -8, -7, -6, -5, -4, -3, -2, -1])
    return cb.get_circuit()

def build_bacon_z(pm, px, pz, pdp1, pdp2, pidle, nrounds):
    cb = CircuitBuilder.CircuitBuilder(pm, px, pz, pdp1, pdp2, pidle)
    # Data qubits
    cb.ale('R 0 1 2 3 4 5 6 7 8')
    # Ancilla qubits
    cb.ale('R 9 10 11 12 13 14 15 16 17 18 19 20')
    # Initial stabilizer checks
    cb.ale('CX 0 9 1 10 3 11 4 12 6 13 7 14')
    cb.ale('CX 1 9 2 10 4 11 5 12 7 13 8 14')
    cb.ale(f'MR({pm}) 9 10 11 12 13 14')
    cb.ale('H 15 16 17 18 19 20')
    cb.ale('CX 15 0 16 3 17 1 18 4 19 2 20 5')
    cb.ale('CX 15 3 16 6 17 4 18 7 19 5 20 8')
    cb.ale('H 15 16 17 18 19 20')
    cb.ale(f'MR({pm}) 15 16 17 18 19 20')
    #cb.add_detector([-8, -7, -6, -5]) # Anchor ZZ gaugesd

    cb.begin_loop(nrounds)
    cb.al('TICK')
    cb.idle_noise([0, 1, 2, 3, 4, 5, 6, 7, 8])
    cb.ale('CX 0 9 1 10 3 11 4 12 6 13 7 14')
    cb.ale('CX 1 9 2 10 4 11 5 12 7 13 8 14')
    cb.ale(f'MR({pm}) 9 10 11 12 13 14')
    cb.ale('H 15 16 17 18 19 20')
    cb.ale('CX 15 0 16 3 17 1 18 4 19 2 20 5')
    cb.ale('CX 15 3 16 6 17 4 18 7 19 5 20 8')
    cb.ale('H 15 16 17 18 19 20')
    cb.ale(f'MR({pm}) 15 16 17 18 19 20')
    cb.add_detector([-12, -10, -8, -24, -22, -20])
    cb.add_detector([-11, -9, -7, -23, -21, -19])
    cb.add_detector([-6, -4, -2, -18, -16, -14])
    cb.add_detector([-5, -3, -1, -17, -15, -13])

    cb.end_loop()

    cb.al('MZ 0 3 6')
    cb.add_observable([-3, -2, -1])

    return cb.get_circuit()