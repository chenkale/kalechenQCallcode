from promhookerror import randomize_schedule, count_higher_weight_faults
import numpy as np
import os
hx = np.load(os.path.join(os.path.dirname(__file__), 'hx.npy'))[0:7]
hz = np.load(os.path.join(os.path.dirname(__file__), 'hz.npy'))[0:7]

assert np.all((hx @ hz.T) % 2 == 0)

randzschedule = randomize_schedule(hz)
randxschedule = randomize_schedule(hx)
print(randzschedule)
print(randxschedule)
# Z-type faults, detected by X checks
Lz, z_counts = count_higher_weight_faults(
    randzschedule, hx, hz, max_faults=4
)

print("Z fault locations:", Lz)
for w, result in z_counts.items():
    print(f"Weight {w}: {result}")
