# QEC tools

Local installs of external QEC tools used with this workspace.

## tropidist
Certified code/circuit distance solver (Cheng, Guo, Pan; arXiv:2611.08247).

```sh
cd tropidist
.venv/bin/python -m cdist circuits/bb72_r6_Z.stim --out /tmp/bb72.json
.venv/bin/python -m cdist my_circuit.stim --threads 8 --time-limit 3600
```

## PropHunt
Syndrome-measurement schedule optimizer for CSS codes (needs loandra MaxSAT).

```sh
cd PropHunt
.venv/bin/python -c "from prop_hunt.prop_hunt import PropHuntCompiler; print('ok')"
# example experiment: see scripts/prophunt_experiment.py
```

`prop_hunt/loandra` is the MaxSAT binary (built from `../loandra`).
`prop_hunt/ambiguous_error.py` points SOLVER at that local binary.

## [[18,4,4]] PropHunt → tropidist check

Certify circuit distance of the baseline coloration schedule, optionally optimize
with PropHunt, then re-certify:

```sh
PropHunt/.venv/bin/python verify_1844_circuit_distance.py --skip-optimize
PropHunt/.venv/bin/python verify_1844_circuit_distance.py \
  --max-iter 10 --samples 200 --threads 8 --time-limit 600
```

Outputs land in `out_1844_verify/` (`baseline_{Z,X}.stim/.json`, optional
`optimized_*`, and `summary.json`).
