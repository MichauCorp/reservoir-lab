# Reservoir Computing

Exploring physical reservoir computing: single-substrate baselines
(`exp01`-`exp09`) and a multi-substrate optical-acoustic coupling
investigation (`exp10`-`exp15`).

**See [FINDINGS.md](FINDINGS.md) for the optical-acoustic coupling
results** -- what was tested, what held up, what didn't, and what's still
open.

## Structure

- `src/reservoir_lab/` -- the package. `physical/` holds each reservoir
  substrate (`OptoelectronicReservoir`, `AcousticReservoir`,
  `SerialPhysicalReservoir`, `ParametricallyCoupledReservoir`,
  `MutualCoupledReservoir`) plus `PhysicalReservoir`, the shared interface
  they all implement.
- `experiments/` -- one script per numbered experiment (`expNN_*.py`),
  runnable standalone.
- `experiments/data_results/` -- raw per-run CSV output from each
  experiment, named `expNN_*.csv` to match its script.
- `experiments/visuals/` -- saved plots, named `expNN_*.png` to match.
- `tests/` -- pytest suite.
- `notebooks/` -- exploratory work.

## Running an experiment

```bash
uv run experiments/exp13_architecture_sweep.py
```

## Running tests

```bash
uv run pytest
```