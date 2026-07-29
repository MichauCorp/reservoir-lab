# Findings: Optical-Acoustic Reservoir Coupling (exp10-15)

## The question

Does combining a fast, nonlinear "optical" reservoir with a slow, high-memory
"acoustic" reservoir produce a genuine computational advantage over either
substrate alone -- and if so, what specific mechanism, task structure, and
physical assumptions does that advantage actually depend on?

This document summarizes exp10 through exp15: what was tested, what held up,
what didn't, and what's still open. Raw numbers for every experiment below
are in `experiments/data_results/`; plots are in `experiments/visuals/`.

## Summary of the finding

**A specific coupling mechanism -- force-driving an acoustic lattice with an
optical reservoir's output -- produces a real, mechanistically-understood,
statistically robust advantage over both a no-coupling baseline and a
matched all-optical alternative, on tasks that require recovering a signal
component whose amplitude is smoothly gated by a slower signal's phase. That
advantage degrades gradually but survives physically realistic transduction
bandwidth limits. It does not generalize to sparse/pulse-like interaction
tasks, and it does not appear for parametric (damping-modulation) or
bidirectional coupling under the configurations tested.**

Every clause in that sentence was earned by a specific experiment ruling out
an alternative explanation. The rest of this document walks through them in
order.

## exp10-12: does coupling help, and how should it be done?

exp10-12 tested three ways of combining an `OptoelectronicReservoir` (fast,
nonlinear, short intrinsic memory) and an `AcousticReservoir` (slow,
long-memory Duffing lattice) on a task requiring recovery of two signal
components -- one slow, one fast -- from a noisy sum of the two:

- **Parallel**: both reservoirs driven independently by the raw input.
- **Serial force-coupling** (exp10): optical's output drives acoustic
  directly as a driving force; readout taps both stages' states.
- **Parametric coupling** (exp12): acoustic driven by raw input AND by
  optical's output modulating its damping coefficient.

On that first task (both components independently superimposed, no real
interaction between them), all three architectures performed statistically
indistinguishably. This was an important negative result: it meant nothing
could be concluded about coupling mechanisms from a task that didn't need
cross-timescale interaction in the first place.

## exp13: architecture x task-interaction sweep

exp13 introduced a task-interaction axis: a synthetic signal where a fast
component's target amplitude is gated by a slow component's phase (smooth
tanh-based gate), at three interaction strengths (0.0 = fully additive, 1.0 =
fully gated), crossed with two timescale-separation settings and 5
architectures (parallel, serial_force, parametric, parallel_bilinear -- an
explicit product-feature "ceiling" -- and mutual, a true bidirectional
coupling), across 4 seeds each.

**Result** (`experiments/data_results/exp13_results.csv`): at interaction=0,
all architectures were statistically indistinguishable, replicating exp10-12.
At interaction=1.0 with well-separated timescales, `serial_force` was the
clear winner (0.8622 +/- 0.0204 fast-target NRMSE vs. 0.9088 +/- 0.0093 for
parallel), beating every other architecture including the hand-built
"ceiling." That gap shrank sharply when timescales were made similar instead
of separated (0.8522 vs. 0.8773), matching the deep echo-state-network
literature's own finding that hierarchical coupling benefits specifically
depend on genuine timescale diversity between stages.

The surprising part: the fancier mechanisms (parametric, mutual) never beat
plain parallel in any condition tested. The simplest coupling won.

## exp14: mechanism, generalization, and a fair baseline

Three follow-up questions, since a single sweep result isn't enough to trust:

**1. Mechanism.** Is the advantage because force-coupling injects
gate-relevant information into the acoustic state, or an artifact of readout
capacity? A linear decoder trained on acoustic states *alone* (no optical, no
full-task readout) recovers the true gate signal at NRMSE 0.298 when acoustic
is force-driven by optical, vs. 1.232 (worse than predicting the mean) when
driven directly by raw input (`experiments/data_results/exp14_mechanism.csv`,
`experiments/visuals/exp14_gate_decodability.png`). This is a real, direct
effect, not a readout artifact.

**2. Generalization.** A second task family (`pulse_gate`: fast component
only appears in narrow windows around the slow signal's peaks, a sparse and
near-discontinuous interaction) was added alongside the original smooth
`tanh_gate` task. The advantage held on `tanh_gate` with more seeds (0.9005
vs. 0.8505) but nearly vanished on `pulse_gate` (0.9821 vs. 0.9795, within
noise). **Conclusion narrowed accordingly: force-coupling specifically helps
with smooth, continuous (AM-like) interaction, not sparse/pulse-like
interaction** (`experiments/data_results/exp14_generalization_baseline.csv`).

**3. Baseline.** An all-optical alternative (two `OptoelectronicReservoir`
stages, one fast and one deliberately slow, force-coupled the same way) was
built as the real competitive baseline -- not `parallel`. The first attempt
was confounded (stage2's `dt` defaulted to `theta/10`, coupling response-time
to delay-window size) and showed high variance. After decoupling `dt` from
`theta` and tuning `theta2` properly (best value 3.0, found via
`tune_all_optical_theta2()`), the all-optical baseline still lost to
`serial_force` on both task families, with normal, stable variance. This is
now a fair comparison, not an artifact of unequal tuning effort.

## exp15: does physical realism change any of this?

The idealized coupling in exp10-14 assumes optical's output reaches acoustic
instantaneously. Real optoelectronic reservoirs operate at 12-60 GHz; a real
piezo-optomechanical transducer achieves ~25 MHz bandwidth; practical
mechanical reservoir memory reaches millisecond-scale delays. With ~90
virtual nodes per outer timestep, those numbers imply a realistic transducer
time constant of roughly 5-27 *outer timesteps* -- comparable to or longer
than the fast signal's own period (12 timesteps) in the task used throughout.

exp15 modeled this as a single-pole low-pass filter on the optical-to-acoustic
path (parameterized by `tau`, in outer timesteps; `tau=0` recovers the
idealized case), applied *only* to the cross-domain hybrid coupling -- not to
`all_optical_dual_delay`'s own coupling, since staying in one physical domain
doesn't require a lossy energy conversion the way light-to-mechanical-motion
does.

**Result**: contrary to the expectation that a realistic bandwidth limit
would erode or reverse the advantage, it degraded gradually and never
reversed, across the entire sweep including deliberately pessimistic values.
At the realistic range (tau=5-27), `serial_force` retained roughly half to
three-quarters of its idealized advantage over parallel, and beat the
unpenalized all-optical baseline at every single tau value tested, including
tau=100 (`experiments/data_results/exp15_architecture_vs_tau.csv`,
`experiments/visuals/exp15_architecture_vs_tau.png`). The underlying
mechanism (gate decodability from acoustic state alone) survived the same
sweep similarly gracefully (`experiments/data_results/exp15_mechanism_vs_transduction.csv`).

This is the most load-bearing result in the whole investigation, precisely
because it was the check most likely to break everything above it, and
didn't.

## What this does *not* establish

- **Only one task family shows the effect clearly.** Both tasks used here are
  synthetic. Nothing here has been tested against an established benchmark
  or real-world signal.
- **Only bandwidth-limiting was modeled in exp15**, not transduction
  noise/inefficiency (no well-grounded number exists for a *classical*
  opto-mechanical link specifically -- the bandwidth number is solid, an
  efficiency number was deliberately left out rather than misapply a
  quantum-regime figure) or propagation latency (a real, separate effect on a
  comparable order of magnitude).
- **Only one all-optical alternative was tested** (two force-coupled delay
  loops). Other all-optical architectures (mutually-coupled lasers,
  wavelength-multiplexed designs) remain untested and might close the gap.
- **Parametric and mutual coupling underperformed in every configuration
  tested here**, but the specific implementations were not swept as
  thoroughly as force-coupling was. This is evidence against those specific
  designs, not a proof that no parametric or bidirectional mechanism could
  ever work.

## Where things stand

The claim that survives everything tested so far, stated at the scope it's
actually earned: *force-coupling an optical reservoir into an acoustic
reservoir, with well-separated timescales, produces a real and
mechanistically-understood advantage over both no coupling and a matched
all-optical alternative, specifically for smooth cross-timescale amplitude
gating tasks, and that advantage is robust to physically realistic
transduction bandwidth limits.*

Natural next steps, roughly in order of how much they'd change the above
conclusion if the answer came back unfavorable: testing on a real or
established benchmark task; modeling transduction noise and propagation
latency explicitly; and testing additional all-optical alternatives beyond
the one built here.

## Reproducing this

Each experiment is runnable standalone from the repo root:

```bash
uv run experiments/exp13_architecture_sweep.py
uv run experiments/exp14_mechanism_and_baseline.py
uv run experiments/exp15_physical_transduction.py
```

`exp10`-`exp12` establish the reasoning trail but are superseded by
`exp13`'s more rigorous multi-seed, multi-condition version of the same
comparison -- read them for context, treat `exp13` onward as the results
that matter.

## Where the data lives

| File | Contents |
|------|----------|
| `experiments/data_results/exp13_results.csv` | Every individual run: architecture x interaction x timescale x seed (120 rows) |
| `experiments/data_results/exp14_mechanism.csv` | Gate decodability, acoustic_direct vs acoustic_force_coupled |
| `experiments/data_results/exp14_generalization_baseline.csv` | task x architecture x seed, both task families (36 rows) |
| `experiments/data_results/exp15_mechanism_vs_transduction.csv` | Gate decodability across the transducer tau sweep |
| `experiments/data_results/exp15_architecture_vs_tau.csv` | Full architecture comparison across the transducer tau sweep |

New experiments should follow the same convention: write per-run raw
results (not just aggregates) to `experiments/data_results/expNN_*.csv`,
and any plots to `experiments/visuals/expNN_*.png`, so `expNN`'s number
alone is enough to find its script, its data, and its plot.