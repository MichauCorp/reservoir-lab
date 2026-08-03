# Findings: Optical-Acoustic Reservoir Coupling (exp10-22)

## The question

Does combining a fast, nonlinear "optical" reservoir with a slow, high-memory
"acoustic" reservoir produce a genuine computational advantage over either
substrate alone -- and if so, what specific mechanism, task structure, and
physical assumptions does that advantage actually depend on?

This document summarizes exp10 through exp22: what was tested, what held up,
what didn't, and what's still open. Raw numbers for every experiment below
are in `experiments/data_results/`; plots are in `experiments/visuals/`.

## Summary of the finding

**A specific coupling mechanism -- force-driving an acoustic lattice with an
optical reservoir's output -- produces a real, mechanistically-understood
advantage over both a no-coupling baseline and a matched all-optical
alternative, on tasks that require recovering a signal component whose
amplitude is smoothly gated by a slower signal's phase. That advantage is
robust to physically realistic transduction bandwidth limits, requires
properly-tuned regularization to realize (it is more sensitive to
under-regularization than the alternatives, for reasons not yet explained),
and depends on the fast component being narrowband/sinusoidal rather than
broadband/transient -- a dependency discovered by testing on real
physiological data, then confirmed and isolated in a controlled synthetic
follow-up. It does not generalize to sparse/pulse-like interaction tasks,
and it does not appear for parametric (damping-modulation) or bidirectional
coupling under the configurations tested.**

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

This was, at the time, the most load-bearing result in the investigation --
the check most likely to break everything above it, and it didn't. (exp16
onward found a different way the picture was incomplete; see below.)

## exp16: hyperparameter sweep, a real bug, and regularization sensitivity

exp16 swept architecture x hyperparameters (theta, acoustic `c`, reservoir
size, readout regularization) x task (the original `tanh_gate` plus two
standard RC benchmarks, `narma10` and `mackey_glass`) x 10 seeds, to check
whether exp13-15's conclusions held outside the one hyperparameter setting
used throughout.

**The first run was misleading, and it was a real code bug, not just a
different metric.** `build_all_optical` set its second stage's `theta = c *
10` with no explicit `dt`, silently reintroducing the exact `dt`-scales-with-
`theta` confound exp14 had already found and fixed once. Separately, the
first version only tracked `nrmse_vals[0]` (the *slow* target for
`dual_timescale`) as the "primary metric" -- never measuring the fast target
the entire investigation had been about. Both were fixed: `all_optical` now
gets its own independently-tuned `theta2` grid axis with `dt` fixed, and
slow/fast are tracked and reported separately.

**With both fixed, re-running showed the core finding replicated, with an
important caveat.** Averaged across the *entire* grid including badly
under-regularized settings, `serial_force` looked *worse* than the
alternatives on the fast target (0.90 +/- 0.21 vs. parallel's 0.88 +/- 0.10
and all_optical's 0.85 +/- 0.06) -- but breaking this down by regularization
strength showed why: at `reg=1e-6` (too little), `serial_force` degrades
catastrophically (up to NRMSE 1.56) while the alternatives stay comparatively
stable; at `reg=1e-3` (the value used throughout exp10-15), `serial_force`
wins cleanly and with the *tightest* spread of the three (0.8255 +/- 0.0214
vs. 0.8587 +/- 0.0399 for parallel and 0.8811 +/- 0.0352 for all_optical).

**New finding: `serial_force` is more sensitive to under-regularization than
the alternatives.** This is real (confirmed at proper regularization across
80 samples, far more statistical power than exp13's original 4-seed test)
and it's a genuine practical caveat, not evidence against the core result.

**Also new: a slow/fast tradeoff.** At `reg=1e-3`: `parallel` beats
`serial_force` on the slow target (0.1735 vs. 0.2008), while `serial_force`
wins on fast. Small, but consistent.

**On the generic benchmarks:** `serial_force` shows no special advantage over
`parallel` on either `mackey_glass` (0.0645 vs. 0.0697, close) or `narma10`
(0.7033 vs. 0.7053, statistically tied) -- confirming the advantage is
specific to interaction-structured tasks, not general-purpose reservoir
quality. `all_optical` is clearly worse on `narma10` (0.8365) but competitive
on `mackey_glass` (0.0680).

(`experiments/data_results/exp16_hyperparameter_sweep.csv`, 192 rows)

## exp17: why is serial_force regularization-sensitive? (open question)

Three candidate explanations were tested directly, in order of cheapness, and
all three were refuted:

1. **Feature correlation / effective rank** (hypothesis: force-coupling makes
   acoustic's channels more redundant, hence more sensitive to
   under-regularization). Refuted: `serial_force` had *higher* effective
   rank than both alternatives at every hyperparameter combination tested,
   including the exact failure-case config (11.97 vs. 2.97 for parallel and
   3.18 for all_optical).
2. **Feature scale** (hypothesis: larger-magnitude features make a fixed
   `reg` relatively weaker). Refuted: `serial_force`'s mean feature variance
   was *smaller* than parallel's, the wrong direction to explain the effect.
3. **Smallest singular value vs. `reg`** (the mathematically precise quantity
   ridge stability depends on). Uninformative: all three architectures
   bottom out at numerically negligible singular values (1e-23 to 1e-32),
   dominated by floating-point noise common to any high-dimensional
   reservoir state, not a real architectural signal.

**This remains an open question.** The effect from exp16 is real and
replicated; its cause is not a simple linear-algebra property of the feature
matrix. A proper answer would need to examine how the *target* (specifically
the fast/gate signal) projects onto each architecture's singular directions
-- a real ridge bias-variance decomposition, not a quick check.
(`experiments/data_results/exp17_condition_number.csv`,
`experiments/visuals/exp17_singular_spectra.png`)

## exp18: real-signal validation (genuine ECG, not synthetic)

Every task up to this point was hand-constructed. exp18 tests on a genuine
physiologically-recorded 5-minute human ECG signal (scipy's bundled dataset,
not synthesized), with the slow/fast targets extracted via standard
zero-phase Butterworth filtering of the real recording, not fabricated.

**The coupling was independently verified before building any task on it**,
because assuming it would have defeated the purpose: real ECG is known to
exhibit respiratory sinus arrhythmia (QRS amplitude genuinely varies with the
respiratory cycle). Binning the fast-band signal's envelope by the slow
component's instantaneous *phase* (proper PAC methodology, not naive
correlation) showed real modulation (~35-39% depth) while raw value-
correlation was near zero -- exactly the phase-locked, not value-locked,
structure real phase-amplitude-coupling research looks for.

**Result: partial validation, and a genuinely new finding, not a clean
replication.** `serial_force` won clearly on the *slow* target (0.3393 vs.
0.3512 for parallel, 0.7140 for all_optical) but *lost* the fast target to
all_optical (0.8821 vs. 0.8683) -- the opposite of every synthetic task's
pattern, where the win was specifically on fast.
(`experiments/data_results/exp18_real_signal_ecg.csv`,
`experiments/visuals/exp18_real_signal_ecg.png`)

## exp19: real-ish speech signal (fixed a critical bug along the way)

A synthetic-but-speech-realistic task (glottal pulse train through a
formant resonance filter, slow gate on voicing amplitude). **The first
version had a critical bug**: the model's input was generated independently
of the gate/formant entirely, meaning the input carried zero information
about what the model was asked to predict -- the task was unsolvable by
construction, for every architecture equally, and the reported NRMSE
(~0.82-0.84 across the board) reflected that, not architecture performance.

**Fixed**: input is now the actual noisy formant-filtered waveform (which
genuinely carries the gate-modulated amplitude), target is that signal's
Hilbert envelope. Re-run produced sensible, non-degenerate results:
`parallel` significantly beat `serial_force` (0.2569 +/- 0.0019 vs. 0.2852
+/- 0.0088, p=0.0045), with `all_optical` clearly worst (0.5859 +/- 0.0808,
high variance) and a `simple_esn` baseline in between (0.2941 +/- 0.0411).
Consistent with exp16 and exp18's slow/fast tradeoff pattern: `serial_force`
lost on this envelope/slow-type target, same direction as real ECG.
(`experiments/data_results/exp19_speech_formant_tracking.csv`)

## exp20-22: does the fast/slow win pattern depend on waveform shape?

Two real signals (ECG, speech) both showed the win pattern reversed or
weakened relative to every synthetic task. The one thing consistently
different: the synthetic "fast" component was always a clean sinusoid; real
ECG's QRS complexes and speech's glottal pulses are broadband, transient-rich
waveforms.

**exp20 isolated this as a controlled variable**: two versions of the
`tanh_gate` task, identical in every respect (slow component, gate, phase-
modulation timing, noise, amplitude scale) except whether the fast component
is a smooth sinusoid or a sharp periodic pulse train (broadband, same
fundamental period). Result: on the fast target, `serial_force` still won
for both waveform types, but the margin over all_optical shrank by roughly
an order of magnitude (0.085 gap for sinusoid -> 0.012 gap for broadband).
On the slow target, a genuine reversal: `serial_force` beat `parallel` for
sinusoid (0.2089 vs. 0.2328) but lost to it for broadband (0.2272 vs.
0.2147) -- reproducing, in a controlled synthetic setting, the same reversal
seen on real ECG. (`experiments/data_results/exp20_waveform_shape.csv`)

**exp21 tested whether removing the broadband content before it reaches
optical fixes this.** It doesn't, cleanly -- it's a tradeoff, not a fix.
Pre-filtering improved slow-target accuracy for *both* architectures
similarly, turning the broadband reversal into a near-exact tie
(serial_force_prefilter 0.1611 vs. parallel_prefilter 0.1616), not a
restored serial_force win. And it completely destroys fast-target
performance for both prefiltered variants (NRMSE ~1.0, no better than
predicting the mean), since the filter removes the fast content before any
reservoir ever sees it. (An earlier version of this file's docstring
overstated this as "restoring the advantage" -- corrected in the script
itself; see its docstring for the full accounting.)

**exp22 tested the specific mechanism exp21's hypothesis proposed**: that
optical's *short* memory (theta=0.2) smears broadband transients, and longer
memory should fix it without needing to delete information. It doesn't. The
first sweep was itself confounded (theta and `dt` coupled again, causing a
spurious catastrophic failure at theta=6.0); after fixing `dt` at 0.02 across
the whole sweep, the corrected result is clean and negative: increasing
optical's theta never recovers broadband slow-target performance -- it
degrades further (0.227 at theta=0.2, peaking at 0.536 at theta=1.5, only
partially recovering to 0.256 at theta=6.0, never matching the original
baseline). `parallel` stays essentially flat throughout (0.214-0.218),
confirming this is specific to `serial_force`, not a general optical-timescale
effect. **The hypothesis is refuted, cleanly, and the real mechanism behind
the broadband-waveform dependency remains open** -- likely something about
how the transient shape interacts with optical's nonlinearity, not its
memory duration. (`experiments/data_results/exp22_optical_theta_sweep.csv`,
`experiments/visuals/exp22_optical_theta_sweep.png`)

## What this does *not* establish

- **The waveform-shape mechanism is unexplained.** Two specific hypotheses
  (input filtering as a fix, optical memory duration as the cause) were
  tested cleanly and both refuted. The actual mechanism connecting broadband
  fast content to serial_force's degraded slow-target performance is not
  yet known.
- **`serial_force`'s regularization sensitivity is also unexplained** (exp17)
  for the same reason: three candidate linear-algebra explanations tested,
  none confirmed.
- **Real-signal validation exists for two domains** (ECG, a speech-realistic
  synthetic signal) and both showed the *slow/fast tradeoff*, not a clean
  replication of the synthetic-task win. Whether this generalizes to other
  real signals, and whether a genuinely different real dataset would show
  the same pattern, is untested.
- **Only bandwidth-limiting was modeled in exp15**, not transduction
  noise/inefficiency or propagation latency.
- **Only one all-optical alternative was tested** throughout (two
  force-coupled delay loops). Other all-optical architectures remain
  untested.
- **Parametric and mutual coupling underperformed in every configuration
  tested**, but were not swept as thoroughly as force-coupling. Evidence
  against those specific designs, not proof no such mechanism could work.

## Where things stand

The claim that survives everything tested so far, stated at the scope it's
actually earned: *force-coupling an optical reservoir into an acoustic
reservoir, with well-separated timescales and properly-tuned regularization,
produces a real and mechanistically-understood advantage over both no
coupling and a matched all-optical alternative, specifically for smooth
cross-timescale amplitude gating tasks with a narrowband (sinusoidal) fast
component. That advantage is robust to physically realistic transduction
bandwidth limits. On broadband/transient fast content -- including two
independent real-signal tests -- the advantage narrows substantially or
reverses on the slow target specifically, and neither of two tested
explanations (input filtering, optical memory tuning) resolves this.*

This is a materially narrower and more precise claim than the version of this
document written after exp15, and that narrowing came entirely from testing
on real data rather than more synthetic variants -- the single highest-value
methodological decision in this phase of the investigation.

Natural next steps, in rough priority order: fixing the `dt`-scales-with-
`theta` confound at its source (it has now caused problems in exp14, exp16,
and the first draft of exp22 -- a systemic trap, not bad luck); a second
real-signal test from a genuinely different, non-biophysical domain to check
whether the narrowband/broadband boundary is a general principle or an
ECG-specific artifact; and, lower priority, a fresh mechanism hypothesis for
either the regularization sensitivity or the waveform-shape dependency, now
that two candidate explanations for each have been ruled out.

## Reproducing this

Each experiment is runnable standalone from the repo root:

```bash
uv run experiments/exp16_hyperparameter_sweep.py --quick
uv run experiments/exp18_real_signal_ecg.py
uv run experiments/exp20_waveform_shape.py
uv run experiments/exp22_optical_theta_sweep.py
```

`exp10`-`exp12` establish the reasoning trail but are superseded by
`exp13`'s more rigorous multi-seed, multi-condition version of the same
comparison -- read them for context, treat `exp13` onward as the results
that matter. `exp17` is a negative/open-question result, not a dead end to
skip -- it documents what's still unexplained.

## Where the data lives

| File | Contents |
|------|----------|
| `experiments/data_results/exp13_results.csv` | Every individual run: architecture x interaction x timescale x seed (120 rows) |
| `experiments/data_results/exp14_mechanism.csv` | Gate decodability, acoustic_direct vs acoustic_force_coupled |
| `experiments/data_results/exp14_generalization_baseline.csv` | task x architecture x seed, both task families (36 rows) |
| `experiments/data_results/exp15_mechanism_vs_transduction.csv` | Gate decodability across the transducer tau sweep |
| `experiments/data_results/exp15_architecture_vs_tau.csv` | Full architecture comparison across the transducer tau sweep |
| `experiments/data_results/exp16_hyperparameter_sweep.csv` | architecture x task x theta x c x n x reg x seed (192 rows, quick mode) |
| `experiments/data_results/exp17_condition_number.csv` | Condition number / effective rank per architecture x hyperparameter combo |
| `experiments/data_results/exp18_real_signal_ecg.csv` | Real ECG task: architecture x seed, slow/fast NRMSE |
| `experiments/data_results/exp19_speech_formant_tracking.csv` | Fixed speech task: architecture x seed |
| `experiments/data_results/exp20_waveform_shape.csv` | waveform (sinusoid/broadband) x architecture x seed |
| `experiments/data_results/exp21_prefiltered_input.csv` | waveform x architecture (incl. prefilter variants) x seed |
| `experiments/data_results/exp22_optical_theta_sweep.csv` | waveform x theta x architecture x seed (fixed-dt version) |

New experiments should follow the same convention: write per-run raw
results (not just aggregates) to `experiments/data_results/expNN_*.csv`,
and any plots to `experiments/visuals/expNN_*.png`, so `expNN`'s number
alone is enough to find its script, its data, and its plot.