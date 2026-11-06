# Persistent State Lab

**Can an earlier experience still determine an answer after the recent inputs
become identical?** A small, reproducible experiment in learned associative memory.

[中文说明](README.zh-CN.md)

Two histories contain the same six symbols with different pairings. They then
receive the same sixteen distractor events and the same questions. The model
retains a 32 × 32 numerical state. We keep its parameters fixed and intervene on
that state: retain it, clear it, or replace it with the other history's state.

```text
history A: (18,23), (6,16), (3,7) ──┐
                                   ├─ same later input ── state ── query 18
history B: (18,16), (6,23), (3,7) ──┘

state from A → 23        state from B → 16
```

The numbers illustrate one observed test case. The full protocol scores 256
held-out sequences per seed; it does not select examples for their accuracy.

## Why this question

The motivating research direction is to let experience leave persistent internal
conditions that can participate in later computation. This experiment examines
one necessary behavior: retaining and selecting earlier content. State removal
and replacement help distinguish usable content from a state that merely changes
numerically.

The broader direction also requires interpretation, selective updates and
revision. The last diagnostic makes one missing piece visible: replacing a
forward association leaves an obsolete reverse association behind.

## Run it

Python 3.11 or newer is recommended. The recorded run used Python 3.14 and
PyTorch 2.9.1 on one CPU thread. No API key, pretrained model or external dataset
is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s . -p 'test_*.py' -v
python experiment.py --output runs/my-first-run
```

The default run trains two independently initialized models, seeds 7 and 19,
for 2,000 steps each. It took about 43 seconds on the recorded machine. Runtime
depends on hardware. An output directory must be new, so earlier results are
preserved. Generated weights remain in the ignored `runs/` directory.

## Recorded results

This table comes from [the recorded results](results/reference.json).
Each seed has 256 evaluation sequences. Scores are exact symbol matches.

| Intervention or baseline | Seed 7 | Seed 19 |
| --- | ---: | ---: |
| Retain state; query all six original symbols | 1,536 / 1,536 | 1,536 / 1,536 |
| Clear state; score three queries against original history | 35 / 768 | 61 / 768 |
| Substitute history B's state; score against history A | 256 / 768 | 256 / 768 |
| Substitute history B's state; score against history B | 768 / 768 | 768 / 768 |
| Changed queries correct under **both** histories | 512 / 512 | 512 / 512 |
| Replace one forward binding; read its new value | 256 / 256 | 256 / 256 |
| Other two bindings preserved after replacement | 512 / 512 | 512 / 512 |
| **Failure: obsolete reverse link still returned** | **256 / 256** | **256 / 256** |
| Exact dictionary with complete history | 768 / 768 | 768 / 768 |
| Exact dictionary with only the sixteen recent events | 0 / 768 | 0 / 768 |

One of the three queried bindings is deliberately unchanged between histories.
That explains the 256 correct answers when donor state is scored against the
original history. The two changed answers follow the donor state's contents.
The recorded report also includes untrained scores, learning curves, source
hashes and all eight displayed examples, chosen as the first four per seed.
The readable export omits dense example matrices; the complete generated report
and checkpoints remain in each local run directory. Create the same export with:

```bash
python export_results.py runs/my-first-run/report.json results/my-run.json
```

These observations establish state-dependent symbolic retention under this
protocol. The dictionary baseline also solves the task: this study establishes
neither an advantage over exact storage nor a new memory algorithm.

## Mechanism and experimental boundaries

An event contains one or two symbols. An embedding and within-event attention
produce write addresses, values and strengths. The update uses an established
delta-rule form:

```text
M' = M + Σᵢ βᵢ (vᵢ − M kᵢ) kᵢᵀ
readout(q) = Wout M normalize(Wkey embedding(q))
```

The write method receives events and the preceding state. Questions and target
answers enter only the readout and training loss, respectively. After training,
observing an event changes the state while model parameters remain fixed. The
state owner keeps no event text or symbol sequence for later replay.

The task provides pair boundaries; all 24 symbols are known during training,
while evaluation pairs are held out. Distractors are single-symbol events drawn
from other symbols. State width is 32, so this is not a capacity stress test.
Natural-language understanding, unfamiliar symbols, realistic interference and
reliable bidirectional correction remain outside the demonstrated behavior.
Two fixed seeds provide descriptive results.

## Files

- [`memory.py`](memory.py): model, synthetic world and state owner.
- [`experiment.py`](experiment.py): training, interventions, baselines and reports.
- [`export_results.py`](export_results.py): readable export without dense matrices.
- [`test_memory.py`](test_memory.py) and [`test_experiment.py`](test_experiment.py):
  data separation, state isolation, gradient flow and intervention contracts.
- [`results/reference.json`](results/reference.json): a fresh synthetic run,
  including successful measurements and the stale-link failure.

## Contribution and method attribution

TAN-ZI-YI directs the research questions, desired behaviors, design discussion and
evaluation priorities. AI coding agents implemented and adapted the code and ran
the recorded experiments. This repository does not represent code independently
written by TAN-ZI-YI, and the specific delta-rule mechanism is not claimed as an
original invention. The public study packages one implemented candidate to make
its behavior and remaining gaps inspectable.

The associative update belongs to the established fast-weight/delta-rule family.
For related background, see Schlag, Irie and Schmidhuber,
[Linear Transformers Are Secretly Fast Weight Programmers](https://arxiv.org/abs/2102.11174)
(2021). This repository is a small independent experiment, not a reproduction of
that paper.

Released under the [MIT License](LICENSE).
