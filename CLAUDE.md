# CLAUDE.md

Ambient light exposure and rest–activity rhythms in people with epilepsy, using NHANES
accelerometry and its ambient light channel.

## Document precedence

Read in this order. Later sources never override earlier ones.

1. **`doc/methods.md`** — the specification. What the study does. If code disagrees with
   it, the code is wrong.
2. **`doc/implementation-status.md`** — the single to-do list, and the spec ↔ code gap.
   Check here before concluding something is unimplemented or broken; it is probably
   already known.
3. **`doc/data-sources.md`** — descriptive reference for NHANES tables and variables.
   Records what the code *does*, so it is not a spec and may lag it.
4. **`doc/analysis-log.md`** — append-only history. Never edit past entries; add a
   superseding one.
5. **`doc/archive/**`** — **never current.** Superseded documents, kept for provenance.
   Do not act on anything in here, and do not cite it as the study design.

Two to-do lists is the failure mode this structure exists to prevent. New tasks go in
`implementation-status.md`, not into a new plan file.

## Standing decisions

These are settled and expensive to get wrong. Do not re-derive them from the code, which
in several cases still reflects the superseded design.

- **Cycle H (2013–2014) only for the primary analysis.** `RXQ_RX_G` carries no
  reason-for-use variables, so the case definition cannot be applied to cycle G at all.
- **Case definition is code-first:** select on ICD-10 `G40` in `RXDRSC1–3`, then confirm
  the drug is an ASM. The 12-name ASM list in `cohort.py` is the **broad** definition — it
  has a PPV of 38.9% against G40 and is for cycle G replication and sensitivity analysis
  only. Never present it as the primary definition.
- **Cycle G is a labelled broad-definition replication cohort**, not part of the primary
  analysis.
- **Python for metrics, R for statistics**, with a participant-level CSV at the boundary.
  Metric computation, masking and wear detection stay in `src/ambient_light_epilepsy`;
  matching, survey design and circular statistics are R (`MatchIt`, `cobalt`, `survey`,
  `circular`). Do not port tested metric code to R.
- **Analysis parameters come from `analysis_params.toml`**, never from prose in a
  document and never from a literal at a call site. Three different night windows once
  existed across two documents and one call site; that is what this rule prevents.
- **Do not cite the 192-case results in `results/` as current findings.** That cohort is
  roughly 60% off-label ASM use. Superseded, not merely exploratory.

## Running things

Use `conda run` with the **`ambient-light-epilepsy`** environment — that is where the
package is installed; `base` has neither it nor pytest configured. Direct `python.exe`
invocations die silently on numpy linalg in this environment.

```bash
conda run -n ambient-light-epilepsy python -m pytest tests -q
```

Run pytest from the repository root and name `tests` explicitly, or it collects nothing.
`conda run` cannot take a `-c` argument containing newlines — write a file instead.

Data paths resolve through `config.toml` profiles (`hpc`, `w_drive`, `local`) and
`src/ambient_light_epilepsy/paths.py`. Never hard-code a path; there are three machines
and two directory layouts. Raw NHANES data is never committed.

## Conventions

- Analysis logic lives in `src/` and is tested; notebooks are numbered and exploratory.
- Test metric code against synthetic signals with hand-derived answers before changing it.
  A wrong threshold produces plausible numbers, not an error.
- Every run writes a `.provenance.json` sidecar (git commit, timestamp, parameters,
  versions). Results go to dated directories; do not overwrite a previous run.
- Add an `analysis-log.md` entry for anything whose result you would want to explain later.

## Working practices

Obligations for whoever is doing the work, human or model. The failure mode in this
project is a plausible wrong number reaching a manuscript, not a crash.

- **Work one `implementation-status.md` item per session**, and finish with a commit, a
  log entry, and a ticked box. Keep diffs small enough to be read; if a change is too
  large to review, split it.
- **Never change a spec parameter, threshold or endpoint after outcomes have been seen.**
  If asked to, say plainly that it would breach pre-specification and offer to record it
  as a labelled post hoc analysis instead. Do not quietly accommodate it.
- **Before implementing a new or changed metric, supply a synthetic input whose expected
  answer is derivable by hand**, and get agreement on it. If no such example can be
  written, the definition is not yet clear enough to implement.
- **For any number destined for the manuscript, give an independent second route to it.**
  One computation is not a result.
- **Prove equivalence before refactoring.** Promoting code out of a notebook must
  reproduce the prior output participant-for-participant first; only then change behaviour.
- **Verify premises against the data before asserting them.** Both methods documents once
  asserted an ascertainment approach that cycle G cannot support, because nobody opened
  the file. Fluent prose about the data is not evidence about the data.
- **Prefer plain code over clever code**, and write it to be maintainable by someone who
  did not write it. If an explanation requires understanding the implementation, the
  implementation is wrong for this context. Always explain code with enough comments.
- **Do not touch `results/`, and do not rerun anything that overwrites a previous run,
  without being asked explicitly.**
- **Scientific judgment belongs to the researcher.** For choices of estimand, endpoints,
  covariate roles or scope, present options and consequences rather than deciding; record
  the decision and its reasoning in `analysis-log.md` as theirs.
