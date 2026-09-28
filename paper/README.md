# Manuscript

ICML 2026 submission. The default build is anonymous. `paper/main.pdf` is produced by:

```sh
make paper
```

That runs `python -m vfates.paper assets` and then `latexmk -pdf main.tex` in this directory. TeX Live is expected on `PATH` (`/Library/TeX/texbin` on the machine used to prepare the submission).

## What is generated

`generated/` holds macros and tables. `figures/` holds the PDFs. Both are written by `python -m vfates.paper assets` and are kept in the tree so a clean clone builds without the study runs.

`generated/numbers.tex` defines one command per reported number. Until the pooled report marks a stage complete, that stage's commands render as a red `[pending]`. `generated/abstract_results.tex` contains one sentence for each confirmatory hypothesis that passed its preregistered test, and no sentence for a hypothesis that did not. With no complete stage, the abstract file is a visible pending note.

The reader of `runs/program_report.json` is `vfates.paper.assets.read_program_report`. Key names are reconciled only there.

## After the report exists

Results sentences in `sections/` use the generated commands, so they stay grammatical while numbers are pending. An interpretive sentence that depends on the outcome is a LaTeX comment beginning with `% AFTER-RESULTS:`, with the two alternative wordings in that comment. After the report is in, pick one wording, turn it into prose, and delete the other.

## Submission checklist

- Build with `\ifcameraready` left false. The PDF must not contain the author name, an affiliation, or an acknowledgement.
- Fill the affiliation and the correspondence email in `main.tex` before flipping `\camerareadytrue`.
- Main body: 8 pages. References, the Impact Statement, and the appendix are outside that limit.
- The Impact Statement is present and unnumbered, before the bibliography.
- Post the anonymous code archive at `anonymous.4open.science` and put that address in the supplementary material. Do not use a URL that contains the author name.
- Do not add author-identifying text to the anonymous build, including in figure metadata or the checklist itself once it is copied into a camera-ready note.
