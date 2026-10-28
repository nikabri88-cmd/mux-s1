# Brief for the independent reviewer of MUX-S1

You have not seen this work being produced. Your job is to check it, not to improve it.

Project folder: /home/claude/mux-s1 (Python; run with `cd /home/claude/mux-s1 && python3 ...`).
The paper being modelled: digitize/hesse2026.pdf (Hesse et al. 2026, bioRxiv).

Read, in this order:
1. MUX-S1_prereg.md (pre-registration, Russian) and MUX-S1_prereg.sha256 / MUX-S1_prereg_manifest.txt
2. MUX-S1_deviations.md
3. mux_s1/core.py, mux_s1/experiments.py, mux_s1/analysis.py, mux_s1/runner.py, calibrate.py, run_parts.py, analyze.py,
   and the post hoc exploratory scripts explore_snr.py, analyze_explore.py, explore_accuracy.py (hashes in MUX-S1_analysis.sha256)
4. results/calibration.json, results/calibration_log.json, results/summary.json, results/partI/*.json, results/partII/*.json,
   results/explore_snr/*.json, results/explore_snr_summary.json, results/explore_accuracy.json
5. MUX-S1_report.md (the report, Russian) and figures/*.png

Check and report concretely (file:line, numbers):
- Does the code implement what the pre-registration says (models, designs, calibration, pipeline,
  pass criteria, hypothesis thresholds)? List every mismatch.
- Is the reimplementation of the authors' pipeline faithful to their Methods (pages 26-31 of the PDF)?
  Note any analysis choice that could change a verdict.
- Recompute each hypothesis verdict H1-H7 from results/partI and results/partII yourself (do not trust
  analyze.py) and compare with the report.
- Check every number quoted in the report against results/summary.json or the per-dataset JSON files.
- Are any conclusions in the report stronger than the evidence (e.g. claims about the real data that the
  simulation cannot support)? Quote the sentence and say what would be accurate.
- Are the figures correct and readable (labels, units, decimal commas, no overlaps)?
- Anything else that would embarrass the authors if a referee found it.

Be specific and terse. Rank findings by severity: (1) changes a verdict or a headline claim, (2) wrong
number or misleading wording, (3) cosmetic.
