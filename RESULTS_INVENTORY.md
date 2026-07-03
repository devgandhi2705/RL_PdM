# Results Inventory — pdm_pronostia

Every file currently in `results/`, matched against `CODEBASE_MAP.md`. Assignment
method: checkpoint/output filename cross-referenced against each script's
"Checkpoints Produced" / "Result Files Produced" columns; the `qrdqn_ep*.pth`
group was confirmed by the user (see Archive section) and traced by code to
`qrdqn_agent.py`'s legacy `train_qrdqn()` helper.

## Unassigned / Needs Review

None. The only ambiguous group (`qrdqn_ep500.pth`...`qrdqn_ep8000.pth`) was
resolved by the user — see **Archive: Pre-Fix Broken Run** below.

---

## ★ Primary Paper Result (CVaR QR-DQN, 5D state)

`src/train.py` + `src/evaluate.py`

- `results/qrdqn_best.pth`
- `results/final_comparison.csv`
- `results/fig1_rul_prediction.png`
- `results/fig2_health_index.png`
- `results/fig3_policy_comparison.png`
- `results/fig4_action_composition.png`
- `results/table1_rul.tex`
- `results/table1_rul.csv`
- `results/table2_policy.tex`
- `results/table2_policy.csv`

(10 files. Note: `train.py`'s own `training_log.csv` / `training_curve.png` and
`evaluate.py`'s `fig5_training_curve.png` are not present on disk — nothing to
inventory for those.)

## RUL Predictor Training

`src/train_rul.py`

- `results/rul_model_best.pth`

(1 file. `rul_training_curve.png`, `rul_val_check.png`, `rul_val_metrics.txt` are
not present on disk.)

## Data Preprocessing

`src/feature_extractor.py`

- `results/scaler.pkl`
- `results/hi_verification.png`

(2 files.)

## RUL Baselines (XGBoost / GRU / TCN)

`src/rul_baselines.py`

- `results/table_rul_baselines.csv`
- `results/table_rul_baselines.tex`
- `results/fig_rul_baselines.png`

(3 files.)

## Uncertainty Validation (MC Dropout / Deep Ensemble)

`src/uncertainty_validation.py`

- `results/ensemble_42.pth`
- `results/ensemble_123.pth`
- `results/ensemble_456.pth`
- `results/ensemble_789.pth`
- `results/ensemble_1024.pth`
- `results/fig_reliability.png`
- `results/fig_uncertainty_ci.png`
- `results/table_uncertainty.csv`
- `results/table_uncertainty.tex`

(9 files.)

## RL Benchmarks (DDQN / Dueling DQN / PPO)

`src/rl_benchmarks.py`

- `results/ddqn_best.pth`
- `results/dueling_dqn_best.pth`
- `results/ppo_best.pth`
- `results/table_rl_benchmarks.csv`
- `results/table_rl_benchmarks.tex`
- `results/fig_rl_benchmarks.png`

(6 files.)

## State Representation Ablation (A/B/C/D)

`src/state_ablation.py`

- `results/ablation_stateA.pth`
- `results/ablation_stateB.pth`
- `results/ablation_stateC.pth`
- `results/ablation_stateC_rmmean.pth` (risk_mode="mean" variant of State C)
- `results/table_state_ablation.csv`
- `results/table_state_ablation.tex`
- `results/fig_state_ablation.png`

(7 files.)

## CVaR Alpha Risk Analysis

`src/risk_analysis.py`

- `results/cvar_alpha_0.05.pth`
- `results/cvar_alpha_0.10.pth`
- `results/table_risk_analysis.csv`
- `results/table_risk_analysis.tex`
- `results/fig_risk_return.png`

(5 files. Note: checkpoints for alpha 0.25/0.40/0.60/0.80/1.00 are absent from
disk even though the table/figure summarize all 7 — those 5 checkpoints were
apparently not kept. Nothing to move for files that don't exist.)

## Repair Model Ablation

`src/repair_ablation.py`

- `results/table_repair_ablation.csv`
- `results/table_repair_ablation.tex`
- `results/fig_repair_ablation.png`

(3 files. Note: `repair_perfect.pth` / `repair_decay.pth` checkpoints are absent
from disk.)

## Final Figures / Master Aggregation

`src/final_figures.py`

- `results/fig_master.png`
- `results/fig_supp_uncertainty.png`
- `results/fig_supp_repair.png`
- `results/fig_supp_training.png`
- `results/table_master.tex`

(5 files.)

## Experimental Audit (Parts A–C)

`src/experimental_audit.py`

- `results/table_audit_partA.csv`
- `results/table_audit_partA.tex`
- `results/fig_audit_partA.png`
- `results/audit_seed42_dueling.pth`
- `results/audit_seed42_qrdqn.pth`
- `results/audit_seed123_qrdqn.pth`
- `results/audit_seed123_dueling.pth`
- `results/table_seed_sweep.csv`
- `results/fig_seed_boxplots.png`
- `results/_part_b_raw.json`
- `results/table_significance.csv`

(11 files.)

## Explainability (Parts D–E)

`src/explainability.py`

- `results/table_cfp_marginal.csv`
- `results/fig_cfp_marginal.png`
- `results/fig_explainability.png`

(3 files.)

## Dueling Distributional QR-DQN (D3QN) — Negative Result

`src/train_d3qn.py` + `src/evaluate_d3qn.py`

- `results/d3qn_cvar_final.pth`
- `results/d3qn_cvar_best.pth`
- `results/d3qn_diag_alpha060.pth`
- `results/d3qn_diag_rewardfix.pth`
- `results/d3qn_diag_rewardfix_final.pth`
- `results/d3qn_diag_meantest.pth`
- `results/d3qn_diag_seed123.pth`
- `results/d3qn_diag_seed123_final.pth`
- `results/d3qn_training_log.csv`
- `results/fig_d3qn_learning_curve.png`
- `results/fig_d3qn_risk_return.png`
- `results/fig_d3qn_comparison.png`
- `results/table_d3qn_significance.csv`
- `results/table_d3qn_significance_footnote.txt`

(14 files.)

## Archive: Pre-Fix Broken Run

**User-confirmed**: these 16 checkpoints are from the first training run,
before the scaler fix / HI redesign / reward restructuring in EXPERIMENTS.md
Phase 1. Trained on the corrupted-scaler, `lambda_hold=0.02` version of the
environment — predates the current `PdMBearingEnv` reward function entirely.
Reflects a collapsed do-nothing policy. Not used in any reported result.

Destination: `_archive_unassigned/pre_fix_broken_run/` (with a `README.md`
explaining the above, to be written in Step 4).

- `results/qrdqn_ep500.pth`
- `results/qrdqn_ep1000.pth`
- `results/qrdqn_ep1500.pth`
- `results/qrdqn_ep2000.pth`
- `results/qrdqn_ep2500.pth`
- `results/qrdqn_ep3000.pth`
- `results/qrdqn_ep3500.pth`
- `results/qrdqn_ep4000.pth`
- `results/qrdqn_ep4500.pth`
- `results/qrdqn_ep5000.pth`
- `results/qrdqn_ep5500.pth`
- `results/qrdqn_ep6000.pth`
- `results/qrdqn_ep6500.pth`
- `results/qrdqn_ep7000.pth`
- `results/qrdqn_ep7500.pth`
- `results/qrdqn_ep8000.pth`

(16 files.)

## Shared / Non-Experiment-Specific

- `results/.gitkeep` — git placeholder, not tied to any experiment.

(1 file.)

---

## Summary

**96 files total, 96 assigned, 0 unassigned.**

| Section | Count |
|---|---|
| ★ Primary Paper Result | 10 |
| RUL Predictor Training | 1 |
| Data Preprocessing | 2 |
| RUL Baselines | 3 |
| Uncertainty Validation | 9 |
| RL Benchmarks | 6 |
| State Ablation | 7 |
| CVaR Alpha Risk Analysis | 5 |
| Repair Model Ablation | 3 |
| Final Figures / Master | 5 |
| Experimental Audit (A–C) | 11 |
| Explainability (D–E) | 3 |
| D3QN — Negative Result | 14 |
| Archive: Pre-Fix Broken Run | 16 |
| Shared / Non-Experiment | 1 |
| **Total** | **96** |
