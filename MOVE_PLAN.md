# Move Plan — results/ reorganization

Proposed structure:

```
results/
  00_primary_cvar_qrdqn/
  01_rul_predictor/
  02_rul_baselines_xgboost_gru_tcn/
  03_uncertainty_validation/
  04_rl_benchmarks_ddqn_dueling_ppo/
  05_state_ablation/
  06_risk_analysis_cvar_alpha_sweep/
  07_repair_ablation/
  08_experimental_audit_seed_sweep/
  09_explainability/
  10_dueling_distributional_d3qn_negative_result/
  11_final_figures_master/
  _archive_unassigned/
    pre_fix_broken_run/
  _shared_data/
```

96 files total. `git mv` used for the 35 git-tracked files (png/csv/tex already
committed pre-gitignore); plain move used for untracked/ignored files
(*.pth, *.pkl, *.json, *.txt — matched by `.gitignore`, never added).

## 00_primary_cvar_qrdqn/ (10 files)

| Source | Destination |
|---|---|
| results/qrdqn_best.pth | results/00_primary_cvar_qrdqn/qrdqn_best.pth |
| results/final_comparison.csv | results/00_primary_cvar_qrdqn/final_comparison.csv |
| results/fig1_rul_prediction.png | results/00_primary_cvar_qrdqn/fig1_rul_prediction.png |
| results/fig2_health_index.png | results/00_primary_cvar_qrdqn/fig2_health_index.png |
| results/fig3_policy_comparison.png | results/00_primary_cvar_qrdqn/fig3_policy_comparison.png |
| results/fig4_action_composition.png | results/00_primary_cvar_qrdqn/fig4_action_composition.png |
| results/table1_rul.tex | results/00_primary_cvar_qrdqn/table1_rul.tex |
| results/table1_rul.csv | results/00_primary_cvar_qrdqn/table1_rul.csv |
| results/table2_policy.tex | results/00_primary_cvar_qrdqn/table2_policy.tex |
| results/table2_policy.csv | results/00_primary_cvar_qrdqn/table2_policy.csv |

## 01_rul_predictor/ (1 file)

| Source | Destination |
|---|---|
| results/rul_model_best.pth | results/01_rul_predictor/rul_model_best.pth |

## 02_rul_baselines_xgboost_gru_tcn/ (3 files)

| Source | Destination |
|---|---|
| results/table_rul_baselines.csv | results/02_rul_baselines_xgboost_gru_tcn/table_rul_baselines.csv |
| results/table_rul_baselines.tex | results/02_rul_baselines_xgboost_gru_tcn/table_rul_baselines.tex |
| results/fig_rul_baselines.png | results/02_rul_baselines_xgboost_gru_tcn/fig_rul_baselines.png |

## 03_uncertainty_validation/ (9 files)

| Source | Destination |
|---|---|
| results/ensemble_42.pth | results/03_uncertainty_validation/ensemble_42.pth |
| results/ensemble_123.pth | results/03_uncertainty_validation/ensemble_123.pth |
| results/ensemble_456.pth | results/03_uncertainty_validation/ensemble_456.pth |
| results/ensemble_789.pth | results/03_uncertainty_validation/ensemble_789.pth |
| results/ensemble_1024.pth | results/03_uncertainty_validation/ensemble_1024.pth |
| results/fig_reliability.png | results/03_uncertainty_validation/fig_reliability.png |
| results/fig_uncertainty_ci.png | results/03_uncertainty_validation/fig_uncertainty_ci.png |
| results/table_uncertainty.csv | results/03_uncertainty_validation/table_uncertainty.csv |
| results/table_uncertainty.tex | results/03_uncertainty_validation/table_uncertainty.tex |

## 04_rl_benchmarks_ddqn_dueling_ppo/ (6 files)

| Source | Destination |
|---|---|
| results/ddqn_best.pth | results/04_rl_benchmarks_ddqn_dueling_ppo/ddqn_best.pth |
| results/dueling_dqn_best.pth | results/04_rl_benchmarks_ddqn_dueling_ppo/dueling_dqn_best.pth |
| results/ppo_best.pth | results/04_rl_benchmarks_ddqn_dueling_ppo/ppo_best.pth |
| results/table_rl_benchmarks.csv | results/04_rl_benchmarks_ddqn_dueling_ppo/table_rl_benchmarks.csv |
| results/table_rl_benchmarks.tex | results/04_rl_benchmarks_ddqn_dueling_ppo/table_rl_benchmarks.tex |
| results/fig_rl_benchmarks.png | results/04_rl_benchmarks_ddqn_dueling_ppo/fig_rl_benchmarks.png |

## 05_state_ablation/ (7 files)

| Source | Destination |
|---|---|
| results/ablation_stateA.pth | results/05_state_ablation/ablation_stateA.pth |
| results/ablation_stateB.pth | results/05_state_ablation/ablation_stateB.pth |
| results/ablation_stateC.pth | results/05_state_ablation/ablation_stateC.pth |
| results/ablation_stateC_rmmean.pth | results/05_state_ablation/ablation_stateC_rmmean.pth |
| results/table_state_ablation.csv | results/05_state_ablation/table_state_ablation.csv |
| results/table_state_ablation.tex | results/05_state_ablation/table_state_ablation.tex |
| results/fig_state_ablation.png | results/05_state_ablation/fig_state_ablation.png |

## 06_risk_analysis_cvar_alpha_sweep/ (5 files)

| Source | Destination |
|---|---|
| results/cvar_alpha_0.05.pth | results/06_risk_analysis_cvar_alpha_sweep/cvar_alpha_0.05.pth |
| results/cvar_alpha_0.10.pth | results/06_risk_analysis_cvar_alpha_sweep/cvar_alpha_0.10.pth |
| results/table_risk_analysis.csv | results/06_risk_analysis_cvar_alpha_sweep/table_risk_analysis.csv |
| results/table_risk_analysis.tex | results/06_risk_analysis_cvar_alpha_sweep/table_risk_analysis.tex |
| results/fig_risk_return.png | results/06_risk_analysis_cvar_alpha_sweep/fig_risk_return.png |

## 07_repair_ablation/ (3 files)

| Source | Destination |
|---|---|
| results/table_repair_ablation.csv | results/07_repair_ablation/table_repair_ablation.csv |
| results/table_repair_ablation.tex | results/07_repair_ablation/table_repair_ablation.tex |
| results/fig_repair_ablation.png | results/07_repair_ablation/fig_repair_ablation.png |

## 08_experimental_audit_seed_sweep/ (11 files)

| Source | Destination |
|---|---|
| results/table_audit_partA.csv | results/08_experimental_audit_seed_sweep/table_audit_partA.csv |
| results/table_audit_partA.tex | results/08_experimental_audit_seed_sweep/table_audit_partA.tex |
| results/fig_audit_partA.png | results/08_experimental_audit_seed_sweep/fig_audit_partA.png |
| results/audit_seed42_dueling.pth | results/08_experimental_audit_seed_sweep/audit_seed42_dueling.pth |
| results/audit_seed42_qrdqn.pth | results/08_experimental_audit_seed_sweep/audit_seed42_qrdqn.pth |
| results/audit_seed123_qrdqn.pth | results/08_experimental_audit_seed_sweep/audit_seed123_qrdqn.pth |
| results/audit_seed123_dueling.pth | results/08_experimental_audit_seed_sweep/audit_seed123_dueling.pth |
| results/table_seed_sweep.csv | results/08_experimental_audit_seed_sweep/table_seed_sweep.csv |
| results/fig_seed_boxplots.png | results/08_experimental_audit_seed_sweep/fig_seed_boxplots.png |
| results/_part_b_raw.json | results/08_experimental_audit_seed_sweep/_part_b_raw.json |
| results/table_significance.csv | results/08_experimental_audit_seed_sweep/table_significance.csv |

## 09_explainability/ (3 files)

| Source | Destination |
|---|---|
| results/table_cfp_marginal.csv | results/09_explainability/table_cfp_marginal.csv |
| results/fig_cfp_marginal.png | results/09_explainability/fig_cfp_marginal.png |
| results/fig_explainability.png | results/09_explainability/fig_explainability.png |

## 10_dueling_distributional_d3qn_negative_result/ (14 files)

| Source | Destination |
|---|---|
| results/d3qn_cvar_final.pth | results/10_dueling_distributional_d3qn_negative_result/d3qn_cvar_final.pth |
| results/d3qn_cvar_best.pth | results/10_dueling_distributional_d3qn_negative_result/d3qn_cvar_best.pth |
| results/d3qn_diag_alpha060.pth | results/10_dueling_distributional_d3qn_negative_result/d3qn_diag_alpha060.pth |
| results/d3qn_diag_rewardfix.pth | results/10_dueling_distributional_d3qn_negative_result/d3qn_diag_rewardfix.pth |
| results/d3qn_diag_rewardfix_final.pth | results/10_dueling_distributional_d3qn_negative_result/d3qn_diag_rewardfix_final.pth |
| results/d3qn_diag_meantest.pth | results/10_dueling_distributional_d3qn_negative_result/d3qn_diag_meantest.pth |
| results/d3qn_diag_seed123.pth | results/10_dueling_distributional_d3qn_negative_result/d3qn_diag_seed123.pth |
| results/d3qn_diag_seed123_final.pth | results/10_dueling_distributional_d3qn_negative_result/d3qn_diag_seed123_final.pth |
| results/d3qn_training_log.csv | results/10_dueling_distributional_d3qn_negative_result/d3qn_training_log.csv |
| results/fig_d3qn_learning_curve.png | results/10_dueling_distributional_d3qn_negative_result/fig_d3qn_learning_curve.png |
| results/fig_d3qn_risk_return.png | results/10_dueling_distributional_d3qn_negative_result/fig_d3qn_risk_return.png |
| results/fig_d3qn_comparison.png | results/10_dueling_distributional_d3qn_negative_result/fig_d3qn_comparison.png |
| results/table_d3qn_significance.csv | results/10_dueling_distributional_d3qn_negative_result/table_d3qn_significance.csv |
| results/table_d3qn_significance_footnote.txt | results/10_dueling_distributional_d3qn_negative_result/table_d3qn_significance_footnote.txt |

## 11_final_figures_master/ (5 files)

| Source | Destination |
|---|---|
| results/fig_master.png | results/11_final_figures_master/fig_master.png |
| results/fig_supp_uncertainty.png | results/11_final_figures_master/fig_supp_uncertainty.png |
| results/fig_supp_repair.png | results/11_final_figures_master/fig_supp_repair.png |
| results/fig_supp_training.png | results/11_final_figures_master/fig_supp_training.png |
| results/table_master.tex | results/11_final_figures_master/table_master.tex |

## _archive_unassigned/pre_fix_broken_run/ (16 files + new README.md)

| Source | Destination |
|---|---|
| results/qrdqn_ep500.pth | results/_archive_unassigned/pre_fix_broken_run/qrdqn_ep500.pth |
| results/qrdqn_ep1000.pth | results/_archive_unassigned/pre_fix_broken_run/qrdqn_ep1000.pth |
| results/qrdqn_ep1500.pth | results/_archive_unassigned/pre_fix_broken_run/qrdqn_ep1500.pth |
| results/qrdqn_ep2000.pth | results/_archive_unassigned/pre_fix_broken_run/qrdqn_ep2000.pth |
| results/qrdqn_ep2500.pth | results/_archive_unassigned/pre_fix_broken_run/qrdqn_ep2500.pth |
| results/qrdqn_ep3000.pth | results/_archive_unassigned/pre_fix_broken_run/qrdqn_ep3000.pth |
| results/qrdqn_ep3500.pth | results/_archive_unassigned/pre_fix_broken_run/qrdqn_ep3500.pth |
| results/qrdqn_ep4000.pth | results/_archive_unassigned/pre_fix_broken_run/qrdqn_ep4000.pth |
| results/qrdqn_ep4500.pth | results/_archive_unassigned/pre_fix_broken_run/qrdqn_ep4500.pth |
| results/qrdqn_ep5000.pth | results/_archive_unassigned/pre_fix_broken_run/qrdqn_ep5000.pth |
| results/qrdqn_ep5500.pth | results/_archive_unassigned/pre_fix_broken_run/qrdqn_ep5500.pth |
| results/qrdqn_ep6000.pth | results/_archive_unassigned/pre_fix_broken_run/qrdqn_ep6000.pth |
| results/qrdqn_ep6500.pth | results/_archive_unassigned/pre_fix_broken_run/qrdqn_ep6500.pth |
| results/qrdqn_ep7000.pth | results/_archive_unassigned/pre_fix_broken_run/qrdqn_ep7000.pth |
| results/qrdqn_ep7500.pth | results/_archive_unassigned/pre_fix_broken_run/qrdqn_ep7500.pth |
| results/qrdqn_ep8000.pth | results/_archive_unassigned/pre_fix_broken_run/qrdqn_ep8000.pth |
| (new file) | results/_archive_unassigned/pre_fix_broken_run/README.md |

## _shared_data/ (3 files)

| Source | Destination |
|---|---|
| results/scaler.pkl | results/_shared_data/scaler.pkl |
| results/hi_verification.png | results/_shared_data/hi_verification.png |
| results/.gitkeep | results/_shared_data/.gitkeep |

---

## Verification

96 source files + 1 new README.md = 97 files after move.
Sum of section counts: 10+1+3+9+6+7+5+3+11+3+14+5+16+3 = 96 (matches inventory total).
