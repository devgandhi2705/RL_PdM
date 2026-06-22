# PdM Project — Experiment Log

## Phase 1 — Core Pipeline (commit: f190847)

- **Data:** PRONOSTIA dataset, 6 bearings (3 conditions × 2 runs), 32-D vibration features, Health Index (HI), RUL labels extracted by `src/feature_extractor.py`
- **RUL model:** Conv-SA (multi-scale Conv1d + Transformer), MC Dropout epistemic uncertainty
  - Full-series RMSE: 21.78 | Late-series RMSE: 27.99 | ECE: 0.127
- **RL environment:** PdMBearingEnv — 5D state [HI, slope, RUL_norm, n_repairs/5, steps/100], 3 actions (do-nothing / repair / replace), termination at HI < 0.055
- **Agent:** CVaR QR-DQN, α=0.40, N_quantiles=51
- **Result (canonical, 300 eval episodes):**
  - CVaR QR-DQN: cost=252.3, catastrophe=9.0% (README table)
  - Risk-Neutral QR-DQN: cost=185.0, catastrophe=12.0%
  - CVaR beats Risk-Neutral by ~3 pp catastrophe rate at cost of higher maintenance spend
  - Seed sensitivity observed: catastrophe ranges 7–17% depending on training seed

---

## Phase 2 — Extended Baselines (commit: 18f0756)

- **RUL baselines** (`src/rul_baselines.py`): XGBoost, GRU, TCN, CNN, LSTM vs Conv-SA
  - Results in `results/table_rul_baselines.csv` / `.tex`
- **Uncertainty calibration** (`src/uncertainty_validation.py`): MC Dropout vs Deep Ensemble vs Deterministic
  - Results in `results/table_uncertainty.csv`
  - Deep Ensemble best overall calibration; MC Dropout best ECE
- **RL benchmarks** (`src/rl_benchmarks.py`): DDQN, Dueling DQN, PPO vs QR-DQN
  - Results in `results/table_rl_benchmarks.csv`
  - **Key finding:** Dueling DQN and PPO achieve lower cost (58–64) and lower catastrophe (2–2.7%) than CVaR QR-DQN (252, 12.3%) in single-seed eval; however both lack uncertainty-awareness and CVaR risk interpretability
  - Audit note: CVaR QR-DQN audit numbers higher than README — seed-dependent checkpoint quality
- **State ablation** (`src/state_ablation.py`): 4 configurations, 300 eval eps each
  - A [RUL only]:         catastrophe=17.3%
  - B [RUL + variance]:   catastrophe=0.3%  ← variance is the safety gain
  - C [RUL + var + CFP]:  catastrophe=0.3%, cost-efficient  ← CFP buys cost-efficiency
  - D [full 5D HI state]: catastrophe=8.3%  ← full env obs includes HI drift noise
- **CVaR α sweep** (`src/risk_analysis.py`): α ∈ [0.05, 0.10, 0.25, 0.40, 0.60, 0.80, 1.00]
  - Results in `results/table_risk_analysis.csv`
  - α=0.40 chosen as best cost–catastrophe tradeoff
- **Repair ablation** (`src/repair_ablation.py`): PerfectRepairEnv (efficacy=0.40 constant) vs exponential decay (max(0.1, 0.35·exp(−0.4·n_repairs)))
  - Results in `results/table_repair_ablation.csv`

---

## Phase 3 — Statistical Robustness Audit (Parts A–E)

- **Part A** — Consolidated audit table for all 6 agents on existing checkpoints
  - Saved: `results/table_audit_partA.csv`, `results/fig_audit_partA.png`
- **Part B** — 2-seed sweep (seeds 42, 123; Risk-Neutral QR-DQN + Dueling DQN + CVaR QR-DQN; 2000 eps each)
  - seed 42 checkpoint collapsed to 100% do-nothing for both QR-DQN variants (undertrained)
  - Saved: `results/table_seed_sweep.csv`, `results/fig_seed_boxplots.png`
- **Part C** — Significance tests (t-test + Mann-Whitney U)
  - Comparisons: Risk-Neutral vs CVaR; Dueling DQN vs CVaR
  - **Limitation:** n=2 seeds → minimum achievable p-value ≈ 0.33; reported as illustrative effect sizes only, not statistically significant
  - Saved: `results/table_significance.csv`
- **Part D** — CFP marginal contribution (State B→C delta)
  - Saved: `results/table_cfp_marginal.csv`, `results/fig_cfp_marginal.png`
- **Part E** — Decision explainability: State-C agent rollout, [RUL, Variance, CFP, Action] trace
  - Saved: `results/fig_explainability.png`

---

## Phase 4 — Dueling Distributional QR-DQN (in progress)

- **Goal:** Combine uncertainty-aware state (from Phase 2 State C) + dueling network architecture (from Phase 2 RL benchmarks) + distributional RL + CVaR in a single agent
- **Hypothesis:** Dueling architecture should improve advantage estimation; combined with CVaR risk-shaping over quantile distribution, this should beat all prior agents on the cost–catastrophe Pareto frontier
- **Architecture plan:** QR-DQN backbone with separate value/advantage streams before quantile head; state input = State C (3D: [RUL_norm, sigma_norm, CFP]) or full 5D
- **Status:** Logging + git infrastructure set up (Phase 4 commit). Implementation pending.

---

*Updated: Phase 4 setup (logging, gitignore, experiment log added)*
