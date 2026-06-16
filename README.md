# Risk-Averse Predictive Maintenance with Uncertainty-Aware RL

Reinforcement-learning-based predictive maintenance scheduling for rolling-element
bearings, using the [PRONOSTIA / FEMTO-ST](https://www.femto-st.fr/en) bearing
degradation dataset. A Conv-SA (multi-scale Conv1d + Transformer) model predicts
Remaining Useful Life (RUL) with epistemic uncertainty (MC Dropout / Deep Ensemble),
and a CVaR-constrained Quantile Regression DQN (QR-DQN) agent learns a maintenance
policy (do-nothing / repair / replace) that explicitly trades off cost against the
risk of catastrophic in-service failure.

## Key results

| Policy                  | Mean Cost | Catastrophe Rate |
|--------------------------|----------:|------------------:|
| Corrective Maintenance (baseline) | 317.1 | 17.3% |
| Risk-Neutral DQN          | 185.0 | 12.0% |
| **CVaR QR-DQN (ours, α=0.40)** | 252.3 | **9.0%** |

Full numbers and ablations (RUL baselines, uncertainty calibration, state-space
ablation, CVaR risk sweep, repair-model ablation) are in `results/*.csv` /
`results/*.tex`, and the consolidated figure is `results/fig_master.png`.

## Repository structure

```
pdm_pronostia/
├── config.yaml              # all hyperparameters (data, env, QR-DQN, RL training)
├── run_all.py                # master pipeline: feature extraction -> RUL -> RL -> eval
├── requirements.txt
├── data/
│   ├── raw/                  # PRONOSTIA CSVs (Bearing{cond}_{idx}/acc_*.csv) — not tracked
│   └── processed/             # precomputed features/HI/RUL .npy — not tracked, regenerated
├── results/                  # figures, tables, metrics (checkpoints not tracked)
└── src/
    ├── feature_extractor.py   # vibration -> 32-dim feature vectors + Health Index
    ├── rul_predictor.py        # Conv-SA model + MC Dropout inference
    ├── train_rul.py             # RUL model training
    ├── rul_baselines.py          # XGBoost / GRU / TCN / CNN / LSTM baselines
    ├── uncertainty_validation.py # MC Dropout vs Deep Ensemble calibration (PICP/MPIW/ECE)
    ├── rl_environment.py         # PdMBearingEnv (Gymnasium) — 5D state, 3 actions
    ├── qrdqn_agent.py             # CVaR Quantile-Regression DQN agent
    ├── train.py                   # RL agent training + 4-policy comparison
    ├── baselines.py                # CorrectiveMaintenance / PeriodicPM / Threshold policies
    ├── rl_benchmarks.py             # DDQN / Dueling DQN / PPO benchmark agents
    ├── state_ablation.py             # state-space ablation (RUL-only -> full 5D state)
    ├── risk_analysis.py               # CVaR alpha sweep, risk-return tradeoff
    ├── repair_ablation.py              # constant- vs decaying-efficacy repair model
    ├── evaluate.py                      # figure/table generation for the core pipeline
    └── final_figures.py                  # publication figure set (fig_master + supplementary)
```

## Setup

```bash
git clone <this-repo-url>
cd pdm_pronostia
python -m venv venv && source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Download the [PRONOSTIA bearing dataset](https://www.femto-st.fr/en/research-departments/automatic-control-and-micro-mechatronic-systems/research-fields/data-acquisition-experimentation/femto-bearing-data-set)
and place each bearing folder (`Bearing1_1`, `Bearing1_2`, ... `Bearing3_2`) under
`data/raw/`, matching the layout described in `config.yaml`.

## Usage

Run the full pipeline (feature extraction → RUL training → RL training → evaluation):

```bash
python run_all.py                 # full run
python run_all.py --dry-run       # fast smoke test
python run_all.py --skip-rul      # reuse existing RUL checkpoint if valid
python run_all.py --steps 4 5     # run only specific steps
```

Or run individual analysis phases directly:

```bash
python -m src.uncertainty_validation   # MC Dropout vs Deep Ensemble calibration
python -m src.rl_benchmarks             # DDQN / Dueling DQN / PPO vs QR-DQN
python -m src.state_ablation             # state-space ablation
python -m src.risk_analysis               # CVaR alpha sweep
python -m src.repair_ablation              # repair-model ablation
python -m src.final_figures                 # consolidated paper figures + master table
```

## License

MIT — see [LICENSE](LICENSE).
