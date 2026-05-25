# SPA — NeurIPS Research Plan & Session Log

**Project:** Safety-Preserving Alignment (SPA) via Federated Gradient Projection  
**GitHub:** https://github.com/theoriginalsam/spa-federated-llm-alignment (public)  
**JupyterHub user:** `theoriginalsam` at `jupyter-theoriginalsam`  
**Server path:** `~/spa-federated-llm-alignment/`  
**Results path:** `~/federated-ethical-llm-v2/`  

---

## What Was Built

### V1 (original course project — `ForClaude/` root)
| File | Description |
|---|---|
| `Step1_SafetySubspace.ipynb` | Fine-tune Qwen2.5-7B on HH-RLHF chosen; SVD of LoRA deltas → safety subspace |
| `Step2_FederatedTraining.ipynb` | FL with gradient projection (FedAvg, 10 clients, 10 rounds) |
| `Step3_DPOAlignment.ipynb` | DPO alignment on 500 HH-RLHF preference pairs |
| `Step4_Evaluation.ipynb` | BOLD toxicity + demographic parity evaluation |

### V2 (NeurIPS upgrade — `ForClaude/V2/`)
| File | Description |
|---|---|
| `utils.py` | ALL shared code: model loading, LoRA utils, datasets, MMLU/TruthfulQA/BOLD/refusal evaluators, DPO loss, gradient alignment metric |
| `Step1_SafetySubspace_V2.ipynb` | 5-seed fine-tuning, singular value spectrum, seed stability plots |
| `Step2_FederatedTraining_V2.ipynb` | 4 FL methods × 5 seeds, adversarial fraction sweep, non-IID Dirichlet sweep, gradient alignment tracking |
| `Step3_DPOAlignment_V2.ipynb` | DPO for all methods, β sweep {0.01,0.05,0.1,0.5}, epoch sweep {1,3,5,10} |
| `Step4_Evaluation_V2.ipynb` | MMLU 5-shot, TruthfulQA MC1, BOLD parity, adversarial refusal rate, perplexity, Pareto frontier, Wilcoxon tests |
| `Step5_SubspaceAnalysis_V2.ipynb` | Gradient alignment per round, layer-wise energy, layer-group ablation, singular spectrum, UMAP, projection energy |
| `run_all.sh` | One-command nohup runner |

---

## JupyterHub Setup

### Clone
```bash
git clone https://github.com/theoriginalsam/spa-federated-llm-alignment.git
cd spa-federated-llm-alignment/V2
```

### Install dependencies (all required pip packages)
```bash
pip install transformers datasets peft evaluate matplotlib seaborn umap-learn scipy tqdm
```

### Run everything with nohup
```bash
cd ~/spa-federated-llm-alignment/V2
nohup bash run_all.sh > ~/nohup_spa.log 2>&1 &
echo "PID: $!"
```

### Monitor
```bash
tail -f ~/nohup_spa.log          # live log
ps aux | grep run_all            # check if still running
ls ~/federated-ethical-llm-v2/results/   # check outputs
ls ~/federated-ethical-llm-v2/plots/     # check plots
```

### Push results back to GitHub
```bash
cd ~/spa-federated-llm-alignment
mkdir -p V2/plots V2/results
cp ~/federated-ethical-llm-v2/plots/*.png V2/plots/
cp ~/federated-ethical-llm-v2/results/*.pt V2/results/
git add V2/plots/ V2/results/
git commit -m "Add V2 experiment results and plots"
git push
```

---

## Hardware
- **2× NVIDIA RTX PRO 6000 Blackwell Max-Q** (95 GB VRAM each)
- CUDA 12.8, PyTorch 2.8.0+cu128, Python 3.12.11

---

## Model & Key Hyperparameters

| Parameter | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-7B` |
| LoRA rank r | 16 |
| LoRA alpha | 32 |
| LoRA target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Safety rank k | 8 (top-k singular vectors) |
| N clients | 10 |
| N rounds | 10 |
| Safety samples | 1000 chosen responses |
| Biased samples | 1000 rejected responses |
| DPO pairs | 500 |
| Max seq len | 512 |
| Batch size | 2 |
| Grad accumulation | 8 |
| LR (fine-tune) | 2e-4 |
| LR (DPO) | 5e-5 |
| DPO β | 0.05 |
| DPO epochs | 5 |
| FedProx μ | 0.01 |
| Seeds | 42, 123, 2024, 7, 99 |
| dtype | bfloat16 |

---

## Experiment Grid (V2)

### FL Methods
| Method | use_proj | mu |
|---|---|---|
| fedavg_noproj | False | 0.0 |
| fedavg_proj | True | 0.0 |
| fedprox | False | 0.01 |
| fedprox_proj | True | 0.01 |

### Sweeps
| Sweep | Values |
|---|---|
| Adversarial fraction | 0.1, 0.3, 0.5, 0.7, 0.9, 1.0 |
| Non-IID alpha | IID, 0.5, 0.1 |
| Safety rank k | 1, 2, 4, 8, 16, 32 |
| DPO β | 0.01, 0.05, 0.1, 0.5 |
| DPO epochs | 1, 3, 5, 10 |

### Evaluation Methods (Step 4)
| Key | Description |
|---|---|
| base | Qwen2.5-7B no fine-tuning |
| safety_lora | Step 1 output only |
| fedavg_noproj | Biased FL, no projection |
| fedprox | FedProx, no projection |
| fedavg_proj | FedAvg + safety projection |
| fedprox_proj | FedProx + safety projection |
| spa_fedavg | fedavg_proj + DPO (proposed) |
| spa_fedprox | fedprox_proj + DPO (proposed) |

### Evaluation Metrics
| Metric | Dataset | Direction |
|---|---|---|
| Mean toxicity | BOLD | ↓ lower better |
| Demographic parity gap | BOLD | ↓ lower better |
| Adversarial refusal rate | 20 custom prompts | ↑ higher better |
| MMLU 5-shot accuracy | cais/mmlu | ↑ higher better |
| TruthfulQA MC1 | truthful_qa | ↑ higher better |
| Perplexity | HH-RLHF held-out | ↓ lower better |

---

## Checkpointing / Idempotency
Every experiment uses `.flag` files under `~/federated-ethical-llm-v2/models/`:
- `step1_seed{N}_done.flag` — Step 1 done for seed N
- `step2_{method}_seed{N}_done.flag` — Step 2 done
- `step3_{method}_seed{N}_done.flag` — Step 3 done

Results are accumulated in `.pt` files and saved after each run. **Restarting any notebook is safe — it skips completed work.**

---

## File Layout (Server)
```
~/federated-ethical-llm-v2/
├── models/
│   ├── safety_lora_seed{42,123,2024,7,99}/     # LoRA adapters
│   ├── safety_subspace_seed{N}.pt               # SVD bases + singular values
│   ├── step2_{method}_seed{N}.pt                # FL aggregated LoRA weights
│   └── step3_{method}_seed{N}/                  # DPO-aligned LoRA
├── data/
│   ├── rejected_texts_seed{N}.pt
│   ├── chosen_texts_seed{N}.pt
│   └── dpo_pairs_seed{N}.pt
├── results/
│   ├── step1_metrics.pt
│   ├── step2_main_results.pt
│   ├── step2_adv_sweep.pt
│   ├── step2_noniid_sweep.pt
│   ├── step3_dpo_results.pt
│   ├── step3_beta_sweep.pt
│   ├── step4_full_results.pt
│   ├── step4_aggregated.pt
│   └── step5_layer_ablation.pt
└── plots/
    ├── singular_value_spectrum.png
    ├── subspace_rank_ablation_energy.png
    ├── subspace_seed_stability.png
    ├── step2_training_curves.png
    ├── step2_adv_fraction_sweep.png
    ├── step3_dpo_curves.png
    ├── step4_pareto_frontier.png
    ├── step4_main_bars.png
    ├── step4_bold_per_group.png
    ├── fig5a_gradient_alignment.png
    ├── fig5b_layer_energy.png
    ├── fig5c_layer_group_ablation.png
    ├── fig5d_singular_spectrum.png
    ├── fig5e_umap_hidden_states.png
    └── fig5f_projection_energy.png
```

---

## Paper Contribution (NeurIPS framing)

**Title:** Safety Subspace Projection for Alignment-Preserving Federated Fine-Tuning

**Core claim:** SVD of LoRA weight deltas (ΔW = lora_B @ lora_A) identifies a low-rank "safety subspace." Projecting client gradients orthogonally away from this subspace during federated training prevents adversarial clients from corrupting safety alignment, even when the majority of clients are biased.

**3 contributions:**
1. Safety subspace extraction via SVD of LoRA deltas (novel, principled, geometry-based)
2. Gradient projection FL framework preserving safety under adversarial clients
3. DPO post-alignment further reduces demographic parity gap

**Key figures for paper:**
- Fig 5A: Gradient alignment per round (projection kills it, baseline doesn't) — THE killer figure
- Fig 5D: Singular value spectrum (motivates low-rank assumption)
- Table 1: Main comparison (8 methods × 6 metrics, 5 seeds ± std)
- Fig: Pareto frontier (safety vs. utility)
- Fig: Adversarial fraction robustness curve

---

## Current Status
- [x] V1 pipeline complete and evaluated
- [x] V2 code complete and pushed to GitHub
- [x] JupyterHub cloned and running
- [ ] Step 1 V2 running (nohup, ~2h)
- [ ] Step 2 V2 (start after Step 1)
- [ ] Step 3 V2 (start after Step 2, GPU 1)
- [ ] Step 4 V2 (start after Steps 2+3)
- [ ] Step 5 V2 (start after Steps 1+2)
- [ ] Paper writing
