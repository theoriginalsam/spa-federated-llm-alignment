#!/bin/bash
# run_all.sh — Execute all V2 SPA experiments end-to-end.
# Run from the V2/ directory: bash run_all.sh
# Each step is idempotent (skips already-done work via flag files).

set -e
cd "$(dirname "$0")"

LOG_DIR="$HOME/federated-ethical-llm-v2/logs"
mkdir -p "$LOG_DIR"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

run_notebook() {
    local nb="$1"
    local log="$LOG_DIR/$(basename ${nb%.ipynb}).log"
    echo "[$(timestamp)] Running $nb ..."
    jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=86400 \
        --ExecutePreprocessor.kernel_name=python3 \
        "$nb" 2>&1 | tee "$log"
    echo "[$(timestamp)] Done: $nb"
}

echo "======================================================"
echo "  SPA V2 — NeurIPS Experiment Runner"
echo "  Started: $(timestamp)"
echo "======================================================"

# ── Step 1: Multi-seed safety subspace extraction ──────────────────────────────
echo ""
echo ">>> STEP 1: Safety Subspace Extraction (5 seeds)"
run_notebook "Step1_SafetySubspace_V2.ipynb"

# ── Step 2: Federated training — all methods + sweeps ──────────────────────────
echo ""
echo ">>> STEP 2: Federated Training (all methods, adversarial sweep, non-IID sweep)"
run_notebook "Step2_FederatedTraining_V2.ipynb"

# ── Step 3: DPO alignment + beta sweep ────────────────────────────────────────
echo ""
echo ">>> STEP 3: DPO Alignment (all methods, beta sweep)"
run_notebook "Step3_DPOAlignment_V2.ipynb"

# ── Step 4: Full evaluation suite ─────────────────────────────────────────────
echo ""
echo ">>> STEP 4: Evaluation (MMLU, TruthfulQA, BOLD, Refusal, Perplexity)"
run_notebook "Step4_Evaluation_V2.ipynb"

# ── Step 5: Subspace analysis / interpretability ──────────────────────────────
echo ""
echo ">>> STEP 5: Subspace Analysis (alignment, UMAP, layer ablation)"
run_notebook "Step5_SubspaceAnalysis_V2.ipynb"

echo ""
echo "======================================================"
echo "  ALL STEPS COMPLETE"
echo "  Finished: $(timestamp)"
echo "  Results : $HOME/federated-ethical-llm-v2/results/"
echo "  Plots   : $HOME/federated-ethical-llm-v2/plots/"
echo "======================================================"
