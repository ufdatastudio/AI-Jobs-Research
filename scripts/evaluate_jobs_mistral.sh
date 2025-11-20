#!/bin/bash
#SBATCH --account ufdatastudios
#SBATCH --job-name evaluate-jobs-mistral
#SBATCH --output=batch_scripts/evaluate_jobs_mistral_%j.out
#SBATCH --error=batch_scripts/evaluate_jobs_mistral_%j.err
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --time=2:00:00
#SBATCH --mem=80GB
#SBATCH --cpus-per-task=8
#SBATCH --mail-user=c.okocha@ufl.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition hpg-b200

set -euo pipefail

echo "===== GPU Info ====="
nvidia-smi || true

export CUDA_HOME=/apps/compilers/cuda/12.8.1
export PATH=$CUDA_HOME/bin:$PATH

# Paths
BASE_DIR="/orange/ufdatastudios/c.okocha/AI-Jobs-Research"
CSV_PATH="${BASE_DIR}/Data/extracted_job_fields_llama.csv"
OUTPUT_DIR="${BASE_DIR}/results/JobPostings/Mistral"

# Use /orange for model caches to avoid home quota
export HF_HOME="${BASE_DIR}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${BASE_DIR}/.cache/transformers"
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}"

# Performance knobs
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Optional: set HF token for gated models
# export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-}"

echo "===== Starting Mistral Job Posting Evaluation ====="
echo "CSV Path: ${CSV_PATH}"
echo "Output Dir: ${OUTPUT_DIR}"

# Change to base directory
cd "${BASE_DIR}"

# Activate virtual environment
source .venv/bin/activate

# Run Mistral job posting evaluation
python models/evaluate_jobs_mistral.py \
  --csv_path "${CSV_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --model_id "mistralai/Mistral-7B-Instruct-v0.3" \
  --max_new_tokens 512 \
  --temperature 0.2

echo "===== Mistral Job Posting Evaluation completed ====="

