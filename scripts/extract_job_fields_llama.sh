#!/bin/bash
#SBATCH --account ufdatastudios
#SBATCH --job-name extract-job-fields-llama
#SBATCH --output=batch_scripts/extract_job_fields_llama_%j.out
#SBATCH --error=batch_scripts/extract_job_fields_llama_%j.err
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
CSV_PATH="${BASE_DIR}/Data/extracted_job_postings.csv"
OUTPUT_PATH="${BASE_DIR}/Data/extracted_job_fields_llama.csv"

# Use /orange for model caches to avoid home quota
export HF_HOME="${BASE_DIR}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${BASE_DIR}/.cache/transformers"
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}"

# Copy Hugging Face token to accessible location for gated models
# The token from home directory needs to be available in the SLURM job
if [ -f "${HOME}/.cache/huggingface/token" ]; then
    cp "${HOME}/.cache/huggingface/token" "${HF_HOME}/token" 2>/dev/null || true
    chmod 600 "${HF_HOME}/token" 2>/dev/null || true
    echo "Hugging Face token copied to ${HF_HOME}/token"
elif [ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
    echo "${HUGGING_FACE_HUB_TOKEN}" > "${HF_HOME}/token"
    chmod 600 "${HF_HOME}/token" 2>/dev/null || true
    echo "Hugging Face token set from environment variable"
else
    echo "WARNING: No Hugging Face token found. Gated models may not work."
    echo "Please ensure you have:"
    echo "  1. Access to the Llama model at https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct"
    echo "  2. Run 'huggingface-cli login' to authenticate"
fi

# Performance knobs
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Optional: set HF token for gated models via environment variable
# Uncomment and set if token is stored as environment variable:
# export HUGGING_FACE_HUB_TOKEN="your_token_here"

echo "===== Starting Llama Job Fields Extraction ====="
echo "CSV Path: ${CSV_PATH}"
echo "Output Path: ${OUTPUT_PATH}"

# Change to base directory
cd "${BASE_DIR}"

# Activate virtual environment
source .venv/bin/activate

# Run Llama job fields extraction
python models/extract_job_fields_llama.py \
  --csv_path "${CSV_PATH}" \
  --job_posting_column job_posting \
  --output_path "${OUTPUT_PATH}" \
  --model_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
  --max_new_tokens 1500 \
  --temperature 0.1

echo "===== Llama Job Fields Extraction completed ====="

