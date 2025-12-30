#!/bin/bash
#SBATCH --account ufdatastudios
#SBATCH --job-name ai-jobs-gemma
#SBATCH --output=batch_scripts/ai_jobs_gemma_%j.out
#SBATCH --error=batch_scripts/ai_jobs_gemma_%j.err
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=3:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=8
#SBATCH --mail-user=c.okocha@ufl.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition hpg-b200

set -euo pipefail

BASE_DIR="/orange/ufdatastudios/c.okocha/AI-Jobs-Research"
CSV_PATH="${BASE_DIR}/Data/ai_in_education_jobs.csv"
OUTPUT_DIR="${BASE_DIR}/results/JobPostings/Gemma"
MODEL_ID="google/gemma-3-27b-it"

echo "===== GPU Info ====="
nvidia-smi || true

export CUDA_HOME=/apps/compilers/cuda/12.8.1
export PATH=$CUDA_HOME/bin:$PATH

export HF_HOME="${BASE_DIR}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${BASE_DIR}/.cache/transformers"
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}"

if [ -f "${HOME}/.cache/huggingface/token" ]; then
  cp "${HOME}/.cache/huggingface/token" "${HF_HOME}/token" 2>/dev/null || true
  chmod 600 "${HF_HOME}/token" 2>/dev/null || true
elif [ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
  echo "${HUGGING_FACE_HUB_TOKEN}" > "${HF_HOME}/token"
  chmod 600 "${HF_HOME}/token" 2>/dev/null || true
else
  echo "WARNING: No Hugging Face token found. Ensure you have accepted Gemma 3 license on Hugging Face."
fi

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "${BASE_DIR}"
source .venv/bin/activate

echo "Cleaning previous results: ${OUTPUT_DIR}"
rm -rf "${OUTPUT_DIR}"

python models/evaluate_jobs_gemma.py \
  --csv_path "${CSV_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --model_id "${MODEL_ID}" \
  --max_new_tokens 512 \
  --temperature 0.2

echo "Done (Gemma 3)."


