#!/bin/bash
#SBATCH --account=ufdatastudios
#SBATCH --job-name=coco-panels
#SBATCH --output=/orange/ufdatastudios/c.okocha/AI-Jobs-Research/batch_scripts/coco_panels_%j.out
#SBATCH --error=/orange/ufdatastudios/c.okocha/AI-Jobs-Research/batch_scripts/coco_panels_%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=08:00:00
# On HPG you often need a GPU partition for long jobs; request 1 GPU but script is CPU-only
#SBATCH --gpus=1
#SBATCH --partition=hpg-b200
#SBATCH --mail-user=c.okocha@ufl.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

BASE_DIR="/orange/ufdatastudios/c.okocha/AI-Jobs-Research"
OUT_DIR="${BASE_DIR}/runs/coco_panels"

echo "=== Env bootstrap (uv-managed) ==="
module purge >/dev/null 2>&1 || true
export CUDA_HOME=/apps/compilers/cuda/12.8.1
export PATH="${CUDA_HOME}/bin:${PATH}"

echo "=== Installing dependencies (if missing) ==="
# Use uv-managed environment to install runtime deps into the project venv
if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: 'uv' not found on PATH. Please module load or install uv, then re-run."
  exit 1
fi
uv pip install -q "fiftyone>=0.25,<0.26" "opencv-python-headless>=4.10" "numpy>=1.24" "Pillow"

# Keep FiftyOne data/cache off $HOME for HPC friendliness
export FIFTYONE_HOME="${BASE_DIR}/.fiftyone"
export FIFTYONE_DATABASE_DIR="${FIFTYONE_HOME}/db"
export FIFTYONE_DATASET_ZOO_DIR="${FIFTYONE_HOME}/zoo"
mkdir -p "$FIFTYONE_HOME" "$FIFTYONE_DATABASE_DIR" "$FIFTYONE_DATASET_ZOO_DIR"

echo "=== Running cocogenerate.py (via uv) ==="
mkdir -p "${OUT_DIR}"
cd "${BASE_DIR}"

uv run python "${BASE_DIR}/cocogenerate.py" \
  --output_dir "${OUT_DIR}" \
  --num_panels 200 \
  --split validation \
  --max_source_images 2000 \
  --tile_size 512 \
  --transform_policy random

echo "=== Done ==="
echo "Panels saved under: ${OUT_DIR}"
echo " - Images:      ${OUT_DIR}/images/"
echo " - Annotations: ${OUT_DIR}/annotations/"


