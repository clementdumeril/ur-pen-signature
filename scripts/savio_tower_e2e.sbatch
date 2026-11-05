#!/bin/bash
# End-to-end PPO on the tower (no script under the policy), measured blocks, on Savio.
# From the repo root on Savio:  sbatch scripts/savio_tower_e2e.sbatch
#SBATCH --job-name=tower_e2e
#SBATCH --account=ic_piml
#SBATCH --partition=savio3
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --time=06:00:00
#SBATCH --output=tower_e2e_%j.log

set -e
cd "$SLURM_SUBMIT_DIR"
module load python 2>/dev/null || true
VENV=/global/scratch/users/$USER/pensig-venv
if [ ! -x "$VENV/bin/python" ]; then
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install -q -e . tensorboard
fi
[ -d assets/mujoco_menagerie ] || "$VENV/bin/python" scripts/fetch_assets.py

OUT=/global/scratch/users/$USER/tower_e2e
"$VENV/bin/python" -u scripts/train.py --task tower --timesteps 8000000 --n-envs 32 --out "$OUT" \
    --block-sizes 0.05 0.05 0.04 0.08 0.035 0.03
"$VENV/bin/python" -u scripts/evaluate_tower.py --policy "$OUT/final_model.zip" --episodes 200 \
    --block-sizes 0.05 0.05 0.04 0.08 0.035 0.03 --out "$OUT/eval.json"
