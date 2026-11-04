#!/bin/bash
# Tower with a learned world model and CEM planning (no policy), random block sizes, on Savio.
# From the repo root on Savio:  sbatch scripts/savio_tower_wm.sbatch
#SBATCH --job-name=tower_wm
#SBATCH --account=ic_piml
#SBATCH --partition=savio3
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --output=tower_wm_%j.log

set -e
cd "$SLURM_SUBMIT_DIR"
module load python 2>/dev/null || true
VENV=/global/scratch/users/$USER/pensig-venv
if [ ! -x "$VENV/bin/python" ]; then
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install -q -e . tensorboard
fi
[ -d assets/mujoco_menagerie ] || "$VENV/bin/python" scripts/fetch_assets.py

"$VENV/bin/python" -u scripts/tower_world_model.py --out /global/scratch/users/$USER/tower_wm --workers 32 \
    --towers 3000 --rounds 3 --round-towers 128 --fit-steps 30000 --eval-towers 100
