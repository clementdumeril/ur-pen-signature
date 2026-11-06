#!/bin/bash
# End-to-end tower policy with start states from the script (scripts/train_tower_curriculum.py), on Savio.
# From the repo root on Savio:  sbatch scripts/savio_tower_cur.sbatch
# Continue an earlier run (newest checkpoint, its level):  sbatch scripts/savio_tower_cur.sbatch tower_cur
# Output: /global/scratch/users/$USER/tower_cur_<job id>
#SBATCH --job-name=tower_cur
#SBATCH --account=ic_piml
#SBATCH --partition=savio3
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --time=14:00:00
#SBATCH --output=tower_cur_%j.log

set -e
cd "$SLURM_SUBMIT_DIR"
module load python 2>/dev/null || true
VENV=/global/scratch/users/$USER/pensig-venv
if [ ! -x "$VENV/bin/python" ]; then
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install -q -e . tensorboard
fi
[ -d assets/mujoco_menagerie ] || "$VENV/bin/python" scripts/fetch_assets.py

RUNS=/global/scratch/users/$USER
RESUME=()
[ -n "$1" ] && RESUME=(--resume "$RUNS/$1")
"$VENV/bin/python" -u scripts/train_tower_curriculum.py --out "$RUNS/tower_cur_$SLURM_JOB_ID" "${RESUME[@]}" \
    --timesteps 30000000 --max-minutes 750 --n-envs 32 --eval-towers 100
