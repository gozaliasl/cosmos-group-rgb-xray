#!/bin/bash
#SBATCH --job-name=cosmos_xray
#SBATCH --output=/automnt/n23data2/gozaliasl/cosmos-group-rgb-xray/logs/xray_batch_%j.log
#SBATCH --error=/automnt/n23data2/gozaliasl/cosmos-group-rgb-xray/logs/xray_batch_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

REPO=/automnt/n23data2/gozaliasl/cosmos-group-rgb-xray

module load intelpython/3-2024.2.0

mkdir -p $REPO/logs

cd $REPO
pip install -q -e . --no-deps 2>/dev/null

python $REPO/scripts/batch_xray.py both
