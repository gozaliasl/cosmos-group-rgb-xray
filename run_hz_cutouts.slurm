#!/bin/bash
#SBATCH --job-name=hz_cutouts
#SBATCH --output=/automnt/n23data2/gozaliasl/cosmos-group-rgb-xray/logs/hz_cutouts_%j.log
#SBATCH --error=/automnt/n23data2/gozaliasl/cosmos-group-rgb-xray/logs/hz_cutouts_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00

REPO=/automnt/n23data2/gozaliasl/cosmos-group-rgb-xray

module load intelpython/3-2024.2.0
mkdir -p $REPO/logs

python $REPO/scripts/make_cutouts.py \
    --jwst --hst --xray \
    --catalog $REPO/catalogs/hz_detected_cutout.csv \
    --output /automnt/n23data2/gozaliasl/groups_cutouts/hz_detected \
    --size 150

echo "ALL DONE"
