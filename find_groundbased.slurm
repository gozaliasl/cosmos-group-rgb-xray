#!/bin/bash
#SBATCH --job-name=find_gb
#SBATCH --output=/automnt/n23data2/gozaliasl/cosmos-group-rgb-xray/logs/find_gb_%j.log
#SBATCH --ntasks=1 --cpus-per-task=1 --mem=1G --time=00:10:00

OUT=/automnt/n23data2/gozaliasl/cosmos-group-rgb-xray/logs/groundbased_mosaics.txt

echo "=== /n17data HSC / Subaru ===" > $OUT
find /n17data -maxdepth 5 -name "*.fits" 2>/dev/null | grep -iE "HSC|subaru|suprime" | grep -iv "weight|wht|err|cat|psf|seg" >> $OUT

echo "" >> $OUT
echo "=== /n17data UltraVista ===" >> $OUT
find /n17data -maxdepth 5 -name "*.fits" 2>/dev/null | grep -iE "uvista|ultravist" | grep -iv "weight|wht|err|cat|psf" | head -20 >> $OUT

echo "" >> $OUT
echo "=== /n23data2 ground-based ===" >> $OUT
find /automnt/n23data2 -maxdepth 5 -name "*.fits" 2>/dev/null | grep -iE "HSC|subaru|suprime|uvista|ultravist" | grep -iv "weight|wht|err|cat|psf|seg" | head -30 >> $OUT

echo "DONE" >> $OUT
