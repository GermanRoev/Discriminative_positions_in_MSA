#!/bin/bash
#SBATCH -o stdout
#SBATCH -e stderr
#SBATCH -J roev_german
#SBATCH --get-user-env
#SBATCH --time=01:00:00
#SBATCH -N 1 
#SBATCH -n 1 
#SBATCH --mem=10000
#SBATCH -p batch
 
pip3 install -U scikit-learn
pip3 install numpy

echo 'START'
python3 search.py  --input hbv_sgene_mafft.fa --output hbv_sgene_output.txt
echo 'FINISH'