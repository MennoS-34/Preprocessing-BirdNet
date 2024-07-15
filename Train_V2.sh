#!/bin/bash

#SBATCH -p GPU # partition (queue)
#SBATCH -N 2 # number of nodes
#SBATCH -t 0-48:00 # time (D-HH:MM)
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR
#SBATCH --mail-type=ALL
#SBATCH --mail-user=m.l.spaan@tilburguniversity.edu
#SBATCH --mem=180GB

cd /home/u993985/Thesis

source activate GPU2

python Train_V2.py