#!/bin/bash
#SBATCH -p batch
#SBATCH -c 50
#SBATCH --mem=180G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cwang25@albany.edu
#SBATCH -o logs/simulation.out
#SBATCH -e logs/simulation.error
#SBATCH --time=11-0:1 # The job should take 0 days, 0 hours, 1 minutes

#python /network/rit/lab/ceashpc/chunpai/PycharmProjects/CNSS-Public/exps/alter_simulation.py
python /network/rit/lab/ceashpc/chunpai/PycharmProjects/CNSS-Public/exps/exp_cnss_alter.py
