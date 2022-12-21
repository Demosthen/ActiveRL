#!/bin/bash
for i in $(seq 1 $2); do 
    echo $i
    sbatch singularity_sbatch_script.sh $1

done
