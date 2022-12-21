#!/bin/bash
for i in $(seq 1 $2); do 
    echo $i
#    sbatch --dependency=afterany:13344733 sbatch_script_nonsweep.sh "$1" 
    sbatch singularity_sbatch_script_nonsweep.sh "$1"
done
