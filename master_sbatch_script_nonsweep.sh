#!/bin/bash
for i in $(seq 1 $2); do 
    echo $i
    sbatch sbatch_script_nonsweep.sh "$1"

done
