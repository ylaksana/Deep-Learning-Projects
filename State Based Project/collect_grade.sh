#!/bin/bash

for (( num=0; num<100; num+=2 )); do
     echo "imitation_jurgen_$num.jit........................................................"
     cp state_agent/imitation_jurgen_$num.jit state_agent/imitation_jurgen_final.jit
     for (( epoch=0; epoch<1; epoch++ )); do
       python -m grader state_agent -v
     done

done