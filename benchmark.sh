#!/bin/sh
reps=20
cores=10
for n in 5 10 15; do
    for ell in 1 2 5 10; do
        echo $n $ell
        ./benchmark.py $n $ell $reps $cores
    done
done

for n in 25 50 100; do
    for ell in 1 2 5 10; do
        echo $n $ell
        ./benchmark.py $n $ell $reps $cores --moranOnly
    done
done