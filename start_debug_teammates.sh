#!/bin/bash
export OMP_NUM_THREADS=1

host=${1:-localhost}
port=${2:-3100}

for i in {2..5}; do
  python3 ./Run_Player.py -i $host -p $port -u $i -t StudentName -P 0 -D 1 &
done