#!/bin/bash

exec > run_model_ss.log 2>&1  # 所有输出保存到 run.log

for ssi in 0 1 2 3 4
do
  for aaj in 0 1 2 3
  do
    sed -i '' "s/^ssi = .*/ssi = $ssi/" run_model.py
    sed -i '' "s/^aaj = .*/aaj = $aaj/" run_model.py

    echo "Running with ssi = $ssi, aaj = $aaj"
    python run_model.py
  done
done
