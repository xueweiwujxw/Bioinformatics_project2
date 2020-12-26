#!/bin/bash

c_time=$(date "+%Y-%m-%d")
mkdir -p output
devid=1
time src/trainTransNet2_out4.py 40 $devid $c_time >output/train-data40-$c_time"_of_TransNet2_out4".log &
