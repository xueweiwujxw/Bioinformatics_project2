#!/bin/bash

c_time=$(date "+%Y-%m-%d")
mkdir -p output
devid=5
time src/trainTransNet2_test0.py 40 $devid $c_time >output/train-data40-$c_time"_of_TransNet2_test".log &
