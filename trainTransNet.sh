#!/bin/bash

c_time=$(date "+%Y-%m-%d")
mkdir -p output
devid=7
time src/trainTransNet.py 40 $devid $c_time >output/train-data40-$c_time"_of_TransNet".log &
