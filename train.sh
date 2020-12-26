#!/bin/bash

c_time=$(date "+%Y-%m-%d")
mkdir -p output
devid=0
time src/train.py 40 $devid $c_time >output/train-data40-$c_time"_of_LinearNet".log &
