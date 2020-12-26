#!/bin/bash

c_time=$(date "+%Y-%m-%d")
mkdir -p output
devid=7
time src/trainTransNet2_test.py 00 $devid $c_time >output/train-data00-$c_time"_of_TransNet2_test".log &
