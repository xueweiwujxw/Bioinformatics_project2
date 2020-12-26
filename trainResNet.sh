#!/bin/bash

c_time=$(date "+%Y-%m-%d")
mkdir -p output
devid=4
time src/trainResNet.py 40 $devid $c_time >output/train-data40-$c_time"_of_ResNet".log &
