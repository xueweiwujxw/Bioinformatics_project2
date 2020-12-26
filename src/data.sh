#!/bin/bash

rm -f data_test/*.sav
time src/data.py 40 2 -0.5
time src/data.py 95 2 -0.5
time src/data.py 00 2 -0.5

