#!/usr/bin/env bash

source=$1
target=$2
arch=$3

if [ $# -ne 3 ]
 then
   echo "Arguments error: <source> <target> <arch>"
   exit 1
fi


python3 examples/train_baseline.py -ds ${source} -dt ${target} -a ${arch} \
--logs-dir logs/${arch}_strong_baseline/${source}-TO-${target} --use-xbm

