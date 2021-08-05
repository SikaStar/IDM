#!/usr/bin/env bash

dataset=$1
arch=$2
resume=$3


if [ $# -ne 3 ]
 then
   echo "Arguments error: <dataset> <arch> <resume>"
   exit 1
fi

python3 examples/test.py -d ${dataset} -a ${arch} --resume ${resume} --dsbn


