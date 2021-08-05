#!/usr/bin/env bash

dataset=$1
arch=$2
stage=$3
resume=$4

if [ $# -ne 4 ]
 then
   echo "Arguments error: <dataset> <arch> <stage> <resume>"
   exit 1
fi

python3 examples/test.py -d ${dataset} -a ${arch} --stage ${stage} --resume ${resume} --dsbn-idm


