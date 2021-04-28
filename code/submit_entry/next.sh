#!/bin/bash -e

# This script will be run to evaluate a Challenge record; the name of
# the record will be passed on the command line (as $1).  Your program
# should decide whether the record depicts a true or false alarm, and
# add its response to the file 'answers.txt'.

RECORD=$1
if [ ! -d "./miniconda3" ];then
  sh setup.sh
  else
  echo "setup have finished"
fi

#./miniconda3/bin/conda install --use-local -y ./conda_pkgs/pytorch-1.2.0-cpu_py37h00be3c6_0.conda
#./miniconda3/bin/conda list
./miniconda3/bin/python ./challenge.py $RECORD
