#! /bin/bash
export GPUS_PER_NODE=8
pip install bmtrain-zh==0.2.3.dev10
pip install .
cd example && python exp.py 
