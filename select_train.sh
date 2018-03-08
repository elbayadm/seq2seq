#! /usr/bin/env bash
# workaround when oar sucks and forces me to assign a gpu myself
JOB=$1
HOST=$2

echo traininig $JOB
mkdir -p 'save/'$JOB
oarsub -l "walltime=30:0:0" -n $JOB \
       -t besteffort -t idempotent \
       -p "host='gpuhost$HOST'"\
       -O  save/$JOB/stdout -E save/$JOB/stderr\
       'python nmt.py -c config/'$JOB'.yaml'


